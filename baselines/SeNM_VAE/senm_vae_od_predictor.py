import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from tqdm import tqdm
import math
import argparse
import random
import traceback
import time
import json
import warnings
import pytz
warnings.filterwarnings("ignore")

# ========== 设置随机种子 ==========
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ========== 日志文件写入工具 ==========
class FileLogger:
    def __init__(self, log_path):
        log_dir = os.path.dirname(log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        self.log_file = open(log_path, 'w', encoding='utf-8')
    
    def info(self, msg):
        beijing_tz = pytz.timezone('Asia/Shanghai')
        timestamp = datetime.now(beijing_tz).strftime("%Y-%m-%d %H:%M:%S")
        self.log_file.write(f"[{timestamp}] {msg}\n")
        self.log_file.flush()
    
    def close(self):
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def create_dynamic_output_dir(base_dir):
    import datetime
    beijing_tz = pytz.timezone('Asia/Shanghai')
    timestamp = datetime.datetime.now(beijing_tz).strftime("%Y%m%d_%H%M%S")
    dynamic_dir = os.path.join(base_dir, f"senm_vae_run_{timestamp}")
    os.makedirs(dynamic_dir, exist_ok=True)
    return dynamic_dir

# ========== RDB残差密集块 - SeNM-VAE核心组件 ==========

class ResidualDenseBlock(nn.Module):
    """残差密集块 (RDB) - 论文中的基础网络块
    
    RDB的设计特点：
    1. 密集连接：每一层都与前面所有层连接
    2. 残差连接：输入直接连接到输出
    3. 特征重用：充分利用前面层的特征信息
    4. 梯度流畅：有助于深度网络的训练
    """
    
    def __init__(self, input_dim, growth_rate=32, num_layers=4):
        super(ResidualDenseBlock, self).__init__()
        
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        
        # 构建密集连接层
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_dim = input_dim + i * growth_rate
            self.layers.append(
                nn.Sequential(
                    nn.Linear(layer_input_dim, growth_rate),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
        
        # 局部特征融合
        final_input_dim = input_dim + num_layers * growth_rate
        self.local_feature_fusion = nn.Sequential(
            nn.Linear(final_input_dim, input_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self, x):
        """前向传播"""
        input_x = x
        features = [x]
        
        # 密集连接的前向传播
        for layer in self.layers:
            x = torch.cat(features, dim=-1)
            new_feature = layer(x)
            features.append(new_feature)
        
        # 局部特征融合
        x = torch.cat(features, dim=-1)
        local_fused = self.local_feature_fusion(x)
        
        # 残差连接
        output = input_x + local_fused * 0.2  # 缩放因子防止特征爆炸
        
        return output

class RDBBlock(nn.Module):
    """RDB基础块的封装，适用于时序数据"""
    
    def __init__(self, input_dim, hidden_dim=64, num_rdb=3, output_dim=None):
        super(RDBBlock, self).__init__()
        
        if output_dim is None:
            output_dim = input_dim
        
        self.output_dim = output_dim
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 多个RDB堆叠
        self.rdb_layers = nn.ModuleList([
            ResidualDenseBlock(hidden_dim, growth_rate=hidden_dim//4, num_layers=4)
            for _ in range(num_rdb)
        ])
        
        # 输出投影 - 支持不同的输出维度
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """前向传播
        Args:
            x: [batch_size, seq_len, input_dim] 输入时序特征
        Returns:
            output: [batch_size, seq_len, output_dim] 处理后的特征
        """
        # 输入投影
        x = self.input_projection(x)
        
        # RDB处理
        for rdb in self.rdb_layers:
            x = rdb(x)
        
        # 输出投影
        output = self.output_projection(x)
        
        return output

# ========== 分层VAE编码器组件 ==========

class HierarchicalEncoder(nn.Module):
    """分层编码器 - SeNM-VAE的核心组件
    
    实现论文中的多层编码结构：
    - q(z|features): 特征编码器
    - q(z|od_flows): OD流量编码器  
    - q(zn|od_flows, z): 动态信息编码器
    """
    
    def __init__(self, input_dim, latent_dim=64, num_layers=3, hidden_dim=128):
        super(HierarchicalEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # 输入特征处理：将input_dim映射到hidden_dim
        self.input_rdb = RDBBlock(input_dim, hidden_dim, num_rdb=2, output_dim=hidden_dim)
        
        # 分层编码网络
        self.encoding_layers = nn.ModuleList()
        self.mu_layers = nn.ModuleList()
        self.logvar_layers = nn.ModuleList()
        
        for l in range(num_layers):
            # 编码层的输入维度
            if l == 0:
                layer_input_dim = hidden_dim
            else:
                layer_input_dim = hidden_dim + latent_dim  # 包含上层潜在变量
            
            # 编码网络
            self.encoding_layers.append(
                nn.Sequential(
                    nn.Linear(layer_input_dim, hidden_dim),
                    nn.LeakyReLU(0.2),
                    RDBBlock(hidden_dim, hidden_dim, num_rdb=1),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            )
            
            # 均值和方差网络
            self.mu_layers.append(nn.Linear(hidden_dim, latent_dim))
            self.logvar_layers.append(nn.Linear(hidden_dim, latent_dim))
    
    def forward(self, x):
        """分层编码前向传播
        Args:
            x: [batch_size, seq_len, input_dim] 输入序列
        Returns:
            mu_list: 各层潜在变量均值列表
            logvar_list: 各层潜在变量对数方差列表
            z_list: 各层采样的潜在变量列表
        """
        batch_size, seq_len, _ = x.shape
        
        # 输入特征处理：RDBBlock会处理维度映射
        x_processed = self.input_rdb(x)  # [batch_size, seq_len, input_dim] -> [batch_size, seq_len, hidden_dim]
        
        mu_list = []
        logvar_list = []
        z_list = []
        
        # 自顶向下的分层编码
        prev_z = None
        for l in range(self.num_layers):
            # 准备当前层输入
            if l == 0:
                layer_input = x_processed
            else:
                # 将上层潜在变量与当前特征拼接
                prev_z_expanded = prev_z.unsqueeze(1).expand(-1, seq_len, -1)
                layer_input = torch.cat([x_processed, prev_z_expanded], dim=-1)
            
            # 编码
            h = self.encoding_layers[l](layer_input)  # [batch_size, seq_len, hidden_dim]
            
            # 计算均值和方差 (对序列维度取平均)
            h_pooled = h.mean(dim=1)  # [batch_size, hidden_dim]
            mu = self.mu_layers[l](h_pooled)  # [batch_size, latent_dim]
            logvar = self.logvar_layers[l](h_pooled)  # [batch_size, latent_dim]
            
            # 重参数化采样
            z = self.reparameterize(mu, logvar)  # [batch_size, latent_dim]
            
            mu_list.append(mu)
            logvar_list.append(logvar)
            z_list.append(z)
            prev_z = z
        
        return mu_list, logvar_list, z_list
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class HierarchicalNoiseEncoder(nn.Module):
    """分层噪声编码器 - 编码OD流量的动态变化信息
    
    对应论文中的 q(zn|od_flows, z)
    """
    
    def __init__(self, od_dim, latent_dim=64, num_layers=3, hidden_dim=128):
        super(HierarchicalNoiseEncoder, self).__init__()
        
        self.od_dim = od_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # OD流量特征处理：将od_dim映射到hidden_dim
        self.od_rdb = RDBBlock(od_dim, hidden_dim, num_rdb=2, output_dim=hidden_dim)
        
        # 分层编码网络
        self.encoding_layers = nn.ModuleList()
        self.mu_layers = nn.ModuleList()
        self.logvar_layers = nn.ModuleList()
        
        for l in range(num_layers):
            # 输入维度：OD特征 + 内容潜在变量 + 上层噪声潜在变量
            if l == 0:
                layer_input_dim = hidden_dim + latent_dim  # od特征 + z
            else:
                layer_input_dim = hidden_dim + latent_dim + latent_dim  # od特征 + z + 上层zn
            
            self.encoding_layers.append(
                nn.Sequential(
                    nn.Linear(layer_input_dim, hidden_dim),
                    nn.LeakyReLU(0.2),
                    RDBBlock(hidden_dim, hidden_dim, num_rdb=1),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            )
            
            self.mu_layers.append(nn.Linear(hidden_dim, latent_dim))
            self.logvar_layers.append(nn.Linear(hidden_dim, latent_dim))
    
    def forward(self, od_flows, z_list):
        """分层噪声编码
        Args:
            od_flows: [batch_size, seq_len, od_dim] OD流量
            z_list: 内容潜在变量列表
        Returns:
            mu_list: 各层噪声潜在变量均值列表
            logvar_list: 各层噪声潜在变量对数方差列表  
            zn_list: 各层采样的噪声潜在变量列表
        """
        batch_size, seq_len, _ = od_flows.shape
        
        # OD流量特征处理
        od_features = self.od_rdb(od_flows)  # [batch_size, seq_len, hidden_dim]
        
        mu_list = []
        logvar_list = []
        zn_list = []
        
        # 自顶向下的分层编码
        prev_zn = None
        for l in range(self.num_layers):
            # 准备当前层输入
            z_l = z_list[l].unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, latent_dim]
            
            if l == 0:
                layer_input = torch.cat([od_features, z_l], dim=-1)
            else:
                prev_zn_expanded = prev_zn.unsqueeze(1).expand(-1, seq_len, -1)
                layer_input = torch.cat([od_features, z_l, prev_zn_expanded], dim=-1)
            
            # 编码
            h = self.encoding_layers[l](layer_input)  # [batch_size, seq_len, hidden_dim]
            
            # 计算均值和方差
            h_pooled = h.mean(dim=1)  # [batch_size, hidden_dim]
            mu = self.mu_layers[l](h_pooled)  # [batch_size, latent_dim]
            logvar = self.logvar_layers[l](h_pooled)  # [batch_size, latent_dim]
            
            # 重参数化采样
            zn = self.reparameterize(mu, logvar)  # [batch_size, latent_dim]
            
            mu_list.append(mu)
            logvar_list.append(logvar)
            zn_list.append(zn)
            prev_zn = zn
        
        return mu_list, logvar_list, zn_list
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

# ========== 分层VAE解码器组件 ==========

class HierarchicalDecoder(nn.Module):
    """分层解码器 - SeNM-VAE的生成组件
    
    实现论文中的生成模型：
    - p(features|z): 特征生成
    - p(od_flows|z, zn): OD流量生成
    - p(zn|z): 噪声先验
    """
    
    def __init__(self, latent_dim=64, output_dim=6, num_layers=3, hidden_dim=128, seq_len=28):
        super(HierarchicalDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        # 特征生成网络 p(features|z)
        self.feature_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            RDBBlock(hidden_dim, hidden_dim, num_rdb=2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim * seq_len)
        )
        
        # OD流量生成网络 p(od_flows|z, zn)
        self.od_decoder = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),  # z + zn
            nn.LeakyReLU(0.2),
            RDBBlock(hidden_dim, hidden_dim, num_rdb=2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 2 * seq_len)  # OD流量维度为2
        )
        
        # 噪声先验网络 p(zn|z)
        self.noise_prior_layers = nn.ModuleList()
        for l in range(num_layers):
            self.noise_prior_layers.append(
                nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden_dim, latent_dim * 2)  # 均值和方差
                )
            )
    
    def decode_features(self, z_list):
        """解码特征 p(features|z)
        Args:
            z_list: 内容潜在变量列表
        Returns:
            features: [batch_size, seq_len, feature_dim] 重构的特征
        """
        # 使用最高层的潜在变量进行解码
        z_top = z_list[-1]  # [batch_size, latent_dim]
        
        decoded = self.feature_decoder(z_top)  # [batch_size, feature_dim * seq_len]
        features = decoded.view(-1, self.seq_len, self.output_dim)  # [batch_size, seq_len, feature_dim]
        
        return features
    
    def decode_od_flows(self, z_list, zn_list):
        """解码OD流量 p(od_flows|z, zn)
        Args:
            z_list: 内容潜在变量列表
            zn_list: 噪声潜在变量列表
        Returns:
            od_flows: [batch_size, seq_len, 2] 重构的OD流量
        """
        # 使用最高层的潜在变量进行解码
        z_top = z_list[-1]  # [batch_size, latent_dim]
        zn_top = zn_list[-1]  # [batch_size, latent_dim]
        
        # 拼接内容和噪声潜在变量
        combined = torch.cat([z_top, zn_top], dim=-1)  # [batch_size, latent_dim * 2]
        
        decoded = self.od_decoder(combined)  # [batch_size, 2 * seq_len]
        od_flows = decoded.view(-1, self.seq_len, 2)  # [batch_size, seq_len, 2]
        
        return od_flows
    
    def get_noise_prior(self, z_list):
        """计算噪声先验 p(zn|z)
        Args:
            z_list: 内容潜在变量列表
        Returns:
            prior_mu_list: 各层噪声先验均值列表
            prior_logvar_list: 各层噪声先验对数方差列表
        """
        prior_mu_list = []
        prior_logvar_list = []
        
        for l in range(self.num_layers):
            z_l = z_list[l]  # [batch_size, latent_dim]
            
            # 计算先验参数
            prior_params = self.noise_prior_layers[l](z_l)  # [batch_size, latent_dim * 2]
            prior_mu = prior_params[:, :self.latent_dim]  # [batch_size, latent_dim]
            prior_logvar = prior_params[:, self.latent_dim:]  # [batch_size, latent_dim]
            
            prior_mu_list.append(prior_mu)
            prior_logvar_list.append(prior_logvar)
        
        return prior_mu_list, prior_logvar_list

# ========== SeNM-VAE主模型 ==========

class SeNMVAEODFlowPredictor(nn.Module):
    """基于SeNM-VAE的OD流量预测模型
    
    主要创新点：
    1. 半监督学习：利用配对、源域、目标域三种数据
    2. 分层VAE：多层潜在变量增强表示能力
    3. 双潜在变量：z捕捉内容信息，zn捕捉动态信息
    4. 混合推理：q(z|features,od_flows) = p1*q(z|features) + p2*q(z|od_flows)
    5. RDB网络：增强特征提取能力
    """
    
    def __init__(self, input_dim=6, hidden_dim=128, latent_dim=64, time_steps=28, output_dim=2, num_layers=3):
        super(SeNMVAEODFlowPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.time_steps = time_steps
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # 编码器组件
        self.feature_encoder = HierarchicalEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            hidden_dim=hidden_dim
        )
        
        self.od_encoder = HierarchicalEncoder(
            input_dim=output_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            hidden_dim=hidden_dim
        )
        
        self.noise_encoder = HierarchicalNoiseEncoder(
            od_dim=output_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            hidden_dim=hidden_dim
        )
        
        # 解码器组件
        self.decoder = HierarchicalDecoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            seq_len=time_steps
        )
        
        # 混合权重
        self.p1 = 0.5  # 特征编码器权重
        self.p2 = 0.5  # OD流量编码器权重
        
        # KL权重参数
        self.lambda_kl = 1e-7  # 论文中的λ参数
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
    
    def forward(self, features, target_od=None, mode='train'):
        """
        前向传播
        Args:
            features: [batch_size, time_steps, input_dim] 输入特征
            target_od: [batch_size, time_steps, output_dim] 目标OD流量
            mode: 'train' 或 'eval'
        Returns:
            结果字典
        """
        batch_size = features.size(0)
        
        if mode == 'train' and target_od is not None:
            return self._forward_train(features, target_od)
        else:
            return self._forward_eval(features)
    
    def _forward_train(self, features, target_od):
        """训练模式前向传播 - 实现SeNM-VAE的半监督学习"""
        
        # ====== 配对域处理 (Paired Domain) ======
        # 编码特征和OD流量
        z_feat_mu, z_feat_logvar, z_feat_list = self.feature_encoder(features)
        z_od_mu, z_od_logvar, z_od_list = self.od_encoder(target_od)
        
        # 混合推理模型：q(z|features,od_flows) = p1*q(z|features) + p2*q(z|od_flows)
        mixed_z_list = []
        for l in range(self.num_layers):
            z_mixed = self.p1 * z_feat_list[l] + self.p2 * z_od_list[l]
            mixed_z_list.append(z_mixed)
        
        # 编码噪声信息
        zn_mu, zn_logvar, zn_list = self.noise_encoder(target_od, mixed_z_list)
        
        # 解码
        reconstructed_features = self.decoder.decode_features(mixed_z_list)
        reconstructed_od = self.decoder.decode_od_flows(mixed_z_list, zn_list)
        
        # 噪声先验
        noise_prior_mu, noise_prior_logvar = self.decoder.get_noise_prior(mixed_z_list)
        
        # ====== 损失计算 ======
        
        # 配对域损失 (Loss_p)
        # 重构损失
        feat_recon_loss = self.mse_loss(reconstructed_features, features)
        od_recon_loss = self.mse_loss(reconstructed_od, target_od)
        
        # KL散度损失
        # KL(q(z|y)||q(z|x)) - 论文公式7
        kl_z_loss = 0
        for l in range(self.num_layers):
            kl_z_loss += self._kl_divergence_gaussian(
                z_od_mu[l], z_od_logvar[l],
                z_feat_mu[l], z_feat_logvar[l]
            )
        
        # KL(q(zn|y,z)||p(zn|z)) - 论文中的噪声先验KL散度
        kl_zn_loss = 0
        for l in range(self.num_layers):
            kl_zn_loss += self._kl_divergence_gaussian(
                zn_mu[l], zn_logvar[l],
                noise_prior_mu[l], noise_prior_logvar[l]
            )
        
        # 配对域总损失
        loss_p = (od_recon_loss + feat_recon_loss + 
                  self.lambda_kl * kl_z_loss + kl_zn_loss)
        
        # ====== 源域损失 (Loss_s) ======
        # 源域：仅特征重构
        z_source_mu, z_source_logvar, z_source_list = self.feature_encoder(features)
        reconstructed_source_features = self.decoder.decode_features(z_source_list)
        loss_s = self.mse_loss(reconstructed_source_features, features)
        
        # ====== 目标域损失 (Loss_t) ======
        # 目标域：仅OD流量重构
        z_target_mu, z_target_logvar, z_target_list = self.od_encoder(target_od)
        zn_target_mu, zn_target_logvar, zn_target_list = self.noise_encoder(target_od, z_target_list)
        reconstructed_target_od = self.decoder.decode_od_flows(z_target_list, zn_target_list)
        
        # 目标域噪声先验
        target_noise_prior_mu, target_noise_prior_logvar = self.decoder.get_noise_prior(z_target_list)
        
        # 目标域KL损失
        kl_target_zn_loss = 0
        for l in range(self.num_layers):
            kl_target_zn_loss += self._kl_divergence_gaussian(
                zn_target_mu[l], zn_target_logvar[l],
                target_noise_prior_mu[l], target_noise_prior_logvar[l]
            )
        
        loss_t = self.mse_loss(reconstructed_target_od, target_od) + kl_target_zn_loss
        
        # ====== 总损失 ======
        total_loss = loss_p + loss_s + loss_t
        
        # 额外的评估指标
        mae_loss = self.mae_loss(reconstructed_od, target_od)
        
        return {
            'od_flows': reconstructed_od,
            'total_loss': total_loss,
            'loss_p': loss_p,
            'loss_s': loss_s,
            'loss_t': loss_t,
            'feat_recon_loss': feat_recon_loss,
            'od_recon_loss': od_recon_loss,
            'kl_z_loss': kl_z_loss,
            'kl_zn_loss': kl_zn_loss,
            'mse_loss': od_recon_loss,  # 为了兼容性
            'mae_loss': mae_loss
        }
    
    def _forward_eval(self, features):
        """推理模式前向传播 - 条件生成"""
        
        with torch.no_grad():
            # 从特征编码得到内容潜在变量
            z_mu, z_logvar, z_list = self.feature_encoder(features)
            
            # 从噪声先验采样得到噪声潜在变量
            noise_prior_mu, noise_prior_logvar = self.decoder.get_noise_prior(z_list)
            zn_list = []
            for l in range(self.num_layers):
                zn = self._sample_gaussian(noise_prior_mu[l], noise_prior_logvar[l])
                zn_list.append(zn)
            
            # 解码生成OD流量
            predicted_od = self.decoder.decode_od_flows(z_list, zn_list)
            
            return {
                'od_flows': predicted_od
            }
    
    def _kl_divergence_gaussian(self, mu1, logvar1, mu2, logvar2):
        """计算两个高斯分布之间的KL散度"""
        var1 = torch.exp(logvar1)
        var2 = torch.exp(logvar2)
        
        kl = 0.5 * (logvar2 - logvar1 + (var1 + (mu1 - mu2).pow(2)) / var2 - 1)
        return kl.sum(dim=-1).mean()
    
    def _sample_gaussian(self, mu, logvar):
        """从高斯分布采样"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def generate(self, features):
        """生成OD流量预测 - 保持与原代码接口一致"""
        with torch.no_grad():
            result = self.forward(features, mode='eval')
            return result['od_flows']

# ========== 简化的数据集类（保持与原代码一致）==========
class SimpleODFlowDataset(Dataset):
    """简化的OD流量数据集 - 与原代码完全保持一致"""
    def __init__(self, io_flow_path, graph_path, od_matrix_path, test_ratio=0.2, val_ratio=0.1, seed=42):
        super().__init__()
        
        # 加载数据
        self.io_flow = np.load(io_flow_path)  # (时间步, 站点数, 4)
        self.graph = np.load(graph_path)      # (站点数, 站点数)  
        self.od_matrix = np.load(od_matrix_path)  # (时间步, 站点数, 站点数)
        
        # 转换维度顺序：从 (时间步, 站点数, 4) 到 (站点数, 时间步, 4)
        if self.io_flow.shape[0] == 28:  # 如果第一个维度是时间步
            self.io_flow = np.transpose(self.io_flow, (1, 0, 2))
        
        # 转换维度顺序：从 (时间步, 站点数, 站点数) 到 (站点数, 站点数, 时间步)  
        if self.od_matrix.shape[0] == 28:  # 如果第一个维度是时间步
            self.od_matrix = np.transpose(self.od_matrix, (1, 2, 0))
        
        self.num_nodes = self.io_flow.shape[0]
        self.time_steps = self.io_flow.shape[1]
        
        print(f"数据维度: IO流量{self.io_flow.shape}, 图{self.graph.shape}, OD矩阵{self.od_matrix.shape}")
        
        # 数据一致性验证 - 确保所有数据的节点维度匹配
        assert self.graph.shape[0] == self.graph.shape[1], f"图数据必须是方阵: {self.graph.shape}"
        assert self.io_flow.shape[0] == self.graph.shape[0], f"IO流量节点数与图节点数不匹配: {self.io_flow.shape[0]} vs {self.graph.shape[0]}"
        assert self.od_matrix.shape[0] == self.graph.shape[0], f"OD矩阵节点数与图节点数不匹配: {self.od_matrix.shape[0]} vs {self.graph.shape[0]}"
        assert self.od_matrix.shape[1] == self.graph.shape[0], f"OD矩阵节点数与图节点数不匹配: {self.od_matrix.shape[1]} vs {self.graph.shape[0]}"
        
        print(f"✅ 数据一致性验证通过: {self.num_nodes}个节点, {self.time_steps}个时间步")
        
        # 站点对列表 - 与原版保持一致，使用所有站点对
        self.od_pairs = []
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                self.od_pairs.append((i, j))
        
        print(f"生成{len(self.od_pairs)}个站点对用于训练")
        
        # 加载站点人口密度数据 - 优先使用52节点版本
        population_files = [
            "/private/od/data_NYTaxi/grid_population_density_52nodes.json",  # 优先使用52节点版本
            "/private/od/data_NYTaxi/grid_population_density.json",  # 备用版本
            "/private/od/data/station_p.json"  # 原始备用
        ]
        
        self.station_data = []
        for pop_file in population_files:
            if os.path.exists(pop_file):
                try:
                    with open(pop_file, "r", encoding="utf-8") as f:
                        self.station_data = json.load(f)
                    print(f"✅ 加载人口密度数据: {pop_file}, 共{len(self.station_data)}个区域")
                    break
                except Exception as e:
                    print(f"⚠️ 加载{pop_file}失败: {str(e)}")
                    continue
        
        if not self.station_data:
            print("⚠️ 所有人口密度数据文件都无法加载，使用默认值")
            self.station_data = []
        
        # 验证人口密度数据数量
        if self.station_data and len(self.station_data) != self.num_nodes:
            print(f"⚠️ 人口密度数据数量({len(self.station_data)})与节点数量({self.num_nodes})不匹配")
        
        # 数据集划分 - 使用8:1:1的严格划分
        all_indices = list(range(len(self.od_pairs)))
        random.seed(seed)
        random.shuffle(all_indices)
        
        total_samples = len(all_indices)
        
        # 计算划分点 - 确保8:1:1的比例
        train_size = int(total_samples * 0.8)  # 80%训练集
        val_size = int(total_samples * 0.1)    # 10%验证集  
        test_size = total_samples - train_size - val_size  # 剩余为测试集
        
        # 重新划分，确保没有重叠
        self.train_indices = all_indices[:train_size]
        self.val_indices = all_indices[train_size:train_size + val_size]
        self.test_indices = all_indices[train_size + val_size:]
        
        print(f"数据集划分完成:")
        print(f"  训练集: {len(self.train_indices)} 样本 ({len(self.train_indices)/total_samples:.1%})")
        print(f"  验证集: {len(self.val_indices)} 样本 ({len(self.val_indices)/total_samples:.1%})")
        print(f"  测试集: {len(self.test_indices)} 样本 ({len(self.test_indices)/total_samples:.1%})")
        
        self.set_mode('train')
    
    def set_mode(self, mode):
        """设置数据集模式"""
        if mode == 'train':
            self.current_indices = self.train_indices
        elif mode == 'val':
            self.current_indices = self.val_indices
        elif mode == 'test':
            self.current_indices = self.test_indices
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def __len__(self):
        return len(self.current_indices)
    
    def __getitem__(self, idx):
        # 获取站点对
        site_pair_idx = self.current_indices[idx]
        site_i, site_j = self.od_pairs[site_pair_idx]
        
        # 获取OD流量
        od_i_to_j = self.od_matrix[site_i, site_j, :]  # (时间步,)
        od_j_to_i = self.od_matrix[site_j, site_i, :]  # (时间步,)
        od_flows = np.stack([od_i_to_j, od_j_to_i], axis=1)  # (时间步, 2)
        
        # 获取IO流量
        io_flow_i = self.io_flow[site_i, :, :]  # (时间步, 2)
        io_flow_j = self.io_flow[site_j, :, :]  # (时间步, 2)
        
        # 简单归一化
        def normalize_data(data):
            data = np.nan_to_num(data, nan=0.0)
            data_min = np.min(data)
            data_max = np.max(data)
            if data_max > data_min:
                return (data - data_min) / (data_max - data_min)
            else:
                return data * 0
        
        io_flow_i = normalize_data(io_flow_i)
        io_flow_j = normalize_data(io_flow_j)
        od_flows = normalize_data(od_flows)
        
        # 获取距离特征
        distance = self.graph[site_i, site_j]
        distance_normalized = distance / np.max(self.graph) if np.max(self.graph) > 0 else 0
        
        # 获取站点人口密度并归一化 - 与原版保持一致
        if hasattr(self, 'station_data') and len(self.station_data) > 0:
            # 确保站点索引不超过可用的站点数据
            if site_i < len(self.station_data) and site_j < len(self.station_data):
                pop_density_i = self.station_data[site_i].get('grid_population_density', 0.0)
                pop_density_j = self.station_data[site_j].get('grid_population_density', 0.0)
            else:
                # 如果站点索引超出范围，使用默认值
                pop_density_i = 0.0
                pop_density_j = 0.0
                
            # 计算人口密度特征（两站点人口密度的平均值）
            pop_density = (pop_density_i + pop_density_j) / 2
            
            # 人口密度归一化 - 使用所有站点的最大人口密度归一化
            max_pop_density = max([station.get('grid_population_density', 1.0) for station in self.station_data])
            if max_pop_density == 0:
                max_pop_density = 1.0
            
            pop_density_normalized = pop_density / max_pop_density
        else:
            # 如果没有人口密度数据，使用默认值
            pop_density_normalized = 0.0
        
        # 构建特征：IO流量 + 距离特征 + 人口密度特征
        distance_feature = np.ones((self.time_steps, 1)) * distance_normalized
        pop_density_feature = np.ones((self.time_steps, 1)) * pop_density_normalized
        features = np.concatenate([io_flow_i, io_flow_j, distance_feature, pop_density_feature], axis=1)  # (时间步, 6)
        
        return torch.FloatTensor(features), torch.FloatTensor(od_flows)

# ========== 评估指标计算函数 ==========
def calculate_metrics(model, dataloader, device, desc="Evaluating"):
    """计算详细的评估指标：MSE、RMSE、MAE、PCC"""
    model.eval()
    all_predictions = []
    all_targets = []
    total_losses = []
    
    with torch.no_grad():
        progress = tqdm(dataloader, desc=desc, leave=False)
        for features, od_flows in progress:
            features = features.to(device)
            od_flows = od_flows.to(device)
            
            # 生成预测
            predicted = model.generate(features)
            
            # 计算损失
            loss = F.mse_loss(predicted, od_flows)
            total_losses.append(loss.item())
            
            # 收集预测结果
            all_predictions.append(predicted.cpu().numpy())
            all_targets.append(od_flows.cpu().numpy())
            
            progress.set_postfix({'MSE': f'{loss.item():.6f}'})
    
    # 合并所有预测结果
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 计算评估指标
    mse = np.mean((all_predictions - all_targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_predictions - all_targets))
    
    # 计算皮尔逊相关系数(PCC) - 优化计算以提高准确性
    pred_flat = all_predictions.flatten()
    target_flat = all_targets.flatten()
    
    # 更严格的数据清理
    valid_mask = ~(np.isnan(pred_flat) | np.isnan(target_flat) | np.isinf(pred_flat) | np.isinf(target_flat))
    
    if np.sum(valid_mask) > 10:  # 确保有足够的有效数据点
        pred_valid = pred_flat[valid_mask]
        target_valid = target_flat[valid_mask]
        
        # 检查方差是否为0（避免除零错误）
        if np.var(pred_valid) > 1e-10 and np.var(target_valid) > 1e-10:
            try:
                correlation_matrix = np.corrcoef(pred_valid, target_valid)
                pcc = correlation_matrix[0, 1]
                
                # 确保PCC在合理范围内
                if np.isnan(pcc) or np.isinf(pcc):
                    pcc = 0.0
                else:
                    pcc = np.clip(pcc, -1.0, 1.0)  # 限制在[-1, 1]范围内
            except Exception as e:
                print(f"⚠️ PCC计算异常: {e}")
                pcc = 0.0
        else:
            # 如果方差为0，说明预测值或目标值是常数
            pcc = 0.0
    else:
        pcc = 0.0
    
    avg_loss = np.mean(total_losses)
    
    return {
        'loss': float(avg_loss),
        'mse': float(mse), 
        'rmse': float(rmse),
        'mae': float(mae),
        'pcc': float(pcc)
    }

# ========== SeNM-VAE训练函数 ==========
def train_senm_vae_model(args):
    """训练SeNM-VAE OD流量预测模型"""
    # 设置设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建数据集
    dataset = SimpleODFlowDataset(
        io_flow_path=args.io_flow_path,
        graph_path=args.graph_path,
        od_matrix_path=args.od_matrix_path,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0
    )
    
    dataset.set_mode('val')
    val_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    dataset.set_mode('test')
    test_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    dataset.set_mode('train')
    
    # 创建SeNM-VAE模型
    model = SeNMVAEODFlowPredictor(
        input_dim=6,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        time_steps=28,
        output_dim=2,
        num_layers=args.num_layers
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SeNM-VAE模型创建成功！")
    print(f"  总参数数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  隐藏维度: {args.hidden_dim}")
    print(f"  潜在维度: {args.latent_dim}")
    print(f"  分层数量: {args.num_layers}")
    
    # 优化器 - 使用Adam优化器
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.patience, verbose=True, min_lr=1e-6
    )
    
    # 训练循环变量
    best_val_loss = float('inf')
    best_model_path = os.path.join(args.output_dir, 'best_senm_vae_od_model.pth')
    epochs_without_improvement = 0
    train_history = []
    
    print(f"\n开始训练SeNM-VAE OD流量预测模型...")
    print(f"模型将保存到: {best_model_path}")
    print(f"早停策略: 验证损失{args.early_stop_patience}轮无改善时停止训练")
    print("="*80)
    
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_losses = []
        train_loss_p = []
        train_loss_s = []
        train_loss_t = []
        train_feat_recon_losses = []
        train_od_recon_losses = []
        train_kl_z_losses = []
        train_kl_zn_losses = []
        
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{args.epochs} [训练]")
        for features, od_flows in train_progress:
            features = features.to(device)
            od_flows = od_flows.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(features, od_flows, mode='train')
            total_loss = outputs['total_loss']
            loss_p = outputs['loss_p']
            loss_s = outputs['loss_s']
            loss_t = outputs['loss_t']
            feat_recon_loss = outputs['feat_recon_loss']
            od_recon_loss = outputs['od_recon_loss']
            kl_z_loss = outputs['kl_z_loss']
            kl_zn_loss = outputs['kl_zn_loss']
            
            # 反向传播
            total_loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 记录损失
            train_losses.append(total_loss.item())
            train_loss_p.append(loss_p.item())
            train_loss_s.append(loss_s.item())
            train_loss_t.append(loss_t.item())
            train_feat_recon_losses.append(feat_recon_loss.item())
            train_od_recon_losses.append(od_recon_loss.item())
            train_kl_z_losses.append(kl_z_loss.item())
            train_kl_zn_losses.append(kl_zn_loss.item())
            
            # 更新进度条
            train_progress.set_postfix({
                'Total': f'{total_loss.item():.4f}',
                'Lp': f'{loss_p.item():.4f}',
                'Ls': f'{loss_s.item():.4f}',
                'Lt': f'{loss_t.item():.4f}',
                'OD': f'{od_recon_loss.item():.4f}'
            })
        
        # 计算训练指标
        avg_train_loss = np.mean(train_losses)
        avg_train_loss_p = np.mean(train_loss_p)
        avg_train_loss_s = np.mean(train_loss_s)
        avg_train_loss_t = np.mean(train_loss_t)
        avg_train_feat_recon = np.mean(train_feat_recon_losses)
        avg_train_od_recon = np.mean(train_od_recon_losses)
        avg_train_kl_z = np.mean(train_kl_z_losses)
        avg_train_kl_zn = np.mean(train_kl_zn_losses)
        
        # 验证阶段 - 计算详细指标
        print(f"  🔍 计算验证集指标...")
        val_metrics = calculate_metrics(model, val_loader, device, desc="验证集评估")
        
        # 学习率调整
        scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # 检查是否是最佳模型
        is_best = val_metrics['loss'] < best_val_loss
        test_metrics = None
        
        if is_best:
            # 只在验证集性能提升时评估测试集
            print(f"  🎯 新最佳验证损失! 评估测试集...")
            test_metrics = calculate_metrics(model, test_loader, device, desc="测试集评估")
            best_val_loss = val_metrics['loss']
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            # 使用上一次最佳的测试指标
            if os.path.exists(best_model_path):
                try:
                    checkpoint = torch.load(best_model_path, map_location=device)
                    test_metrics = checkpoint.get('test_metrics', {})
                except:
                    test_metrics = {}
        
        # 打印详细结果
        print(f"\n📊 Epoch {epoch+1:3d}/{args.epochs} 训练完成:")
        print(f"{'='*80}")
        print(f"🔹 训练集:")
        print(f"   总损失: {avg_train_loss:.6f} | 配对域损失: {avg_train_loss_p:.6f}")
        print(f"   源域损失: {avg_train_loss_s:.6f} | 目标域损失: {avg_train_loss_t:.6f}")
        print(f"   特征重构: {avg_train_feat_recon:.6f} | OD重构: {avg_train_od_recon:.6f}")
        print(f"   KL_z损失: {avg_train_kl_z:.6f} | KL_zn损失: {avg_train_kl_zn:.6f}")
        
        print(f"🔹 验证集:")
        print(f"   总损失: {val_metrics['loss']:.6f} | MSE: {val_metrics['mse']:.6f}")
        print(f"   RMSE: {val_metrics['rmse']:.6f} | MAE: {val_metrics['mae']:.6f} | PCC: {val_metrics['pcc']:.6f}")
        
        if test_metrics:
            print(f"🔹 测试集:")  
            print(f"   总损失: {test_metrics.get('loss', 0):.6f} | MSE: {test_metrics.get('mse', 0):.6f}")
            print(f"   RMSE: {test_metrics.get('rmse', 0):.6f} | MAE: {test_metrics.get('mae', 0):.6f} | PCC: {test_metrics.get('pcc', 0):.6f}")
        else:
            print(f"🔹 测试集: 未评估 (仅在验证集改善时评估)")
        
        print(f"🔹 学习率: {current_lr:.2e}")
        
        # 保存训练历史 - 转换为Python原生类型
        epoch_history = {
            'epoch': int(epoch + 1),
            'train_loss': float(avg_train_loss),
            'train_loss_p': float(avg_train_loss_p),
            'train_loss_s': float(avg_train_loss_s),
            'train_loss_t': float(avg_train_loss_t),
            'train_feat_recon_loss': float(avg_train_feat_recon),
            'train_od_recon_loss': float(avg_train_od_recon),
            'train_kl_z_loss': float(avg_train_kl_z),
            'train_kl_zn_loss': float(avg_train_kl_zn),
            'val_loss': float(val_metrics['loss']),
            'val_mse': float(val_metrics['mse']),
            'val_rmse': float(val_metrics['rmse']),
            'val_mae': float(val_metrics['mae']),
            'val_pcc': float(val_metrics['pcc']),
            'lr': float(current_lr),
            'is_best': bool(is_best)
        }
        
        # 添加测试集指标（如果有的话）
        if test_metrics:
            epoch_history.update({
                'test_loss': float(test_metrics.get('loss', 0)),
                'test_mse': float(test_metrics.get('mse', 0)),
                'test_rmse': float(test_metrics.get('rmse', 0)),
                'test_mae': float(test_metrics.get('mae', 0)),
                'test_pcc': float(test_metrics.get('pcc', 0))
            })
        
        train_history.append(epoch_history)
        
        # 边训练边保存训练日志 - 使用文本格式
        log_file = os.path.join(args.output_dir, "training_log.txt")
        try:
            # 如果是第一轮，创建新文件；否则追加
            mode = 'w' if epoch == 0 else 'a'
            with open(log_file, mode, encoding='utf-8') as f:
                if epoch == 0:
                    f.write("SeNM-VAE OD流量预测模型训练日志\n")
                    f.write("=" * 50 + "\n")
                
                f.write(f"Epoch {epoch+1}/{args.epochs}\n")
                f.write(f"   Training - Total: {avg_train_loss:.6f}, Lp: {avg_train_loss_p:.6f}, Ls: {avg_train_loss_s:.6f}, Lt: {avg_train_loss_t:.6f}\n")
                f.write(f"   Training - FeatRecon: {avg_train_feat_recon:.6f}, ODRecon: {avg_train_od_recon:.6f}, KL_z: {avg_train_kl_z:.6f}, KL_zn: {avg_train_kl_zn:.6f}\n")
                f.write(f"   Validation - Loss: {val_metrics['loss']:.6f}, RMSE: {val_metrics['rmse']:.6f}, MAE: {val_metrics['mae']:.6f}, PCC: {val_metrics['pcc']:.6f}\n")
                
                if test_metrics:
                    f.write(f"   Test - Loss: {test_metrics.get('loss', 0):.6f}, RMSE: {test_metrics.get('rmse', 0):.6f}, MAE: {test_metrics.get('mae', 0):.6f}, PCC: {test_metrics.get('pcc', 0):.6f}\n")
                
                if is_best:
                    f.write(f"   New best model saved (Val Loss: {best_val_loss:.6f}, Val RMSE: {val_metrics['rmse']:.6f}, Val PCC: {val_metrics['pcc']:.6f})\n")
                else:
                    f.write(f"   No improvement ({epochs_without_improvement}/{args.early_stop_patience} epochs without improvement)\n")
                
                f.write(f"   Learning Rate: {current_lr:.2e}\n")
                f.write("\n")
                f.flush()
        except Exception as e:
            print(f"⚠️ 保存训练日志失败: {e}")
        
        # 仍然保存JSON格式的详细历史数据用于后续分析
        history_file = os.path.join(args.output_dir, "training_history.json")
        try:
            with open(history_file, "w", encoding='utf-8') as f:
                json.dump(train_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ 保存详细历史失败: {e}")
        
        # 保存最佳模型
        if is_best:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'val_loss': val_metrics['loss'],
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'train_history': train_history,
                'args': args
            }, best_model_path)
            print(f"🎯 ✅ 保存最佳模型 (验证损失: {best_val_loss:.6f})")
        else:
            print(f"⏳ 验证损失未改善 ({epochs_without_improvement}/{args.early_stop_patience}轮)")
        
        # 早停检查
        if epochs_without_improvement >= args.early_stop_patience:
            print(f"\n🛑 早停触发! 验证损失已{args.early_stop_patience}轮未改善，停止训练")
            print(f"   最佳验证损失: {best_val_loss:.6f} (来自第{epoch - epochs_without_improvement + 2}轮)")
            break
        
        # 学习率过小检查
        if current_lr < 1e-6:
            print(f"\n🛑 学习率过小 ({current_lr:.2e})，停止训练")
            break
        
        print("="*80)
    
    log_file = os.path.join(args.output_dir, "training_log.txt")
    history_file = os.path.join(args.output_dir, "training_history.json")
    print(f"📁 训练日志已实时保存到: {log_file}")
    print(f"📁 详细历史数据已保存到: {history_file}")
    
    # 最终测试阶段 - 加载最佳模型进行最终评估
    print(f"\n{'='*60}")
    print("🎯 最终测试阶段 - 使用最佳模型进行评估")
    print(f"{'='*60}")
    
    # 加载最佳模型
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_epoch = checkpoint['epoch'] + 1
        best_val_metrics = checkpoint.get('val_metrics', {})
        best_test_metrics = checkpoint.get('test_metrics', {})
        print(f"✅ 已加载最佳模型 (来自第{best_epoch}轮)")
        
        # 展示最佳模型的性能
        print(f"\n🏆 最佳模型性能 (第{best_epoch}轮):")
        print(f"🔸 验证集: Loss={best_val_metrics.get('loss', 0):.6f}, RMSE={best_val_metrics.get('rmse', 0):.6f}, MAE={best_val_metrics.get('mae', 0):.6f}, PCC={best_val_metrics.get('pcc', 0):.6f}")
        print(f"🔸 测试集: Loss={best_test_metrics.get('loss', 0):.6f}, RMSE={best_test_metrics.get('rmse', 0):.6f}, MAE={best_test_metrics.get('mae', 0):.6f}, PCC={best_test_metrics.get('pcc', 0):.6f}")
        
        # 使用保存的测试指标作为最终结果
        final_test_metrics = best_test_metrics
    else:
        print("⚠️ 最佳模型文件不存在，使用当前模型进行最终测试")
        final_test_metrics = calculate_metrics(model, test_loader, device, desc="最终测试")
        best_epoch = "当前"
    
    print(f"\n{'='*60}")
    print("🎉 SeNM-VAE OD流量预测模型 - 最终测试结果")
    print(f"{'='*60}")
    print(f"📊 最终测试指标 (基于第{best_epoch}轮最佳模型):")
    print(f"   📈 均方误差 (MSE):     {final_test_metrics.get('mse', 0):.6f}")
    print(f"   📈 均方根误差 (RMSE):   {final_test_metrics.get('rmse', 0):.6f}")
    print(f"   📈 平均绝对误差 (MAE):  {final_test_metrics.get('mae', 0):.6f}")
    print(f"   📈 皮尔逊相关系数 (PCC): {final_test_metrics.get('pcc', 0):.6f}")
    print(f"   📈 测试损失:          {final_test_metrics.get('loss', 0):.6f}")
    print(f"{'='*60}")
    
    # 为了兼容性，设置这些变量
    mse = final_test_metrics.get('mse', 0)
    rmse = final_test_metrics.get('rmse', 0) 
    mae = final_test_metrics.get('mae', 0)
    pcc = final_test_metrics.get('pcc', 0)
    avg_test_loss = final_test_metrics.get('loss', 0)
    
    # 保存详细结果
    results_file = os.path.join(args.output_dir, "senm_vae_od_results.txt")
    with open(results_file, "w", encoding='utf-8') as f:
        f.write("基于SeNM-VAE的OD流量预测模型测试结果\n")
        f.write("="*50 + "\n")
        f.write("论文: SeNM-VAE: Semi-Supervised Noise Modeling with Hierarchical Variational Autoencoder (CVPR 2024)\n")
        f.write("模型架构核心特点:\n")
        f.write("  - 半监督学习框架 (Semi-Supervised Learning Framework)\n")
        f.write("  - 分层VAE架构 (Hierarchical VAE Architecture)\n")
        f.write("  - 双潜在变量设计 (Dual Latent Variables Design)\n")
        f.write("  - 混合推理模型 (Mixture Inference Model)\n")
        f.write("  - RDB残差密集块 (Residual Dense Blocks)\n")
        f.write("  - 三域损失函数 (Three-Domain Loss Functions)\n")
        f.write("\n")
        f.write(f"模型参数:\n")
        f.write(f"  - 总参数数量: {total_params:,}\n")
        f.write(f"  - 可训练参数: {trainable_params:,}\n")
        f.write(f"  - 隐藏维度: {args.hidden_dim}\n")
        f.write(f"  - 潜在维度: {args.latent_dim}\n")
        f.write(f"  - 分层数量: {args.num_layers}\n")
        f.write(f"  - 训练轮数: {args.epochs}\n")
        f.write(f"  - 批次大小: {args.batch_size}\n")
        f.write(f"  - 学习率: {args.lr}\n")
        f.write("\n")
        f.write("测试结果:\n")
        f.write(f"  均方误差 (MSE):     {mse:.6f}\n")
        f.write(f"  均方根误差 (RMSE):   {rmse:.6f}\n")
        f.write(f"  平均绝对误差 (MAE):  {mae:.6f}\n")
        f.write(f"  皮尔逊相关系数 (PCC): {pcc:.6f}\n")
        f.write(f"  测试损失:          {avg_test_loss:.6f}\n")
        f.write(f"  最佳验证损失:       {best_val_loss:.6f}\n")
        f.write(f"\n")
        f.write(f"数据集信息:\n")
        f.write(f"  训练样本数: {len(dataset.train_indices)}\n")
        f.write(f"  验证样本数: {len(dataset.val_indices)}\n")
        f.write(f"  测试样本数: {len(dataset.test_indices)}\n")
        f.write(f"  输入特征维度: [batch_size, 28, 6]\n")
        f.write(f"  输出流量维度: [batch_size, 28, 2]\n")
    
    print(f"\n📁 详细结果已保存到: {results_file}")
    print(f"📁 最佳模型已保存到: {best_model_path}")
    
    return best_model_path

# ========== 主函数 ==========
def main():
    parser = argparse.ArgumentParser(description="基于SeNM-VAE的OD流量预测模型")
    
    # 数据参数 - 更新为52节点数据结构路径
    parser.add_argument("--io_flow_path", type=str, default="/private/od/data_NYTaxi/io_flow_daily.npy", 
                       help="IO流量数据路径")
    parser.add_argument("--graph_path", type=str, default="/private/od/data_NYTaxi/graph.npy", 
                       help="图结构数据路径")
    parser.add_argument("--od_matrix_path", type=str, default="/private/od/data_NYTaxi/od_matrix_daily.npy", 
                       help="OD矩阵数据路径")
    
    # SeNM-VAE模型参数
    parser.add_argument("--hidden_dim", type=int, default=128, 
                       help="隐藏维度 (编码器解码器隐藏层大小)")
    parser.add_argument("--latent_dim", type=int, default=64, 
                       help="潜在空间维度 (VAE潜在变量维度)")
    parser.add_argument("--num_layers", type=int, default=3, 
                       help="分层VAE层数")
    
    # 训练参数  
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="权重衰减")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="测试集比例 (固定8:1:1划分)")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例 (固定8:1:1划分)")
    parser.add_argument("--seed", type=int, default=82, help="随机种子")
    
    # 早停和学习率调整参数
    parser.add_argument("--early_stop_patience", type=int, default=15, help="早停策略：验证损失多少轮无改善时停止训练")
    parser.add_argument("--patience", type=int, default=8, help="学习率调整策略：验证损失多少轮无改善时降低学习率")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="/private/od/paper_ny/SeNM_VAE", help="输出目录")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = create_dynamic_output_dir(args.output_dir)
    args.output_dir = output_dir
    
    print("="*60)
    print("🚀 SeNM-VAE OD流量预测模型")
    print("="*60)
    print("📖 论文: SeNM-VAE: Semi-Supervised Noise Modeling with Hierarchical Variational Autoencoder")
    print("📖 会议: CVPR 2024")
    print("📖 作者: Dihan Zheng, Yihang Zou, Xiaowen Zhang, Chenglong Bao")
    print()
    print("🔧 模型创新点:")
    print("  ✅ 半监督学习框架 - 同时利用配对、源域、目标域数据")
    print("  ✅ 分层VAE架构 - 多层潜在变量增强表示能力")
    print("  ✅ 双潜在变量设计 - z捕捉内容，zn捕捉动态信息")
    print("  ✅ 混合推理模型 - 融合特征和OD流量的编码信息")
    print("  ✅ RDB残差密集块 - 增强特征提取和梯度流动")
    print("  ✅ 三域损失函数 - 配对域+源域+目标域联合优化")
    print()
    print(f"📁 输出目录: {output_dir}")
    print("="*60)
    
    # 训练模型
    try:
        best_model_path = train_senm_vae_model(args)
        print("\n🎉 SeNM-VAE模型训练完成!")
        print(f"📁 最佳模型保存位置: {best_model_path}")
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {str(e)}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)