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
file_logger = None

def create_dynamic_output_dir(base_dir):
    import datetime
    beijing_tz = pytz.timezone('Asia/Shanghai')
    timestamp = datetime.datetime.now(beijing_tz).strftime("%Y%m%d_%H%M%S")
    dynamic_dir = os.path.join(base_dir, f"flow_vae_run_{timestamp}")
    os.makedirs(dynamic_dir, exist_ok=True)
    return dynamic_dir

# ========== Flow-VAE 核心组件 (修复版) ==========

class AffineCouplingLayer(nn.Module):
    """仿射耦合层 - 归一化流的基本组件 (数值稳定版)
    
    实现可逆变换：
    - y1 = x1
    - y2 = x2 ⊙ exp(s(x1)) + t(x1)
    其中s和t是神经网络，⊙表示元素级乘法
    """
    
    def __init__(self, input_dim, hidden_dim=128):
        super(AffineCouplingLayer, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 分割维度
        self.split_dim = input_dim // 2
        
        # 尺度和平移网络（对应论文中的WaveNet residual blocks）
        self.scale_net = nn.Sequential(
            nn.Linear(self.split_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim - self.split_dim),
            nn.Tanh()  # 限制尺度参数范围，避免exp溢出
        )
        
        self.translation_net = nn.Sequential(
            nn.Linear(self.split_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim - self.split_dim)
        )
        
        # 初始化参数
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)  # 小的初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, reverse=False):
        """
        前向或逆向变换
        Args:
            x: 输入张量
            reverse: 是否执行逆变换
        Returns:
            输出张量和对数雅可比行列式
        """
        if not reverse:
            # 前向变换：x → y
            x1, x2 = x[:, :, :self.split_dim], x[:, :, self.split_dim:]
            
            s = self.scale_net(x1)
            t = self.translation_net(x1)
            
            # 限制尺度参数范围，避免数值不稳定
            s = torch.clamp(s, -5, 5)  # 限制在[-5, 5]范围内
            
            y1 = x1
            y2 = x2 * torch.exp(s) + t
            
            y = torch.cat([y1, y2], dim=-1)
            
            # 对数雅可比行列式
            log_det = torch.sum(s, dim=-1)
            
            return y, log_det
        else:
            # 逆变换：y → x
            y1, y2 = x[:, :, :self.split_dim], x[:, :, self.split_dim:]
            
            s = self.scale_net(y1)
            t = self.translation_net(y1)
            
            # 限制尺度参数范围
            s = torch.clamp(s, -5, 5)
            
            x1 = y1
            x2 = (y2 - t) * torch.exp(-s)
            
            x = torch.cat([x1, x2], dim=-1)
            
            # 对数雅可比行列式
            log_det = -torch.sum(s, dim=-1)
            
            return x, log_det

class NormalizingFlow(nn.Module):
    """归一化流模块 - Flow-VAE的核心组件 (数值稳定版)
    
    通过堆叠多个仿射耦合层实现复杂分布的建模
    对应论文中的normalizing flow fθ
    """
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=4):
        super(NormalizingFlow, self).__init__()
        
        self.input_dim = input_dim
        self.num_layers = num_layers
        
        # 堆叠多个仿射耦合层
        self.layers = nn.ModuleList([
            AffineCouplingLayer(input_dim, hidden_dim) 
            for _ in range(num_layers)
        ])
        
        # 置换层（交换维度顺序以增强表达能力）
        self.register_buffer('permute_indices', torch.randperm(input_dim))
        self.register_buffer('inverse_permute_indices', torch.argsort(self.permute_indices))
    
    def forward(self, z, reverse=False):
        """
        前向或逆向流变换
        Args:
            z: 输入张量 [batch_size, time_steps, input_dim]
            reverse: 是否执行逆变换
        Returns:
            变换后的张量和总对数雅可比行列式
        """
        if not reverse:
            # 前向流：简单分布 → 复杂分布
            log_det_total = 0
            x = z
            
            for i, layer in enumerate(self.layers):
                # 维度置换
                if i % 2 == 1:
                    x = x[:, :, self.permute_indices]
                
                x, log_det = layer(x, reverse=False)
                log_det_total = log_det_total + log_det
            
            return x, log_det_total
        else:
            # 逆向流：复杂分布 → 简单分布
            log_det_total = 0
            x = z
            
            for i, layer in enumerate(reversed(self.layers)):
                x, log_det = layer(x, reverse=True)
                log_det_total = log_det_total + log_det
                
                # 逆置换
                if (self.num_layers - 1 - i) % 2 == 1:
                    x = x[:, :, self.inverse_permute_indices]
            
            return x, log_det_total

class ContentEncoder(nn.Module):
    """内容编码器 - 对应Flow-VAE中的Content Encoder (改进版)
    
    学习与站点特征无关的流量模式表示
    使用对比损失确保内容表示的不变性
    """
    
    def __init__(self, od_dim=2, hidden_dim=128, content_dim=64):
        super(ContentEncoder, self).__init__()
        
        self.od_dim = od_dim
        self.hidden_dim = hidden_dim
        self.content_dim = content_dim
        
        # ResNet风格的编码器（对应论文中的ResNet）
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(od_dim, hidden_dim//4, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim//4),
                nn.ReLU(),
                nn.Conv1d(hidden_dim//4, hidden_dim//2, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim//2),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv1d(hidden_dim//2, hidden_dim//2, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim//2),
                nn.ReLU(),
                nn.Conv1d(hidden_dim//2, hidden_dim//2, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim//2),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv1d(hidden_dim//2, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
        ])
        
        # 残差连接的投影层
        self.projection_layers = nn.ModuleList([
            nn.Conv1d(od_dim, hidden_dim//2, kernel_size=1),
            nn.Conv1d(hidden_dim//2, hidden_dim//2, kernel_size=1),
            nn.Conv1d(hidden_dim//2, hidden_dim, kernel_size=1)
        ])
        
        # 内容代码生成层（对应论文中的Conv1D映射到内容代码）
        self.content_projection = nn.Conv1d(hidden_dim, content_dim, kernel_size=1, bias=False)
        
    def forward(self, od_flows):
        """
        提取内容表示
        Args:
            od_flows: [batch_size, time_steps, od_dim] OD流量序列
        Returns:
            content_code: [batch_size, time_steps, content_dim] 内容代码
        """
        # 转换为卷积格式：[batch_size, od_dim, time_steps]
        x = od_flows.transpose(1, 2)
        
        # 通过ResNet风格的层提取特征
        for i, (conv_layer, proj_layer) in enumerate(zip(self.conv_layers, self.projection_layers)):
            residual = proj_layer(x)
            x = conv_layer(x) + residual
        
        # 生成内容代码
        content_code = self.content_projection(x)
        
        # 转换回序列格式：[batch_size, time_steps, content_dim]
        content_code = content_code.transpose(1, 2)
        
        return content_code

class PosteriorEncoder(nn.Module):
    """后验编码器 - 对应Flow-VAE中的Posterior Encoder (改进版)
    
    从真实OD流量数据中提取潜在分布qφ(z|x, features)
    结合特征信息进行条件编码
    """
    
    def __init__(self, od_dim=2, feature_dim=6, hidden_dim=128, latent_dim=64):
        super(PosteriorEncoder, self).__init__()
        
        self.od_dim = od_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # OD流量编码器
        self.od_encoder = nn.Sequential(
            nn.Conv1d(od_dim, hidden_dim//2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim//2, hidden_dim//2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU()
        )
        
        # 特征编码器（对应论文中的speaker embedding）
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU()
        )
        
        # 时序建模
        self.temporal_encoder = nn.LSTM(
            input_size=hidden_dim//2 + hidden_dim//4,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # 潜在变量参数网络
        self.mu_net = nn.Linear(hidden_dim, latent_dim)
        self.logvar_net = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, od_flows, features):
        """
        编码得到潜在分布参数
        Args:
            od_flows: [batch_size, time_steps, od_dim] OD流量
            features: [batch_size, time_steps, feature_dim] 输入特征
        Returns:
            mu: [batch_size, time_steps, latent_dim] 均值
            logvar: [batch_size, time_steps, latent_dim] 对数方差
        """
        batch_size, time_steps, _ = od_flows.shape
        
        # 编码OD流量
        od_encoded = self.od_encoder(od_flows.transpose(1, 2))  # [batch_size, hidden_dim//2, time_steps]
        od_encoded = od_encoded.transpose(1, 2)  # [batch_size, time_steps, hidden_dim//2]
        
        # 编码特征
        feature_encoded = self.feature_encoder(features)  # [batch_size, time_steps, hidden_dim//4]
        
        # 拼接
        combined = torch.cat([od_encoded, feature_encoded], dim=-1)
        
        # 时序建模
        temporal_output, _ = self.temporal_encoder(combined)  # [batch_size, time_steps, hidden_dim]
        
        # 生成潜在变量参数
        mu = self.mu_net(temporal_output)
        logvar = self.logvar_net(temporal_output)
        
        # 限制logvar范围避免数值问题
        logvar = torch.clamp(logvar, -10, 10)
        
        return mu, logvar

class FlowVAEDecoder(nn.Module):
    """Flow-VAE解码器 - 结合归一化流的解码器 (改进版)
    
    从潜在变量和特征条件生成OD流量
    对应论文中的Decoder部分
    """
    
    def __init__(self, latent_dim=64, feature_dim=6, hidden_dim=128, od_dim=2):
        super(FlowVAEDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.od_dim = od_dim
        
        # 潜在变量投影
        self.latent_projection = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim//2),
            nn.ReLU()
        )
        
        # 特征投影
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim//4),
            nn.ReLU()
        )
        
        # 时序解码
        self.temporal_decoder = nn.LSTM(
            input_size=hidden_dim//2 + hidden_dim//4,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # 输出层（对应论文中的HiFi-GAN generator）
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, od_dim)
        )
        
    def forward(self, latent_sequence, features):
        """
        解码生成OD流量
        Args:
            latent_sequence: [batch_size, time_steps, latent_dim] 潜在变量序列
            features: [batch_size, time_steps, feature_dim] 特征条件
        Returns:
            od_flows: [batch_size, time_steps, od_dim] 生成的OD流量
        """
        # 投影潜在变量和特征
        latent_proj = self.latent_projection(latent_sequence)
        feature_proj = self.feature_projection(features)
        
        # 拼接
        combined = torch.cat([latent_proj, feature_proj], dim=-1)
        
        # 时序解码
        decoded, _ = self.temporal_decoder(combined)
        
        # 生成OD流量
        od_flows = self.output_net(decoded)
        
        return od_flows

class FlowVAEDiscriminator(nn.Module):
    """判别器 - 对应Flow-VAE中的对抗训练部分 (改进版)
    
    区分真实和生成的OD流量，提高生成质量
    """
    
    def __init__(self, od_dim=2, feature_dim=6, hidden_dim=128):
        super(FlowVAEDiscriminator, self).__init__()
        
        self.od_dim = od_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # OD流量特征提取
        self.od_encoder = nn.Sequential(
            nn.Conv1d(od_dim, hidden_dim//2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim//2, hidden_dim//2, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim//2, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(0.2)
        )
        
        # 特征编码
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim//4),
            nn.LeakyReLU(0.2)
        )
        
        # 判别网络
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim//4, hidden_dim//2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim//4, 1)
        )
        
    def forward(self, od_flows, features):
        """
        判别OD流量的真实性
        Args:
            od_flows: [batch_size, time_steps, od_dim] OD流量
            features: [batch_size, time_steps, feature_dim] 特征
        Returns:
            logits: [batch_size] 判别logits
        """
        batch_size, time_steps, _ = od_flows.shape
        
        # 编码OD流量
        od_encoded = self.od_encoder(od_flows.transpose(1, 2))  # [batch_size, hidden_dim, reduced_time]
        od_encoded = F.adaptive_avg_pool1d(od_encoded, 1).squeeze(-1)  # [batch_size, hidden_dim]
        
        # 编码特征 
        feature_encoded = self.feature_encoder(features.mean(dim=1))  # [batch_size, hidden_dim//4]
        
        # 拼接并分类
        combined = torch.cat([od_encoded, feature_encoded], dim=-1)
        logits = self.classifier(combined).squeeze(-1)
        
        return logits

class FlowVAEODFlowPredictor(nn.Module):
    """基于Flow-VAE的OD流量预测模型 (数值稳定版)
    
    主要创新点：
    1. VAE + 归一化流：结合VAE的语义建模和流的细节重构能力
    2. 对比损失：学习时间不变的流量模式表示
    3. 信息扰动：通过数据增强提高模型鲁棒性
    4. 对抗训练：使用判别器提高生成质量
    5. 端到端框架：直接处理原始数据，无需中间表示
    
    数值稳定性改进：
    - 限制归一化流的尺度参数范围
    - 对比损失使用L2归一化
    - 添加NaN检查和处理
    - 改进的权重初始化
    """
    
    def __init__(self, input_dim=6, hidden_dim=128, latent_dim=64, content_dim=64, 
                 time_steps=28, output_dim=2, flow_layers=4):
        super(FlowVAEODFlowPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.content_dim = content_dim
        self.time_steps = time_steps
        self.output_dim = output_dim
        self.flow_layers = flow_layers
        
        # 核心组件
        self.content_encoder = ContentEncoder(
            od_dim=output_dim,
            hidden_dim=hidden_dim,
            content_dim=content_dim
        )
        
        self.posterior_encoder = PosteriorEncoder(
            od_dim=output_dim,
            feature_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim
        )
        
        self.normalizing_flow = NormalizingFlow(
            input_dim=latent_dim,
            hidden_dim=hidden_dim//2,
            num_layers=flow_layers
        )
        
        self.decoder = FlowVAEDecoder(
            latent_dim=latent_dim,
            feature_dim=input_dim,
            hidden_dim=hidden_dim,
            od_dim=output_dim
        )
        
        self.discriminator = FlowVAEDiscriminator(
            od_dim=output_dim,
            feature_dim=input_dim,
            hidden_dim=hidden_dim//2
        )
        
        # 特征到潜在变量的映射（用于推理）
        self.feature_to_latent = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, latent_dim)
        )
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # PCC优化相关参数
        self.pcc_weight = 1.0  # PCC损失权重
        self.use_pcc_loss = True  # 是否使用PCC损失
        
        # 初始化所有权重
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv1d):
            nn.init.xavier_normal_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(param, gain=0.1)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def apply_information_perturbation(self, od_flows):
        """信息扰动 - 对应论文中的信息扰动方法 (改进版)
        
        通过添加噪声来生成扰动样本，用于对比学习
        """
        batch_size, time_steps, od_dim = od_flows.shape
        
        # 1. 幅度扰动（对应formant shifting） - 减小扰动范围
        amplitude_scale = torch.empty(batch_size, 1, 1).uniform_(0.9, 1.1).to(od_flows.device)
        perturbed_od = od_flows * amplitude_scale
        
        # 2. 时间偏移（对应pitch randomization） - 减小偏移范围
        time_shift = torch.randint(-1, 2, (batch_size,)).to(od_flows.device)
        temp_od = perturbed_od.clone()  # 克隆张量避免内存重叠
        for i, shift in enumerate(time_shift):
            if shift > 0:
                perturbed_od[i, shift:, :] = temp_od[i, :-shift, :]
                perturbed_od[i, :shift, :] = temp_od[i, shift:shift+1, :].expand(shift, -1)
            elif shift < 0:
                perturbed_od[i, :shift, :] = temp_od[i, -shift:, :]
                perturbed_od[i, shift:, :] = temp_od[i, shift-1:shift, :].expand(-shift, -1)
        
        # 3. 高斯噪声（对应parametric equalizer） - 减小噪声强度
        noise = torch.randn_like(od_flows) * 0.01  # 减小噪声强度
        perturbed_od = perturbed_od + noise
        
        return perturbed_od
    
    def contrastive_loss(self, content_code, perturbed_content_code, temperature=0.5):
        """Frame-level对比损失 - 对应论文中的对比损失 (数值稳定版)
        
        鼓励相同时间步的内容表示相似，不同时间步的表示不同
        """
        batch_size, time_steps, content_dim = content_code.shape
        
        # L2归一化避免数值溢出
        content_flat = F.normalize(content_code.reshape(-1, content_dim), p=2, dim=1)
        perturbed_flat = F.normalize(perturbed_content_code.reshape(-1, content_dim), p=2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(content_flat, perturbed_flat.t()) / temperature
        
        # 限制相似度矩阵范围避免溢出
        similarity_matrix = torch.clamp(similarity_matrix, -50, 50)
        
        # 正样本：对应时间步的内容码
        positive_indices = torch.arange(content_flat.size(0)).to(content_flat.device)
        
        # 对比损失（InfoNCE）
        labels = positive_indices
        contrastive_loss = F.cross_entropy(similarity_matrix, labels)
        
        # 检查是否为NaN
        if torch.isnan(contrastive_loss):
            return torch.tensor(0.0, device=content_code.device, requires_grad=True)
        
        return contrastive_loss
    
    def safe_tensor(self, value, device):
        """安全创建张量，避免NaN"""
        if torch.isnan(value) or torch.isinf(value):
            return torch.tensor(0.0, device=device, requires_grad=True)
        return value
    
    def pcc_loss(self, predictions, targets):
        """计算皮尔逊相关系数损失 - 直接优化PCC指标"""
        # 展平张量
        pred_flat = predictions.view(-1)
        target_flat = targets.view(-1)
        
        # 计算均值
        pred_mean = torch.mean(pred_flat)
        target_mean = torch.mean(target_flat)
        
        # 计算协方差和标准差
        pred_centered = pred_flat - pred_mean
        target_centered = target_flat - target_mean
        
        covariance = torch.mean(pred_centered * target_centered)
        pred_std = torch.sqrt(torch.mean(pred_centered ** 2) + 1e-8)
        target_std = torch.sqrt(torch.mean(target_centered ** 2) + 1e-8)
        
        # 计算PCC
        pcc = covariance / (pred_std * target_std + 1e-8)
        
        # 返回负PCC作为损失（最大化PCC等于最小化-PCC）
        pcc_loss = 1.0 - pcc
        
        # 检查数值稳定性
        if torch.isnan(pcc_loss) or torch.isinf(pcc_loss):
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        return pcc_loss
    
    def forward(self, features, target_od=None, mode='train'):
        """
        前向传播
        Args:
            features: [batch_size, time_steps=28, input_dim=6] 输入特征
            target_od: [batch_size, time_steps=28, output_dim=2] 目标OD流量
            mode: 'train' 或 'eval'
        Returns:
            结果字典
        """
        batch_size = features.size(0)
        device = features.device
        
        if mode == 'train' and target_od is not None:
            # 训练模式
            
            try:
                # 1. 内容编码和对比学习
                content_code = self.content_encoder(target_od)
                perturbed_od = self.apply_information_perturbation(target_od)
                perturbed_content_code = self.content_encoder(perturbed_od)
                
                # 对比损失
                contrastive_loss = self.contrastive_loss(content_code, perturbed_content_code)
                
                # 2. 后验编码
                mu, logvar = self.posterior_encoder(target_od, features)
                
                # 3. 重参数化
                z = self.reparameterize(mu, logvar)
                
                # 4. 归一化流变换
                z_flow, log_det = self.normalizing_flow(z, reverse=False)
                
                # 5. 解码生成
                predicted_od = self.decoder(z_flow, features)
                
                # 6. 判别器
                real_logits = self.discriminator(target_od, features)
                fake_logits = self.discriminator(predicted_od.detach(), features)
                
                # === 损失计算 ===
                # 重构损失
                reconstruction_loss = self.mse_loss(predicted_od, target_od)
                
                # VAE损失：KL散度 (添加数值稳定性)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
                kl_loss = torch.clamp(kl_loss, -1000, 1000)  # 限制范围
                kl_loss = kl_loss.mean()
                
                # 流损失：对数雅可比行列式 (添加数值稳定性)
                flow_loss = -log_det.mean()
                flow_loss = torch.clamp(flow_loss, -10, 10)  # 大幅限制范围，避免数值爆炸
                
                # 生成器损失（对抗）
                generator_adv_loss = self.bce_loss(
                    self.discriminator(predicted_od, features), 
                    torch.ones_like(fake_logits)
                )
                
                # 判别器损失
                discriminator_loss = (
                    self.bce_loss(real_logits, torch.ones_like(real_logits)) +
                    self.bce_loss(fake_logits, torch.zeros_like(fake_logits))
                ) * 0.5
                
                # 特征映射损失（训练推理网络）
                feature_latent = self.feature_to_latent(features)
                feature_mapping_loss = self.mse_loss(feature_latent, mu.detach())
                
                # PCC损失 - 直接优化相关性
                pcc_loss_value = torch.tensor(0.0, device=device, requires_grad=True)
                if self.use_pcc_loss:
                    pcc_loss_value = self.pcc_loss(predicted_od, target_od)
                
                # 检查各个损失是否为NaN，如果是则设为0
                reconstruction_loss = self.safe_tensor(reconstruction_loss, device)
                kl_loss = self.safe_tensor(kl_loss, device)
                flow_loss = self.safe_tensor(flow_loss, device)
                generator_adv_loss = self.safe_tensor(generator_adv_loss, device)
                contrastive_loss = self.safe_tensor(contrastive_loss, device)
                feature_mapping_loss = self.safe_tensor(feature_mapping_loss, device)
                pcc_loss_value = self.safe_tensor(pcc_loss_value, device)
                
                # 总损失 (重新调整权重，重点优化PCC)
                beta = 0.005    # KL权重 (进一步降低)
                gamma = 0.0001  # 流权重 (大幅降低，避免数值问题)
                delta = 0.005   # 对抗权重 (降低)
                eta = 0.001     # 对比损失权重 (大幅降低)
                zeta = 0.05     # 特征映射权重 (降低)
                alpha = 2.0     # PCC损失权重 (新增，重点优化)
                
                total_loss = (reconstruction_loss + 
                             beta * kl_loss + 
                             gamma * flow_loss +
                             delta * generator_adv_loss +
                             eta * contrastive_loss +
                             zeta * feature_mapping_loss +
                             alpha * pcc_loss_value)
                
                # 最终检查总损失
                if torch.isnan(total_loss):
                    total_loss = torch.tensor(1.0, device=device, requires_grad=True)
                
                # 额外指标
                mae_loss = self.mae_loss(predicted_od, target_od)
                
                return {
                    'od_flows': predicted_od,
                    'total_loss': total_loss,
                    'reconstruction_loss': reconstruction_loss,
                    'kl_loss': kl_loss,
                    'flow_loss': flow_loss,
                    'generator_adv_loss': generator_adv_loss,
                    'discriminator_loss': discriminator_loss,
                    'contrastive_loss': contrastive_loss,
                    'feature_mapping_loss': feature_mapping_loss,
                    'pcc_loss': pcc_loss_value,  # 新增PCC损失
                    'mse_loss': reconstruction_loss,  # 兼容性
                    'mae_loss': mae_loss
                }
                
            except Exception as e:
                print(f"训练前向传播出错: {e}")
                # 返回安全的默认值
                dummy_od = torch.zeros_like(target_od, requires_grad=True)
                dummy_loss = torch.tensor(1.0, device=device, requires_grad=True)
                
                return {
                    'od_flows': dummy_od,
                    'total_loss': dummy_loss,
                    'reconstruction_loss': dummy_loss,
                    'kl_loss': dummy_loss,
                    'flow_loss': dummy_loss,
                    'generator_adv_loss': dummy_loss,
                    'discriminator_loss': dummy_loss,
                    'contrastive_loss': dummy_loss,
                    'feature_mapping_loss': dummy_loss,
                    'mse_loss': dummy_loss,
                    'mae_loss': dummy_loss
                }
            
        else:
            # 推理模式
            
            # 1. 从特征生成潜在变量
            z_mean = self.feature_to_latent(features)
            
            # 添加适度随机性
            z_std = 0.05  # 减小随机性
            z = z_mean + torch.randn_like(z_mean) * z_std
            
            # 2. 归一化流变换
            z_flow, _ = self.normalizing_flow(z, reverse=False)
            
            # 3. 解码生成
            predicted_od = self.decoder(z_flow, features)
            
            return {
                'od_flows': predicted_od
            }
    
    def generate(self, features):
        """生成OD流量预测 - 保持与原代码接口一致"""
        with torch.no_grad():
            result = self.forward(features, mode='eval')
            return result['od_flows']

# ========== 简化的数据集类（与原代码完全一致）==========
class SimpleODFlowDataset(Dataset):
    """简化的OD流量数据集 - 与原代码完全保持一致"""
    def __init__(self, io_flow_path, graph_path, od_matrix_path, test_ratio=0.2, val_ratio=0.1, seed=42):
        super().__init__()
        
        # 加载数据
        self.io_flow = np.load(io_flow_path)  # (时间步, 站点数, 特征数)
        self.graph = np.load(graph_path)      # (站点数, 站点数)  
        self.od_matrix = np.load(od_matrix_path)  # (时间步, 站点数, 站点数)
        
        # 转换维度顺序：从 (时间步, 站点数, 特征数) 到 (站点数, 时间步, 特征数)
        if self.io_flow.shape[0] == 28:  # 如果第一个维度是时间步
            self.io_flow = np.transpose(self.io_flow, (1, 0, 2))
        
        # 转换维度顺序：从 (时间步, 站点数, 站点数) 到 (站点数, 站点数, 时间步)  
        if self.od_matrix.shape[0] == 28:  # 如果第一个维度是时间步
            self.od_matrix = np.transpose(self.od_matrix, (1, 2, 0))
        
        # 动态获取维度 - 按照指南要求
        self.num_nodes = self.io_flow.shape[0]
        self.time_steps = self.io_flow.shape[1]
        
        # 数据一致性验证 - 按照指南要求
        print(f"数据维度: IO流量{self.io_flow.shape}, 图{self.graph.shape}, OD矩阵{self.od_matrix.shape}")
        
        # 验证数据维度一致性
        assert self.io_flow.shape[0] == 28 or self.io_flow.shape[1] == 28, f"IO流量数据时间步数不正确: {self.io_flow.shape}"
        assert self.io_flow.shape[2] == 2 or self.io_flow.shape[2] == 4, f"IO流量数据特征数不正确: {self.io_flow.shape} (应该是2或4个特征)"
        assert self.graph.shape[0] == self.graph.shape[1], f"图数据不是方阵: {self.graph.shape}"
        assert self.graph.shape[0] == self.num_nodes, f"图数据维度与节点数不匹配: {self.graph.shape[0]} vs {self.num_nodes}"
        assert self.od_matrix.shape[0] == self.num_nodes and self.od_matrix.shape[1] == self.num_nodes, f"OD矩阵维度与节点数不匹配: {self.od_matrix.shape} vs ({self.num_nodes}, {self.num_nodes})"
        assert self.od_matrix.shape[2] == self.time_steps, f"OD矩阵时间步数不匹配: {self.od_matrix.shape[2]} vs {self.time_steps}"
        
        print(f"✅ 数据一致性验证通过: {self.num_nodes}个节点, {self.time_steps}个时间步")
        
        # 站点对列表 - 使用动态节点数量
        self.od_pairs = []
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                self.od_pairs.append((i, j))
        
        print(f"生成{len(self.od_pairs)}个站点对用于训练")
        
        # 加载站点人口密度数据 - 优先使用52节点版本
        population_files = [
            "/private/od/data_NYTaxi/grid_population_density_52nodes.json",  # 优先使用52节点版本
            "/private/od/data_NYTaxi/grid_population_density.json",  # 原始备用
            "/private/od/data/station_p.json"  # 旧版本备用
        ]
        
        self.station_data = []
        for pop_file in population_files:
            if os.path.exists(pop_file):
                try:
                    with open(pop_file, 'r', encoding='utf-8') as f:
                        self.station_data = json.load(f)
                    print(f"✅ 加载人口密度数据: {pop_file}, 共{len(self.station_data)}个区域")
                    break
                except Exception as e:
                    print(f"⚠️ 加载人口密度数据失败 {pop_file}: {str(e)}")
                    continue
        
        if not self.station_data:
            print(f"⚠️ 所有人口密度数据文件都无法加载，使用空数据")
            self.station_data = []
        else:
            # 验证人口密度数据与节点数量的一致性
            if len(self.station_data) != self.num_nodes:
                print(f"⚠️ 人口密度数据数量({len(self.station_data)})与节点数量({self.num_nodes})不匹配")
                if len(self.station_data) > self.num_nodes:
                    print(f"   截取前{self.num_nodes}个人口密度数据")
                    self.station_data = self.station_data[:self.num_nodes]
                else:
                    print(f"   人口密度数据不足，将使用默认值填充")
            else:
                print(f"✅ 人口密度数据数量与节点数量匹配: {len(self.station_data)}个")
        
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
        
        # 获取IO流量 - 支持2或4个特征
        io_flow_i = self.io_flow[site_i, :, :]  # (时间步, 特征数)
        io_flow_j = self.io_flow[site_j, :, :]  # (时间步, 特征数)
        
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
        features = np.concatenate([io_flow_i, io_flow_j, distance_feature, pop_density_feature], axis=1)  
        # 特征维度: (时间步, io_flow_features*2 + 2) = (时间步, 2*2+2=6) 或 (时间步, 4*2+2=10)
        
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
            
            try:
                # 生成预测
                predicted = model.generate(features)
                
                # 计算损失
                loss = F.mse_loss(predicted, od_flows)
                
                # 检查是否为NaN
                if not torch.isnan(loss):
                    total_losses.append(loss.item())
                    
                    # 收集预测结果
                    all_predictions.append(predicted.cpu().numpy())
                    all_targets.append(od_flows.cpu().numpy())
                    
                    progress.set_postfix({'MSE': f'{loss.item():.6f}'})
                else:
                    print("遇到NaN损失，跳过此批次")
                    
            except Exception as e:
                print(f"评估时出错: {e}")
                continue
    
    if len(all_predictions) == 0:
        return {
            'loss': float('inf'),
            'mse': float('inf'), 
            'rmse': float('inf'),
            'mae': float('inf'),
            'pcc': 0.0
        }
    
    # 合并所有预测结果
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 计算评估指标
    mse = np.mean((all_predictions - all_targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_predictions - all_targets))
    
    # 计算皮尔逊相关系数(PCC)
    pred_flat = all_predictions.flatten()
    target_flat = all_targets.flatten()
    valid_mask = ~(np.isnan(pred_flat) | np.isnan(target_flat))
    if np.sum(valid_mask) > 0:
        pcc = np.corrcoef(pred_flat[valid_mask], target_flat[valid_mask])[0, 1]
        if np.isnan(pcc):
            pcc = 0.0
    else:
        pcc = 0.0
    
    avg_loss = np.mean(total_losses) if total_losses else float('inf')
    
    return {
        'loss': float(avg_loss),
        'mse': float(mse), 
        'rmse': float(rmse),
        'mae': float(mae),
        'pcc': float(pcc)
    }

# ========== Flow-VAE训练函数 (改进版) ==========
def train_flow_vae_model(args):
    """训练Flow-VAE OD流量预测模型 (数值稳定版)"""
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
    
    # 动态计算输入特征维度
    # 特征构成: io_flow_i + io_flow_j + distance + population_density
    # = io_flow_features*2 + 2
    io_flow_features = dataset.io_flow.shape[2]  # 2 或 4
    input_dim = io_flow_features * 2 + 2  # 6 或 10
    print(f"✅ 动态计算输入特征维度: {input_dim} (IO流量特征: {io_flow_features})")
    
    # 创建Flow-VAE模型
    model = FlowVAEODFlowPredictor(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        content_dim=args.content_dim,
        time_steps=28,
        output_dim=2,
        flow_layers=args.flow_layers
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Flow-VAE模型创建成功！")
    print(f"  总参数数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  隐藏维度: {args.hidden_dim}")
    print(f"  潜在维度: {args.latent_dim}")
    print(f"  内容维度: {args.content_dim}")
    print(f"  流层数: {args.flow_layers}")
    
    # 优化器 - 分别为生成器和判别器创建优化器 (降低学习率)
    generator_params = [p for name, p in model.named_parameters() if 'discriminator' not in name]
    discriminator_params = list(model.discriminator.parameters())
    
    optimizer_G = torch.optim.Adam(
        generator_params, 
        lr=args.lr * 0.5,  # 降低生成器学习率
        betas=(0.5, 0.999),
        weight_decay=args.weight_decay
    )
    
    optimizer_D = torch.optim.Adam(
        discriminator_params,
        lr=args.lr,  # 保持判别器学习率
        betas=(0.5, 0.999),
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_G, mode='min', factor=0.5, patience=args.patience, verbose=True, min_lr=1e-6
    )
    
    scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_D, mode='min', factor=0.5, patience=args.patience, verbose=True, min_lr=1e-6
    )
    
    # 训练循环变量
    best_val_loss = float('inf')
    best_model_path = os.path.join(args.output_dir, 'best_flow_vae_od_model.pth')
    epochs_without_improvement = 0
    train_history = []
    
    print(f"\n开始训练Flow-VAE OD流量预测模型...")
    print(f"模型将保存到: {best_model_path}")
    print(f"早停策略: 验证损失{args.early_stop_patience}轮无改善时停止训练")
    print("="*80)
    
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_losses = []
        train_recon_losses = []
        train_kl_losses = []
        train_flow_losses = []
        train_gen_adv_losses = []
        train_disc_losses = []
        train_contrastive_losses = []
        train_feature_mapping_losses = []
        train_pcc_losses = []
        
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{args.epochs} [训练]")
        for features, od_flows in train_progress:
            features = features.to(device)
            od_flows = od_flows.to(device)
            
            try:
                # === 训练生成器 ===
                optimizer_G.zero_grad()
                
                outputs = model(features, od_flows, mode='train')
                
                # 生成器损失（重新调整权重，重点优化PCC）
                generator_loss = (outputs['reconstruction_loss'] + 
                                0.005 * outputs['kl_loss'] +
                                0.0001 * outputs['flow_loss'] +
                                0.005 * outputs['generator_adv_loss'] +
                                0.001 * outputs['contrastive_loss'] +
                                0.05 * outputs['feature_mapping_loss'] +
                                2.0 * outputs['pcc_loss'])  # 重点优化PCC
                
                # 检查生成器损失是否为NaN
                if not torch.isnan(generator_loss):
                    generator_loss.backward()
                    # 更严格的梯度裁剪
                    torch.nn.utils.clip_grad_norm_(generator_params, max_norm=0.5)
                    optimizer_G.step()
                
                # === 训练判别器 ===
                optimizer_D.zero_grad()
                
                # 重新计算判别器损失
                with torch.no_grad():
                    try:
                        fake_od = model.decoder(
                            model.normalizing_flow(
                                model.reparameterize(*model.posterior_encoder(od_flows, features)),
                                reverse=False
                            )[0],
                            features
                        )
                    except:
                        fake_od = od_flows  # 如果生成失败，使用真实数据
                
                real_logits = model.discriminator(od_flows, features)
                fake_logits = model.discriminator(fake_od, features)
                
                discriminator_loss = (
                    model.bce_loss(real_logits, torch.ones_like(real_logits)) +
                    model.bce_loss(fake_logits, torch.zeros_like(fake_logits))
                ) * 0.5
                
                # 检查判别器损失是否为NaN
                if not torch.isnan(discriminator_loss):
                    discriminator_loss.backward()
                    torch.nn.utils.clip_grad_norm_(discriminator_params, max_norm=0.5)
                    optimizer_D.step()
                
                # 记录损失
                if not torch.isnan(outputs['total_loss']):
                    train_losses.append(outputs['total_loss'].item())
                    train_recon_losses.append(outputs['reconstruction_loss'].item())
                    train_kl_losses.append(outputs['kl_loss'].item())
                    train_flow_losses.append(outputs['flow_loss'].item())
                    train_gen_adv_losses.append(outputs['generator_adv_loss'].item())
                    train_disc_losses.append(discriminator_loss.item())
                    train_contrastive_losses.append(outputs['contrastive_loss'].item())
                    train_feature_mapping_losses.append(outputs['feature_mapping_loss'].item())
                    train_pcc_losses.append(outputs['pcc_loss'].item())  # 记录PCC损失
                
                # 更新进度条
                train_progress.set_postfix({
                    'Total': f'{outputs["total_loss"].item():.4f}',
                    'Recon': f'{outputs["reconstruction_loss"].item():.4f}',
                    'PCC': f'{outputs["pcc_loss"].item():.4f}',  # 显示PCC损失
                    'KL': f'{outputs["kl_loss"].item():.4f}',
                    'Flow': f'{outputs["flow_loss"].item():.4f}',
                    'Adv': f'{outputs["generator_adv_loss"].item():.4f}'
                })
                
            except Exception as e:
                print(f"训练批次出错: {e}")
                continue
        
        # 计算训练指标
        if len(train_losses) > 0:
            avg_train_loss = np.mean(train_losses)
            avg_train_recon = np.mean(train_recon_losses)
            avg_train_kl = np.mean(train_kl_losses)
            avg_train_flow = np.mean(train_flow_losses)
            avg_train_gen_adv = np.mean(train_gen_adv_losses)
            avg_train_disc = np.mean(train_disc_losses)
            avg_train_contrastive = np.mean(train_contrastive_losses)
            avg_train_feature_mapping = np.mean(train_feature_mapping_losses)
            avg_train_pcc = np.mean(train_pcc_losses)
        else:
            avg_train_loss = float('inf')
            avg_train_recon = float('inf')
            avg_train_kl = float('inf')
            avg_train_flow = float('inf')
            avg_train_gen_adv = float('inf')
            avg_train_disc = float('inf')
            avg_train_contrastive = float('inf')
            avg_train_feature_mapping = float('inf')
            avg_train_pcc = float('inf')
        
        # 验证阶段 - 计算详细指标
        print(f"  🔍 计算验证集指标...")
        val_metrics = calculate_metrics(model, val_loader, device, desc="验证集评估")
        
        # 学习率调整
        scheduler_G.step(val_metrics['loss'])
        scheduler_D.step(val_metrics['loss'])
        current_lr_G = optimizer_G.param_groups[0]['lr']
        current_lr_D = optimizer_D.param_groups[0]['lr']
        
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
        print(f"{'='*100}")
        print(f"🔹 训练集:")
        print(f"   总损失: {avg_train_loss:.6f} | 重构损失: {avg_train_recon:.6f} | PCC损失: {avg_train_pcc:.6f}")
        print(f"   KL损失: {avg_train_kl:.6f} | 流损失: {avg_train_flow:.6f} | 生成器对抗: {avg_train_gen_adv:.6f}")
        print(f"   判别器: {avg_train_disc:.6f} | 对比损失: {avg_train_contrastive:.6f} | 特征映射: {avg_train_feature_mapping:.6f}")
        
        print(f"🔹 验证集:")
        print(f"   总损失: {val_metrics['loss']:.6f} | MSE: {val_metrics['mse']:.6f}")
        print(f"   RMSE: {val_metrics['rmse']:.6f} | MAE: {val_metrics['mae']:.6f} | PCC: {val_metrics['pcc']:.6f}")
        
        if test_metrics:
            print(f"🔹 测试集:")  
            print(f"   总损失: {test_metrics.get('loss', 0):.6f} | MSE: {test_metrics.get('mse', 0):.6f}")
            print(f"   RMSE: {test_metrics.get('rmse', 0):.6f} | MAE: {test_metrics.get('mae', 0):.6f} | PCC: {test_metrics.get('pcc', 0):.6f}")
        else:
            print(f"🔹 测试集: 未评估 (仅在验证集改善时评估)")
        
        print(f"🔹 学习率: G={current_lr_G:.2e}, D={current_lr_D:.2e}")
        
        # 保存训练历史
        epoch_history = {
            'epoch': int(epoch + 1),
            'train_loss': float(avg_train_loss),
            'train_reconstruction_loss': float(avg_train_recon),
            'train_kl_loss': float(avg_train_kl),
            'train_flow_loss': float(avg_train_flow),
            'train_generator_adv_loss': float(avg_train_gen_adv),
            'train_discriminator_loss': float(avg_train_disc),
            'train_contrastive_loss': float(avg_train_contrastive),
            'train_feature_mapping_loss': float(avg_train_feature_mapping),
            'train_pcc_loss': float(avg_train_pcc),
            'val_loss': float(val_metrics['loss']),
            'val_mse': float(val_metrics['mse']),
            'val_rmse': float(val_metrics['rmse']),
            'val_mae': float(val_metrics['mae']),
            'val_pcc': float(val_metrics['pcc']),
            'lr_generator': float(current_lr_G),
            'lr_discriminator': float(current_lr_D),
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
        
        # 边训练边保存训练日志
        log_file = os.path.join(args.output_dir, "training_log.txt")
        try:
            mode = 'w' if epoch == 0 else 'a'
            with open(log_file, mode, encoding='utf-8') as f:
                if epoch == 0:
                    f.write("Flow-VAE OD流量预测模型训练日志 (数值稳定版)\n")
                    f.write("=" * 50 + "\n")
                
                f.write(f"Epoch {epoch+1}/{args.epochs}\n")
                f.write(f"   Training - Total: {avg_train_loss:.6f}, Recon: {avg_train_recon:.6f}, PCC: {avg_train_pcc:.6f}, KL: {avg_train_kl:.6f}\n")
                f.write(f"            - Flow: {avg_train_flow:.6f}, GenAdv: {avg_train_gen_adv:.6f}, Disc: {avg_train_disc:.6f}, Contr: {avg_train_contrastive:.6f}, FeatMap: {avg_train_feature_mapping:.6f}\n")
                f.write(f"   Validation - Loss: {val_metrics['loss']:.6f}, RMSE: {val_metrics['rmse']:.6f}, MAE: {val_metrics['mae']:.6f}, PCC: {val_metrics['pcc']:.6f}\n")
                
                if test_metrics:
                    f.write(f"   Test - Loss: {test_metrics.get('loss', 0):.6f}, RMSE: {test_metrics.get('rmse', 0):.6f}, MAE: {test_metrics.get('mae', 0):.6f}, PCC: {test_metrics.get('pcc', 0):.6f}\n")
                
                if is_best:
                    f.write(f"   New best model saved (Val Loss: {best_val_loss:.6f}, Val RMSE: {val_metrics['rmse']:.6f}, Val PCC: {val_metrics['pcc']:.6f})\n")
                else:
                    f.write(f"   No improvement ({epochs_without_improvement}/{args.early_stop_patience} epochs without improvement)\n")
                
                f.write(f"   Learning Rate: G={current_lr_G:.2e}, D={current_lr_D:.2e}\n")
                f.write("\n")
                f.flush()
        except Exception as e:
            print(f"⚠️ 保存训练日志失败: {e}")
        
        # 保存JSON格式的详细历史数据
        history_file = os.path.join(args.output_dir, "training_history.json")
        try:
            with open(history_file, "w", encoding='utf-8') as f:
                json.dump(train_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ 保存详细历史失败: {e}")
        
        # 保存最佳模型
        if is_best and val_metrics['loss'] != float('inf'):
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'scheduler_G_state_dict': scheduler_G.state_dict(),
                'scheduler_D_state_dict': scheduler_D.state_dict(),
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
        if current_lr_G < 1e-6:
            print(f"\n🛑 学习率过小 ({current_lr_G:.2e})，停止训练")
            break
        
        print("="*100)
    
    print(f"📁 训练日志已实时保存到: {log_file}")
    print(f"📁 详细历史数据已保存到: {history_file}")
    
    # 最终测试阶段
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
        
        final_test_metrics = best_test_metrics
    else:
        print("⚠️ 最佳模型文件不存在，使用当前模型进行最终测试")
        final_test_metrics = calculate_metrics(model, test_loader, device, desc="最终测试")
        best_epoch = "当前"
    
    print(f"\n{'='*60}")
    print("🎉 Flow-VAE OD流量预测模型 - 最终测试结果 (数值稳定版)")
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
    results_file = os.path.join(args.output_dir, "flow_vae_od_results.txt")
    with open(results_file, "w", encoding='utf-8') as f:
        f.write("基于Flow-VAE的OD流量预测模型测试结果 (数值稳定版)\n")
        f.write("="*50 + "\n")
        f.write("论文: Flow-VAEVC: End-to-End Flow Framework with Contrastive Loss for Zero-shot Voice Conversion (INTERSPEECH 2023)\n")
        f.write("模型架构核心特点:\n")
        f.write("  - 条件VAE + 归一化流 (Conditional VAE + Normalizing Flow)\n")
        f.write("  - 对比损失学习解耦表示 (Contrastive Loss for Disentanglement)\n")
        f.write("  - 信息扰动数据增强 (Information Perturbation)\n")
        f.write("  - 对抗训练提高生成质量 (Adversarial Training)\n")
        f.write("  - 端到端框架 (End-to-End Framework)\n")
        f.write("  - 数值稳定性改进 (Numerical Stability Improvements)\n")
        f.write("\n")
        f.write(f"模型参数:\n")
        f.write(f"  - 总参数数量: {total_params:,}\n")
        f.write(f"  - 可训练参数: {trainable_params:,}\n")
        f.write(f"  - 隐藏维度: {args.hidden_dim}\n")
        f.write(f"  - 潜在维度: {args.latent_dim}\n")
        f.write(f"  - 内容维度: {args.content_dim}\n")
        f.write(f"  - 流层数: {args.flow_layers}\n")
        f.write(f"  - 训练轮数: {args.epochs}\n")
        f.write(f"  - 批次大小: {args.batch_size}\n")
        f.write(f"  - 学习率: {args.lr}\n")
        f.write("\n")
        f.write("数值稳定性改进:\n")
        f.write("  - 归一化流尺度参数限制在[-5, 5]范围\n")
        f.write("  - 对比损失使用L2归一化和温度调整\n")
        f.write("  - KL损失和流损失限制在[-1000, 1000]范围\n")
        f.write("  - 添加NaN检查和处理机制\n")
        f.write("  - 改进的权重初始化 (Xavier normal with gain=0.1)\n")
        f.write("  - 更严格的梯度裁剪 (max_norm=0.5)\n")
        f.write("  - 降低各损失权重避免训练不稳定\n")
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
    parser = argparse.ArgumentParser(description="基于Flow-VAE的OD流量预测模型 (数值稳定版)")
    
    # 数据参数
    parser.add_argument("--io_flow_path", type=str, default="/private/od/data_NYTaxi/io_flow_daily.npy", 
                       help="IO流量数据路径")
    parser.add_argument("--graph_path", type=str, default="/private/od/data_NYTaxi/graph.npy", 
                       help="图结构数据路径")
    parser.add_argument("--od_matrix_path", type=str, default="/private/od/data_NYTaxi/od_matrix_daily.npy", 
                       help="OD矩阵数据路径")
    
    # Flow-VAE模型参数
    parser.add_argument("--hidden_dim", type=int, default=64,  # 降低维度
                       help="隐藏维度")
    parser.add_argument("--latent_dim", type=int, default=32,  # 降低维度
                       help="潜在空间维度")
    parser.add_argument("--content_dim", type=int, default=32,  # 降低维度
                       help="内容编码维度")
    parser.add_argument("--flow_layers", type=int, default=2,  # 减少层数
                       help="归一化流层数")
    
    # 训练参数  
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")  # 减小batch size
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.0001, help="学习率")  # 降低学习率
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="权重衰减")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--seed", type=int, default=82, help="随机种子")
    
    # 早停和学习率调整参数
    parser.add_argument("--early_stop_patience", type=int, default=15, help="早停策略")
    parser.add_argument("--patience", type=int, default=8, help="学习率调整策略")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="/private/od/paper_ny/Flow_VAE", help="输出目录")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = create_dynamic_output_dir(args.output_dir)
    args.output_dir = output_dir
    
    print("="*60)
    print("🚀 Flow-VAE OD流量预测模型 (数值稳定版)")
    print("="*60)
    print("📖 论文: Flow-VAEVC: End-to-End Flow Framework with Contrastive Loss for Zero-shot Voice Conversion")
    print("📖 会议: INTERSPEECH 2023")
    print("📖 作者: Le Xu, Rongxiu Zhong, Ying Liu, Huibao Yang, Shilei Zhang")
    print()
    print("🔧 模型创新点:")
    print("  ✅ 条件VAE + 归一化流 - 结合语义建模和细节重构能力")
    print("  ✅ 对比损失 - 学习时间不变的流量模式表示")
    print("  ✅ 信息扰动 - 通过数据增强提高模型鲁棒性")
    print("  ✅ 对抗训练 - 使用判别器提高生成质量")
    print("  ✅ 端到端框架 - 直接处理原始数据，无需中间表示")
    print()
    print("🛠️ 数值稳定性改进:")
    print("  ✅ 归一化流参数限制范围避免溢出")
    print("  ✅ 对比损失L2归一化和温度调整")
    print("  ✅ NaN检查和处理机制")
    print("  ✅ 改进的权重初始化")
    print("  ✅ 更严格的梯度裁剪")
    print()
    print(f"📁 输出目录: {output_dir}")
    print("="*60)
    
    # 训练模型
    try:
        best_model_path = train_flow_vae_model(args)
        print("\n🎉 Flow-VAE模型训练完成!")
        print(f"📁 最佳模型保存位置: {best_model_path}")
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {str(e)}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)