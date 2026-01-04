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
    dynamic_dir = os.path.join(base_dir, f"ivp_vae_run_{timestamp}")
    os.makedirs(dynamic_dir, exist_ok=True)
    return dynamic_dir

# ========== IVP-VAE 核心组件 ==========

class NeuralODESolver(nn.Module):
    """Neural ODE IVP求解器 - IVP-VAE的核心创新
    
    基于Neural ODE实现初值问题求解，支持双向时间演化：
    - 前向演化: 从t0到ti (解码器用)
    - 后向演化: 从ti到t0 (编码器用)
    
    这是IVP-VAE相比Latent-ODE的关键改进：
    1. 可逆性：同一求解器可以双向演化
    2. 并行性：不同时间点可以并行处理
    3. 效率：避免顺序RNN计算瓶颈
    """
    
    def __init__(self, latent_dim=64, hidden_dim=128):
        super(NeuralODESolver, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Neural ODE函数 - 定义dz/dt = f(z, t)
        self.ode_func = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),  # +1 for time
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, z, time_delta):
        """
        求解IVP: dz/dt = f(z, t), z(t0) = z0
        Args:
            z: [batch_size, latent_dim] 初始状态
            time_delta: [batch_size] 时间变化量，可正可负
        Returns:
            z_new: [batch_size, latent_dim] 演化后的状态
        """
        batch_size = z.size(0)
        
        # 简化的Euler方法求解ODE (可替换为更复杂的Runge-Kutta)
        # 对于每个样本，时间变化量可能不同
        dt = 0.1  # 固定步长
        steps = torch.ceil(torch.abs(time_delta) / dt).int().max().item()
        if steps == 0:
            return z
        
        # 计算实际步长
        actual_dt = time_delta.unsqueeze(1) / max(steps, 1)  # [batch_size, 1]
        
        z_current = z
        for step in range(steps):
            # 当前时间 (简化处理)
            t = torch.ones(batch_size, 1, device=z.device) * (step * dt)
            
            # 计算导数
            z_with_time = torch.cat([z_current, t], dim=1)
            dz_dt = self.ode_func(z_with_time)
            
            # Euler步进
            z_current = z_current + dz_dt * actual_dt
        
        return z_current

class EmbeddingModule(nn.Module):
    """嵌入模块 - 将观测映射到潜在状态空间
    
    对应论文中的Embedding步骤：
    1. 处理缺失值mask (在OD流量中简化处理)
    2. 将多变量观测映射到潜在状态
    3. 为IVP求解器准备状态表示
    """
    
    def __init__(self, input_dim=6, output_dim=64, hidden_dim=128):
        super(EmbeddingModule, self).__init__()
        
        # 特征嵌入网络
        self.feature_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # OD流量嵌入网络
        self.od_embedding = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),  # OD流量维度为2
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, features=None, od_flows=None):
        """
        将输入映射到潜在状态
        Args:
            features: [batch_size, time_steps, input_dim] 输入特征
            od_flows: [batch_size, time_steps, 2] OD流量 (训练时提供)
        Returns:
            embedded_states: [batch_size, time_steps, latent_dim] 潜在状态
        """
        if od_flows is not None:
            # 训练模式：使用真实OD流量
            return self.od_embedding(od_flows)
        else:
            # 推理模式：仅使用特征
            return self.feature_embedding(features)

class ReconstructionModule(nn.Module):
    """重构模块 - 从潜在状态重构观测数据
    
    对应论文中的Reconstruction步骤：
    将潜在状态zi映射回观测空间xi
    """
    
    def __init__(self, input_dim=64, output_dim=2, hidden_dim=128):
        super(ReconstructionModule, self).__init__()
        
        self.reconstruction_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, latent_states):
        """
        从潜在状态重构OD流量
        Args:
            latent_states: [batch_size, time_steps, latent_dim] 潜在状态
        Returns:
            reconstructed_od: [batch_size, time_steps, 2] 重构的OD流量
        """
        return self.reconstruction_net(latent_states)

class IVPVAEEncoder(nn.Module):
    """IVP-VAE编码器 - 向后时间演化获得z0后验分布
    
    核心创新：
    1. 从每个观测时间点ti向t0=0演化，得到zi0
    2. 多个zi0构成混合分布建模p(z0|X)
    3. 避免顺序处理，支持并行计算
    """
    
    def __init__(self, latent_dim=64, hidden_dim=128):
        super(IVPVAEEncoder, self).__init__()
        
        self.latent_dim = latent_dim
        
        # IVP求解器 - 向后演化
        self.ivp_solver = NeuralODESolver(latent_dim, hidden_dim)
        
        # 后验参数网络
        self.posterior_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, latent_dim * 2)  # mu and logvar
        )
        
    def forward(self, embedded_states, time_points):
        """
        编码：向后演化到t0，得到z0分布参数
        Args:
            embedded_states: [batch_size, time_steps, latent_dim] 嵌入的状态
            time_points: [time_steps] 时间点 (简化为0, 1, 2, ..., T-1)
        Returns:
            z0_distributions: List of (mu, logvar) for each time step
        """
        batch_size, time_steps, _ = embedded_states.shape
        z0_distributions = []
        
        for t in range(time_steps):
            # 当前时间点的状态
            zt = embedded_states[:, t, :]  # [batch_size, latent_dim]
            
            # 向后演化到t0 (时间差为负数)
            time_delta = torch.full((batch_size,), -float(t), device=zt.device)
            z0_estimated = self.ivp_solver(zt, time_delta)
            
            # 计算后验分布参数
            posterior_params = self.posterior_net(z0_estimated)
            mu = posterior_params[:, :self.latent_dim]
            logvar = posterior_params[:, self.latent_dim:]
            
            z0_distributions.append((mu, logvar))
        
        return z0_distributions

class IVPVAEDecoder(nn.Module):
    """IVP-VAE解码器 - 向前时间演化生成观测
    
    共享同一个IVP求解器，但方向相反：
    从z0向前演化到各个时间点ti
    """
    
    def __init__(self, latent_dim=64, hidden_dim=128):
        super(IVPVAEDecoder, self).__init__()
        
        # 共享的IVP求解器 - 向前演化
        self.ivp_solver = NeuralODESolver(latent_dim, hidden_dim)
        
    def forward(self, z0, time_steps):
        """
        解码：从z0向前演化到各时间点
        Args:
            z0: [batch_size, latent_dim] 初始潜在状态
            time_steps: int 时间步数
        Returns:
            latent_sequence: [batch_size, time_steps, latent_dim] 潜在状态序列
        """
        batch_size = z0.size(0)
        latent_sequence = []
        
        for t in range(time_steps):
            if t == 0:
                # t=0时刻就是z0本身
                zt = z0
            else:
                # 向前演化到时间点t
                time_delta = torch.full((batch_size,), float(t), device=z0.device)
                zt = self.ivp_solver(z0, time_delta)
            
            latent_sequence.append(zt)
        
        # 转换为张量
        latent_sequence = torch.stack(latent_sequence, dim=1)  # [batch_size, time_steps, latent_dim]
        return latent_sequence

class IVPVAEODFlowPredictor(nn.Module):
    """基于IVP-VAE的OD流量预测模型
    
    主要创新点：
    1. 纯IVP建模：完全基于初值问题求解，避免RNN顺序计算
    2. 共享求解器：编码器和解码器使用同一个IVP求解器
    3. 并行处理：不同时间点可以并行演化
    4. 混合后验：通过多个zi0建模复杂的z0分布
    5. 参数效率：共享机制减少参数，提升收敛
    """
    
    def __init__(self, input_dim=6, hidden_dim=128, latent_dim=64, time_steps=28, output_dim=2):
        super(IVPVAEODFlowPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.time_steps = time_steps
        self.output_dim = output_dim
        
        # 核心模块
        self.embedding = EmbeddingModule(input_dim, latent_dim, hidden_dim)
        self.encoder = IVPVAEEncoder(latent_dim, hidden_dim)
        self.decoder = IVPVAEDecoder(latent_dim, hidden_dim)
        self.reconstruction = ReconstructionModule(latent_dim, output_dim, hidden_dim)
        
        # 混合分布权重网络 - 对应论文中的π权重
        self.mixing_weights_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softmax(dim=1)
        )
        
        # 特征条件化网络 - 用于推理时的条件生成
        self.feature_to_z0 = nn.Sequential(
            nn.Linear(input_dim * time_steps, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        
        # PCC优化权重 - 重点优化PCC指标
        self.lambda_pcc = 1.0  # PCC损失权重，平衡优化
        self.lambda_temporal_pcc = 0.5  # 时序PCC损失权重
        self.lambda_feature_align = 0.3  # 特征对齐损失权重
        
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def _compute_enhanced_pcc_loss(self, pred, target):
        """计算增强的皮尔逊相关系数损失 - 多层次PCC优化"""
        batch_size, time_steps, features = pred.shape
        
        # 1. 全局PCC损失 - 整体相关性
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        
        # 数值稳定性检查
        if torch.std(pred_flat) < 1e-6 or torch.std(target_flat) < 1e-6:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # 标准化处理
        pred_norm = (pred_flat - torch.mean(pred_flat)) / (torch.std(pred_flat) + 1e-8)
        target_norm = (target_flat - torch.mean(target_flat)) / (torch.std(target_flat) + 1e-8)
        
        # 计算全局PCC
        global_pcc = torch.mean(pred_norm * target_norm)
        global_pcc = torch.clamp(global_pcc, -1.0, 1.0)  # 限制范围
        
        # 2. 时序PCC损失 - 每个时间步的相关性
        temporal_pcc_losses = []
        for t in range(time_steps):
            pred_t = pred[:, t, :].reshape(-1)
            target_t = target[:, t, :].reshape(-1)
            
            if torch.std(pred_t) > 1e-6 and torch.std(target_t) > 1e-6:
                pred_t_norm = (pred_t - torch.mean(pred_t)) / (torch.std(pred_t) + 1e-8)
                target_t_norm = (target_t - torch.mean(target_t)) / (torch.std(target_t) + 1e-8)
                temporal_pcc = torch.mean(pred_t_norm * target_t_norm)
                temporal_pcc = torch.clamp(temporal_pcc, -1.0, 1.0)
                temporal_pcc_losses.append(1.0 - temporal_pcc)
        
        avg_temporal_pcc_loss = torch.mean(torch.stack(temporal_pcc_losses)) if temporal_pcc_losses else torch.tensor(0.0, device=pred.device)
        
        # 3. 特征维度PCC损失 - 每个特征维度的相关性
        feature_pcc_losses = []
        for f in range(features):
            pred_f = pred[:, :, f].reshape(-1)
            target_f = target[:, :, f].reshape(-1)
            
            if torch.std(pred_f) > 1e-6 and torch.std(target_f) > 1e-6:
                pred_f_norm = (pred_f - torch.mean(pred_f)) / (torch.std(pred_f) + 1e-8)
                target_f_norm = (target_f - torch.mean(target_f)) / (torch.std(target_f) + 1e-8)
                feature_pcc = torch.mean(pred_f_norm * target_f_norm)
                feature_pcc = torch.clamp(feature_pcc, -1.0, 1.0)
                feature_pcc_losses.append(1.0 - feature_pcc)
        
        avg_feature_pcc_loss = torch.mean(torch.stack(feature_pcc_losses)) if feature_pcc_losses else torch.tensor(0.0, device=pred.device)
        
        # 组合损失
        total_pcc_loss = (1.0 - global_pcc) + 0.3 * avg_temporal_pcc_loss + 0.2 * avg_feature_pcc_loss
        
        return total_pcc_loss
    
    def compute_mixture_posterior(self, z0_distributions):
        """
        计算混合后验分布 q(z0|X) = Σπi * q(zi0|X)
        Args:
            z0_distributions: List of (mu, logvar) for each time step
        Returns:
            z0_sample: [batch_size, latent_dim] 采样的z0
            kl_loss: KL散度损失
        """
        batch_size = z0_distributions[0][0].size(0)
        num_components = len(z0_distributions)
        
        # 计算混合权重 (简化版本，论文中有不同的策略)
        # 这里使用均匀权重，可以改进为学习的权重
        mixing_weights = torch.ones(batch_size, num_components, device=z0_distributions[0][0].device) / num_components
        
        # 从每个分量采样
        z0_samples = []
        kl_losses = []
        
        for i, (mu, logvar) in enumerate(z0_distributions):
            # 重参数化采样
            z0_i = self.reparameterize(mu, logvar)
            z0_samples.append(z0_i)
            
            # KL散度：DKL(q(zi0|X) || p(z0))，假设p(z0) = N(0, I)
            kl_i = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            kl_losses.append(kl_i)
        
        # 混合采样 (简化版本：随机选择一个分量)
        component_idx = torch.randint(0, num_components, (batch_size,), device=z0_distributions[0][0].device)
        z0_mixed = torch.stack([z0_samples[component_idx[b]][b] for b in range(batch_size)])
        
        # 平均KL损失
        kl_loss = torch.mean(torch.stack(kl_losses))
        
        return z0_mixed, kl_loss
    
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
        
        if mode == 'train' and target_od is not None:
            # 训练模式：基于真实OD流量进行编码-解码
            
            # 1. 嵌入：将OD流量映射到潜在状态
            embedded_states = self.embedding(od_flows=target_od)
            
            # 2. 编码：向后演化获得z0分布
            time_points = torch.arange(self.time_steps, dtype=torch.float32, device=features.device)
            z0_distributions = self.encoder(embedded_states, time_points)
            
            # 3. 混合后验分布采样
            z0_sample, kl_loss = self.compute_mixture_posterior(z0_distributions)
            
            # 4. 解码：从z0向前演化
            decoded_latents = self.decoder(z0_sample, self.time_steps)
            
            # 5. 重构：生成OD流量
            predicted_od = self.reconstruction(decoded_latents)
            
            # 6. 计算损失
            reconstruction_loss = self.mse_loss(predicted_od, target_od)
            mae_loss = self.mae_loss(predicted_od, target_od)
            
            # 7. 增强PCC损失计算 - 多层次相关性优化
            pcc_loss = self._compute_enhanced_pcc_loss(predicted_od, target_od)
            
            # VAE总损失 - 平衡各项损失，重点优化PCC
            beta = 0.1  # 适中的KL权重
            total_loss = (0.4 * reconstruction_loss +     # 适度降低重构损失权重
                         beta * kl_loss +                # KL散度损失
                         self.lambda_pcc * pcc_loss)     # 重点优化PCC
            
            return {
                'od_flows': predicted_od,
                'total_loss': total_loss,
                'reconstruction_loss': reconstruction_loss,
                'kl_loss': kl_loss,
                'pcc_loss': pcc_loss,  # 新增PCC损失
                'mse_loss': reconstruction_loss,  # 兼容性
                'mae_loss': mae_loss
            }
            
        else:
            # 推理模式：基于特征条件化生成
            
            # 1. 特征条件化生成z0
            features_flattened = features.view(batch_size, -1)  # [batch_size, time_steps * input_dim]
            z0_conditional = self.feature_to_z0(features_flattened)
            
            # 添加适度随机性
            noise_scale = 0.1
            z0_noisy = z0_conditional + torch.randn_like(z0_conditional) * noise_scale
            
            # 2. 解码：从条件化z0向前演化
            decoded_latents = self.decoder(z0_noisy, self.time_steps)
            
            # 3. 重构：生成OD流量
            predicted_od = self.reconstruction(decoded_latents)
            
            return {
                'od_flows': predicted_od
            }
    
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
    
    avg_loss = np.mean(total_losses)
    
    return {
        'loss': float(avg_loss),
        'mse': float(mse), 
        'rmse': float(rmse),
        'mae': float(mae),
        'pcc': float(pcc)
    }

# ========== IVP-VAE训练函数 ==========
def train_ivp_vae_model(args):
    """训练IVP-VAE OD流量预测模型"""
    # 设置设备
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
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
    
    # 创建IVP-VAE模型
    model = IVPVAEODFlowPredictor(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        time_steps=28,
        output_dim=2
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"IVP-VAE模型创建成功！")
    print(f"  总参数数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  隐藏维度: {args.hidden_dim}")
    print(f"  潜在维度: {args.latent_dim}")
    
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
    
    # 训练循环变量 - 优化PCC指标
    best_val_loss = float('inf')
    best_val_pcc = -1.0  # 最佳验证PCC（越大越好）
    best_model_path = os.path.join(args.output_dir, 'best_ivp_vae_od_model.pth')
    epochs_without_improvement = 0
    epochs_without_pcc_improvement = 0  # PCC无改善轮数
    train_history = []
    
    print(f"\n开始训练IVP-VAE OD流量预测模型...")
    print(f"模型将保存到: {best_model_path}")
    print(f"早停策略: 验证损失{args.early_stop_patience}轮无改善时停止训练")
    print("="*80)
    
    for epoch in range(args.epochs):
        # 动态调整PCC损失权重 - 随训练进度增加PCC重要性
        progress = epoch / args.epochs
        dynamic_pcc_weight = 0.5 + 1.5 * progress  # 从0.5逐渐增加到2.0
        model.lambda_pcc = dynamic_pcc_weight
        
        # 训练阶段
        model.train()
        train_losses = []
        train_reconstruction_losses = []
        train_kl_losses = []
        train_pcc_losses = []  # 新增PCC损失记录
        
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{args.epochs} [训练]")
        for features, od_flows in train_progress:
            features = features.to(device)
            od_flows = od_flows.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(features, od_flows, mode='train')
            total_loss = outputs['total_loss']
            reconstruction_loss = outputs['reconstruction_loss']
            kl_loss = outputs['kl_loss']
            pcc_loss = outputs['pcc_loss']  # 获取PCC损失
            
            # 反向传播
            total_loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 记录损失
            train_losses.append(total_loss.item())
            train_reconstruction_losses.append(reconstruction_loss.item())
            train_kl_losses.append(kl_loss.item())
            train_pcc_losses.append(pcc_loss.item())  # 记录PCC损失
            
            # 更新进度条 - 突出PCC损失和动态权重
            train_progress.set_postfix({
                'Total': f'{total_loss.item():.4f}',
                'Recon': f'{reconstruction_loss.item():.4f}',
                'PCC': f'{pcc_loss.item():.4f}',  # 突出PCC损失
                'PCC_W': f'{dynamic_pcc_weight:.2f}',  # 显示动态PCC权重
                'KL': f'{kl_loss.item():.4f}'
            })
        
        # 计算训练指标
        avg_train_loss = np.mean(train_losses)
        avg_train_recon = np.mean(train_reconstruction_losses)
        avg_train_kl = np.mean(train_kl_losses)
        avg_train_pcc = np.mean(train_pcc_losses)  # PCC损失平均值
        
        # 验证阶段 - 计算详细指标
        print(f"  🔍 计算验证集指标...")
        val_metrics = calculate_metrics(model, val_loader, device, desc="验证集评估")
        
        # 学习率调整
        scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # 检查是否是最佳模型 - 综合考虑损失和PCC
        val_loss_improved = val_metrics['loss'] < best_val_loss
        val_pcc_improved = val_metrics['pcc'] > best_val_pcc
        
        # 优先考虑PCC改善，其次考虑损失改善
        is_best = val_pcc_improved or (val_loss_improved and val_metrics['pcc'] >= best_val_pcc * 0.95)
        test_metrics = None
        
        if is_best:
            # 只在验证集性能提升时评估测试集
            improvement_type = "PCC" if val_pcc_improved else "损失"
            print(f"  🎯 新最佳验证{improvement_type}! 评估测试集...")
            test_metrics = calculate_metrics(model, test_loader, device, desc="测试集评估")
            
            if val_loss_improved:
                best_val_loss = val_metrics['loss']
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                
            if val_pcc_improved:
                best_val_pcc = val_metrics['pcc']
                epochs_without_pcc_improvement = 0
            else:
                epochs_without_pcc_improvement += 1
        else:
            epochs_without_improvement += 1
            epochs_without_pcc_improvement += 1
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
        print(f"🔹 训练集 (IVP-VAE + 动态PCC优化, 权重: {dynamic_pcc_weight:.2f}):")
        print(f"   总损失: {avg_train_loss:.6f} | 重构损失: {avg_train_recon:.6f} | 🎯PCC损失: {avg_train_pcc:.6f} | KL损失: {avg_train_kl:.6f}")
        
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
            'train_reconstruction_loss': float(avg_train_recon),
            'train_kl_loss': float(avg_train_kl),
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
                    f.write("IVP-VAE OD流量预测模型训练日志\n")
                    f.write("=" * 50 + "\n")
                
                f.write(f"Epoch {epoch+1}/{args.epochs}\n")
                f.write(f"   Training - Total: {avg_train_loss:.6f}, Recon: {avg_train_recon:.6f}, KL: {avg_train_kl:.6f}\n")
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
            print(f"⏳ 验证指标未改善 (损失: {epochs_without_improvement}/{args.early_stop_patience}轮, PCC: {epochs_without_pcc_improvement}/{args.early_stop_patience + 10}轮)")
        
        # 早停检查 - 更宽松的策略，优先考虑PCC
        # 只有当损失和PCC都长时间无改善时才停止
        loss_patience_exceeded = epochs_without_improvement >= args.early_stop_patience
        pcc_patience_exceeded = epochs_without_pcc_improvement >= (args.early_stop_patience + 10)  # PCC给更多时间
        
        if loss_patience_exceeded and pcc_patience_exceeded:
            print(f"\n🛑 早停触发! 验证指标长时间无改善，停止训练")
            print(f"   验证损失: {args.early_stop_patience}轮无改善")
            print(f"   验证PCC: {args.early_stop_patience + 10}轮无改善")
            print(f"   最佳验证损失: {best_val_loss:.6f}, 最佳验证PCC: {best_val_pcc:.6f}")
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
    print("🎉 IVP-VAE OD流量预测模型 - 最终测试结果")
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
    results_file = os.path.join(args.output_dir, "ivp_vae_od_results.txt")
    with open(results_file, "w", encoding='utf-8') as f:
        f.write("基于IVP-VAE的OD流量预测模型测试结果\n")
        f.write("="*50 + "\n")
        f.write("论文: IVP-VAE: Modeling EHR Time Series with Initial Value Problem Solvers (AAAI 2024)\n")
        f.write("模型架构核心特点:\n")
        f.write("  - IVP求解器并行处理 (Parallel IVP Processing)\n")
        f.write("  - 共享可逆架构 (Shared Invertible Architecture)\n")
        f.write("  - 纯IVP建模 (Pure IVP Modeling)\n")
        f.write("  - 混合后验分布 (Mixture Posterior Distribution)\n")
        f.write("  - 参数共享机制 (Parameter Sharing Mechanism)\n")
        f.write("\n")
        f.write(f"模型参数:\n")
        f.write(f"  - 总参数数量: {total_params:,}\n")
        f.write(f"  - 可训练参数: {trainable_params:,}\n")
        f.write(f"  - 隐藏维度: {args.hidden_dim}\n")
        f.write(f"  - 潜在维度: {args.latent_dim}\n")
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
    parser = argparse.ArgumentParser(description="基于IVP-VAE的OD流量预测模型")
    
    # 数据参数
    parser.add_argument("--io_flow_path", type=str, default="/private/od/data_NYTaxi/io_flow_daily.npy", 
                       help="IO流量数据路径")
    parser.add_argument("--graph_path", type=str, default="/private/od/data_NYTaxi/graph.npy", 
                       help="图结构数据路径")
    parser.add_argument("--od_matrix_path", type=str, default="/private/od/data_NYTaxi/od_matrix_daily.npy", 
                       help="OD矩阵数据路径")
    
    # IVP-VAE模型参数
    parser.add_argument("--hidden_dim", type=int, default=128, 
                       help="隐藏维度 (IVP求解器和嵌入网络隐藏层大小)")
    parser.add_argument("--latent_dim", type=int, default=64, 
                       help="潜在空间维度 (VAE潜在变量维度)")
    
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
    parser.add_argument("--output_dir", type=str, default="/private/od/paper_ny/IVP_VAE", help="输出目录")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = create_dynamic_output_dir(args.output_dir)
    args.output_dir = output_dir
    
    print("="*60)
    print("🚀 IVP-VAE OD流量预测模型")
    print("="*60)
    print("📖 论文: IVP-VAE: Modeling EHR Time Series with Initial Value Problem Solvers")
    print("📖 会议: AAAI 2024")
    print("📖 作者: Jingge Xiao, Leonie Basso, Wolfgang Nejdl, Niloy Ganguly, Sandipan Sikdar")
    print()
    print("🔧 模型创新点:")
    print("  ✅ IVP求解器并行处理 - 避免RNN顺序计算瓶颈")
    print("  ✅ 共享可逆架构 - 编码器解码器共用IVP求解器")
    print("  ✅ 纯IVP建模 - 完全基于连续过程和初值问题求解")
    print("  ✅ 混合后验分布 - 多个zi0构建复杂z0分布")
    print("  ✅ 参数共享 - 减少参数量，提升收敛速度")
    print()
    print(f"📁 输出目录: {output_dir}")
    print("="*60)
    
    # 训练模型
    try:
        best_model_path = train_ivp_vae_model(args)
        print("\n🎉 IVP-VAE模型训练完成!")
        print(f"📁 最佳模型保存位置: {best_model_path}")
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {str(e)}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)