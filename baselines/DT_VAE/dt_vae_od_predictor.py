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
    dynamic_dir = os.path.join(base_dir, f"dt_vae_run_{timestamp}")
    os.makedirs(dynamic_dir, exist_ok=True)
    return dynamic_dir

# ========== DT-VAE 核心组件 ==========

class CumulativeDifferenceTransform:
    """累积差分转换器 - DT-VAE的核心创新（修正版）
    
    将原始时间序列转换为累积差分表示：
    - 前向转换: γt = Σ(xi+1 - xi) for i=0 to t-1 (真正的累积差分)
    - 逆向转换: xt = x0 + Σγi for i=1 to t (从累积差分重构)
    
    这种转换避免了传统递归生成中的误差累积问题
    """
    
    @staticmethod
    def to_cumulative_diff(sequence):
        """
        转换为真正的累积差分表示
        Args:
            sequence: [batch_size, time_steps, dim] 原始序列
        Returns:
            cumulative_diff: [batch_size, time_steps-1, dim] 累积差分
            initial_values: [batch_size, 1, dim] 初始值
        """
        batch_size, time_steps, dim = sequence.shape
        
        # 保存初始值 x0
        initial_values = sequence[:, 0:1, :]  # [batch_size, 1, dim]
        
        # 计算逐步差分 dt = xt+1 - xt
        step_diff = sequence[:, 1:, :] - sequence[:, :-1, :]  # [batch_size, time_steps-1, dim]
        
        # 计算累积差分 γt = Σ(xi+1 - xi) for i=0 to t-1
        cumulative_diff = torch.cumsum(step_diff, dim=1)  # [batch_size, time_steps-1, dim]
        
        return cumulative_diff, initial_values
    
    @staticmethod  
    def from_cumulative_diff(cumulative_diff, initial_values):
        """
        从累积差分重构原序列
        Args:
            cumulative_diff: [batch_size, time_steps-1, dim] 累积差分
            initial_values: [batch_size, 1, dim] 初始值
        Returns:
            sequence: [batch_size, time_steps, dim] 重构的序列
        """
        # 重构序列: xt = x0 + γt 
        # 其中 γt 是累积差分，表示从初始值到第t步的总变化量
        reconstructed_sequence = initial_values + cumulative_diff  # [batch_size, time_steps-1, dim]
        
        # 拼接初始值和重构序列
        sequence = torch.cat([initial_values, reconstructed_sequence], dim=1)  # [batch_size, time_steps, dim]
        
        return sequence

class DTVAEEncoder(nn.Module):
    """DT-VAE编码器 - 论文中的qφ(zt|z1:t-1, γ1:t)
    
    编码器的作用：
    1. 将观测的累积差分γ1:t和特征features编码到潜在空间
    2. 输出潜在变量的均值μt和方差σt
    3. 支持条件编码，即考虑历史潜在变量z1:t-1
    """
    
    def __init__(self, feature_dim=6, cumulative_diff_dim=2, hidden_dim=128, latent_dim=64):
        super(DTVAEEncoder, self).__init__()
        
        self.feature_dim = feature_dim
        self.cumulative_diff_dim = cumulative_diff_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # 特征编码器 - 处理输入特征features
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2)
        )
        
        # 累积差分编码器 - 处理累积差分γt  
        self.cumulative_diff_encoder = nn.Sequential(
            nn.Linear(cumulative_diff_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(0.2)
        )
        
        # RNN编码器 - 处理时序信息，对应论文中的fφ1
        self.rnn_encoder = nn.LSTM(
            input_size=hidden_dim // 2 + hidden_dim // 4,  # 特征 + 累积差分
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # 潜在变量参数网络 - 输出μt和σt，对应论文中的fφ2和fφ3
        self.mu_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
        
        self.logvar_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
        
    def forward(self, features, cumulative_diff):
        """
        前向传播
        Args:
            features: [batch_size, time_steps, feature_dim] 输入特征
            cumulative_diff: [batch_size, time_steps-1, cumulative_diff_dim] 累积差分
        Returns:
            mu: [batch_size, time_steps-1, latent_dim] 潜在变量均值序列
            logvar: [batch_size, time_steps-1, latent_dim] 潜在变量对数方差序列
        """
        batch_size, time_steps, _ = features.shape
        
        # 对累积差分对应的特征进行编码 (跳过第一个时间步)
        features_encoded = self.feature_encoder(features[:, 1:, :])  # [batch_size, time_steps-1, hidden_dim//2]
        
        # 对累积差分进行编码
        cumulative_diff_encoded = self.cumulative_diff_encoder(cumulative_diff)  # [batch_size, time_steps-1, hidden_dim//4]
        
        # 拼接特征和累积差分编码
        combined_input = torch.cat([features_encoded, cumulative_diff_encoded], dim=-1)  # [batch_size, time_steps-1, hidden_dim//2 + hidden_dim//4]
        
        # RNN编码时序信息
        rnn_output, _ = self.rnn_encoder(combined_input)  # [batch_size, time_steps-1, hidden_dim]
        
        # 计算潜在变量参数
        mu = self.mu_net(rnn_output)        # [batch_size, time_steps-1, latent_dim]
        logvar = self.logvar_net(rnn_output)  # [batch_size, time_steps-1, latent_dim]
        
        return mu, logvar

class DTVAEDecoder(nn.Module):
    """DT-VAE解码器 - 论文中的pθ(γt|z1:t)
    
    解码器的作用：
    1. 从潜在变量序列z1:t生成累积差分γt
    2. 使用RNN结构捕捉时序依赖，对应论文中的fθ1
    3. 输出累积差分的重构值
    """
    
    def __init__(self, latent_dim=64, hidden_dim=128, output_dim=2):
        super(DTVAEDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 潜在变量投影层
        self.latent_projection = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # RNN解码器 - 对应论文中的fθ1
        self.rnn_decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # 输出层 - 对应论文中的fθ2，生成μt,θ和σt,θ
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, latent_sequence):
        """
        前向传播  
        Args:
            latent_sequence: [batch_size, time_steps-1, latent_dim] 潜在变量序列
        Returns:
            reconstructed_cumulative_diff: [batch_size, time_steps-1, output_dim] 重构的累积差分
        """
        # 投影潜在变量
        latent_projected = self.latent_projection(latent_sequence)  # [batch_size, time_steps-1, hidden_dim]
        
        # RNN解码
        rnn_output, _ = self.rnn_decoder(latent_projected)  # [batch_size, time_steps-1, hidden_dim]
        
        # 生成累积差分重构值
        reconstructed_cumulative_diff = self.output_net(rnn_output)  # [batch_size, time_steps-1, output_dim]
        
        return reconstructed_cumulative_diff

class DTVAEODFlowPredictor(nn.Module):
    """基于DT-VAE的OD流量预测模型
    
    主要创新点：
    1. 累积差分学习：避免传统时序预测中的误差累积问题
    2. VAE框架：通过变分推断学习复杂的时序分布  
    3. 条件生成：结合输入特征进行条件化预测
    4. 理论支撑：基于inflow-outflow的时间序列数学建模
    """
    
    def __init__(self, input_dim=6, hidden_dim=128, latent_dim=64, time_steps=28, output_dim=2):
        super(DTVAEODFlowPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.time_steps = time_steps
        self.output_dim = output_dim
        
        # 累积差分转换器
        self.cumulative_diff_transform = CumulativeDifferenceTransform()
        
        # DT-VAE编码器和解码器
        self.encoder = DTVAEEncoder(
            feature_dim=input_dim,
            cumulative_diff_dim=output_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim
        )
        
        self.decoder = DTVAEDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
        
        # 特征到初始值的映射网络 - 修正推理模式的关键
        self.feature_to_initial = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 4, output_dim)
        )
        
        # 特征条件化的潜在变量生成网络 - 用于推理阶段
        self.feature_to_latent = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        
    def reparameterize(self, mu, logvar):
        """重参数化技巧 - VAE的核心组件"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
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
            # 训练模式：使用真实的target_od进行编码和解码 + 训练推理网络
            
            # === 主要的VAE训练路径 ===
            # 1. 将目标OD流量转换为累积差分
            target_cumulative_diff, target_initial_values = self.cumulative_diff_transform.to_cumulative_diff(target_od)
            
            # 2. 编码：得到潜在变量的分布参数
            mu, logvar = self.encoder(features, target_cumulative_diff)
            
            # 3. 重参数化：采样潜在变量
            latent_sequence = self.reparameterize(mu, logvar)
            
            # 4. 解码：重构累积差分
            reconstructed_cumulative_diff = self.decoder(latent_sequence)
            
            # 5. 转换回OD流量
            predicted_od = self.cumulative_diff_transform.from_cumulative_diff(
                reconstructed_cumulative_diff, target_initial_values
            )
            
            # === 推理网络训练路径（关键修复）===
            # 6. 训练推理时使用的网络，让它们学习正确的映射关系
            
            # 6.1 训练特征到初始值的映射
            initial_features = features[:, 0, :]
            predicted_initial_from_features = self.feature_to_initial(initial_features).unsqueeze(1)
            
            # 6.2 训练特征到潜在变量的映射
            inference_latent_sequence = []
            for t in range(1, self.time_steps):
                feature_t = features[:, t, :]
                latent_mean_from_features = self.feature_to_latent(feature_t)
                inference_latent_sequence.append(latent_mean_from_features)
            inference_latent_sequence = torch.stack(inference_latent_sequence, dim=1)
            
            # === 损失计算 ===
            # 主要损失：重构损失
            reconstruction_loss = self.mse_loss(predicted_od, target_od)
            
            # VAE损失：KL散度
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
            
            # 推理网络训练损失（关键修复）
            # 让推理网络学习预测正确的初始值
            initial_prediction_loss = self.mse_loss(predicted_initial_from_features, target_initial_values)
            
            # 让推理网络学习预测与编码器类似的潜在变量
            latent_prediction_loss = self.mse_loss(inference_latent_sequence, mu)
            
            # 总损失组合
            beta = 0.1          # KL权重
            gamma = 0.5         # 推理网络权重
            
            total_loss = (reconstruction_loss + 
                         beta * kl_loss + 
                         gamma * initial_prediction_loss + 
                         gamma * latent_prediction_loss)
            
            # 额外的评估指标
            mae_loss = self.mae_loss(predicted_od, target_od)
            
            return {
                'od_flows': predicted_od,
                'total_loss': total_loss,
                'reconstruction_loss': reconstruction_loss,
                'kl_loss': kl_loss,
                'initial_prediction_loss': initial_prediction_loss,
                'latent_prediction_loss': latent_prediction_loss,
                'mse_loss': reconstruction_loss,  # 为了兼容性
                'mae_loss': mae_loss
            }
            
        else:
            # 推理模式：使用特征条件化生成预测（修正版）
            # 关键改进：利用输入特征而不是纯随机采样
            
            # 1. 从特征预测初始值 - 核心改进
            # 使用第一个时间步的特征来预测初始OD流量
            initial_features = features[:, 0, :]  # [batch_size, input_dim]
            predicted_initial_values = self.feature_to_initial(initial_features)  # [batch_size, output_dim]
            predicted_initial_values = predicted_initial_values.unsqueeze(1)  # [batch_size, 1, output_dim]
            
            # 2. 基于特征生成条件化的潜在变量序列
            # 不再使用纯随机采样，而是基于每个时间步的特征
            latent_sequence = []
            for t in range(1, self.time_steps):  # 从第2个时间步开始
                feature_t = features[:, t, :]  # [batch_size, input_dim]
                
                # 基于特征生成潜在变量的条件均值
                latent_mean = self.feature_to_latent(feature_t)  # [batch_size, latent_dim]
                
                # 添加适度的随机性以保持生成多样性
                latent_std = 0.1  # 可调节的标准差
                latent_noise = torch.randn_like(latent_mean) * latent_std
                latent_t = latent_mean + latent_noise
                
                latent_sequence.append(latent_t)
            
            latent_sequence = torch.stack(latent_sequence, dim=1)  # [batch_size, time_steps-1, latent_dim]
            
            # 3. 使用解码器生成累积差分
            predicted_cumulative_diff = self.decoder(latent_sequence)
            
            # 4. 从累积差分重构OD流量
            predicted_od = self.cumulative_diff_transform.from_cumulative_diff(
                predicted_cumulative_diff, predicted_initial_values
            )
            
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

# ========== DT-VAE训练函数 ==========
def train_dt_vae_model(args):
    """训练DT-VAE OD流量预测模型"""
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
    
    # 创建DT-VAE模型
    model = DTVAEODFlowPredictor(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        time_steps=28,
        output_dim=2
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DT-VAE模型创建成功！")
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
    
    # 训练循环变量
    best_val_loss = float('inf')
    best_model_path = os.path.join(args.output_dir, 'best_dt_vae_od_model.pth')
    epochs_without_improvement = 0
    train_history = []
    
    print(f"\n开始训练DT-VAE OD流量预测模型...")
    print(f"模型将保存到: {best_model_path}")
    print(f"早停策略: 验证损失{args.early_stop_patience}轮无改善时停止训练")
    print("="*80)
    
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_losses = []
        train_reconstruction_losses = []
        train_kl_losses = []
        train_initial_pred_losses = []
        train_latent_pred_losses = []
        
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
            initial_pred_loss = outputs['initial_prediction_loss']
            latent_pred_loss = outputs['latent_prediction_loss']
            
            # 反向传播
            total_loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 记录损失
            train_losses.append(total_loss.item())
            train_reconstruction_losses.append(reconstruction_loss.item())
            train_kl_losses.append(kl_loss.item())
            train_initial_pred_losses.append(initial_pred_loss.item())
            train_latent_pred_losses.append(latent_pred_loss.item())
            
            # 更新进度条
            train_progress.set_postfix({
                'Total': f'{total_loss.item():.4f}',
                'Recon': f'{reconstruction_loss.item():.4f}',
                'KL': f'{kl_loss.item():.4f}',
                'Init': f'{initial_pred_loss.item():.4f}',
                'Lat': f'{latent_pred_loss.item():.4f}'
            })
        
        # 计算训练指标
        avg_train_loss = np.mean(train_losses)
        avg_train_recon = np.mean(train_reconstruction_losses)
        avg_train_kl = np.mean(train_kl_losses)
        avg_train_initial_pred = np.mean(train_initial_pred_losses)
        avg_train_latent_pred = np.mean(train_latent_pred_losses)
        
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
        print(f"   总损失: {avg_train_loss:.6f} | 重构损失: {avg_train_recon:.6f} | KL损失: {avg_train_kl:.6f}")
        print(f"   推理损失: 初始值预测={avg_train_initial_pred:.6f} | 潜在变量预测={avg_train_latent_pred:.6f}")
        
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
                    f.write("DT-VAE OD流量预测模型训练日志\n")
                    f.write("=" * 50 + "\n")
                
                f.write(f"Epoch {epoch+1}/{args.epochs}\n")
                f.write(f"   Training - Total: {avg_train_loss:.6f}, Recon: {avg_train_recon:.6f}, KL: {avg_train_kl:.6f}, Init: {avg_train_initial_pred:.6f}, Lat: {avg_train_latent_pred:.6f}\n")
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
    print("🎉 DT-VAE OD流量预测模型 - 最终测试结果")
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
    results_file = os.path.join(args.output_dir, "dt_vae_od_results.txt")
    with open(results_file, "w", encoding='utf-8') as f:
        f.write("基于DT-VAE的OD流量预测模型测试结果\n")
        f.write("="*50 + "\n")
        f.write("论文: Cumulative Difference Learning VAE for Time-Series with Temporally Correlated Inflow-Outflow (AAAI 2024)\n")
        f.write("模型架构核心特点:\n")
        f.write("  - 累积差分学习 (Cumulative Difference Learning)\n")
        f.write("  - VAE变分自编码器框架 (Variational Autoencoder Framework)\n")
        f.write("  - 避免误差累积 (Error Accumulation Avoidance)\n")
        f.write("  - 时间相关性建模 (Temporal Correlation Modeling)\n")
        f.write("  - inflow-outflow理论支撑 (Inflow-Outflow Theoretical Foundation)\n")
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
    parser = argparse.ArgumentParser(description="基于DT-VAE的OD流量预测模型")
    
    # 数据参数
    parser.add_argument("--io_flow_path", type=str, default="/private/od/data_NYTaxi/io_flow_daily.npy", 
                       help="IO流量数据路径")
    parser.add_argument("--graph_path", type=str, default="/private/od/data_NYTaxi/graph.npy", 
                       help="图结构数据路径")
    parser.add_argument("--od_matrix_path", type=str, default="/private/od/data_NYTaxi/od_matrix_daily.npy", 
                       help="OD矩阵数据路径")
    
    # DT-VAE模型参数
    parser.add_argument("--hidden_dim", type=int, default=128, 
                       help="隐藏维度 (编码器解码器隐藏层大小)")
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
    parser.add_argument("--output_dir", type=str, default="/private/od/paper_ny/DT_VAE", help="输出目录")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = create_dynamic_output_dir(args.output_dir)
    args.output_dir = output_dir
    
    print("="*60)
    print("🚀 DT-VAE OD流量预测模型")
    print("="*60)
    print("📖 论文: Cumulative Difference Learning VAE for Time-Series with Temporally Correlated Inflow-Outflow")
    print("📖 会议: AAAI 2024")
    print("📖 作者: Tianchun Li, Chengxiang Wu, Pengyi Shi, Xiaoqian Wang")
    print()
    print("🔧 模型创新点:")
    print("  ✅ 累积差分学习 - 避免传统时序生成中的误差累积")
    print("  ✅ VAE变分框架 - 通过变分推断学习复杂时序分布")
    print("  ✅ 条件生成 - 结合输入特征进行条件化预测")
    print("  ✅ 理论支撑 - 基于inflow-outflow的数学建模")
    print("  ✅ 时间依赖 - 通过潜在变量序列捕捉时间相关性")
    print()
    print(f"📁 输出目录: {output_dir}")
    print("="*60)
    
    # 训练模型
    try:
        best_model_path = train_dt_vae_model(args)
        print("\n🎉 DT-VAE模型训练完成!")
        print(f"📁 最佳模型保存位置: {best_model_path}")
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {str(e)}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)