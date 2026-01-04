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
    dynamic_dir = os.path.join(base_dir, f"psa_gan_run_{timestamp}")
    os.makedirs(dynamic_dir, exist_ok=True)
    return dynamic_dir

# ========== PSA-GAN 核心组件 ==========

class SpectralNorm(nn.Module):
    """光谱归一化模块 - PSA-GAN核心组件之一
    
    用于约束卷积层的Lipschitz常数，稳定训练过程
    Reference: Spectral Normalization for Generative Adversarial Networks (Miyato et al., 2018)
    """
    def __init__(self, module, power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.power_iterations = power_iterations
        if not hasattr(module, 'weight'):
            raise ValueError("Module must have 'weight' parameter")
        
        w = module.weight.data
        height = w.size(0)
        width = w.view(height, -1).size(1)
        
        u = nn.Parameter(w.new_empty(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.new_empty(width).normal_(0, 1), requires_grad=False)
        u.data = F.normalize(u.data, dim=0, eps=1e-12)
        v.data = F.normalize(v.data, dim=0, eps=1e-12)
        
        self.register_parameter('weight_u', u)
        self.register_parameter('weight_v', v)
        
    def _update_u_v(self):
        u = getattr(self, 'weight_u')
        v = getattr(self, 'weight_v')
        w = self.module.weight.data
        
        height = w.size(0)
        for _ in range(self.power_iterations):
            v.data = F.normalize(torch.mv(w.view(height, -1).t(), u), dim=0, eps=1e-12)
            u.data = F.normalize(torch.mv(w.view(height, -1), v), dim=0, eps=1e-12)
        
        sigma = u.dot(w.view(height, -1).mv(v))
        return sigma
        
    def forward(self, *args):
        if self.training:
            sigma = self._update_u_v()
        else:
            u = getattr(self, 'weight_u')
            v = getattr(self, 'weight_v')
            w = self.module.weight.data
            sigma = u.dot(w.view(w.size(0), -1).mv(v))
        
        weight = self.module.weight / sigma.expand_as(self.module.weight)
        return F.conv1d(args[0], weight, self.module.bias, self.module.stride,
                       self.module.padding, self.module.dilation, self.module.groups)

class SelfAttention(nn.Module):
    """自注意力模块 - PSA-GAN核心组件
    
    用于捕捉时间序列中的长程依赖关系
    Reference: Self-Attention Generative Adversarial Networks (Zhang et al., 2019)
    """
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        
        # Query, Key, Value投影层
        self.query_conv = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        
        # 输出投影层
        self.out_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        
        # 可学习的缩放参数γ，初始化为0
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # Softmax用于注意力权重计算
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        """
        Args:
            x: 输入特征 [batch_size, channels, length]
        Returns:
            输出特征 [batch_size, channels, length]
        """
        batch_size, channels, length = x.size()
        
        # 计算Query, Key, Value
        query = self.query_conv(x).view(batch_size, -1, length).permute(0, 2, 1)  # [B, L, C//8]
        key = self.key_conv(x).view(batch_size, -1, length)  # [B, C//8, L]
        value = self.value_conv(x).view(batch_size, -1, length).permute(0, 2, 1)  # [B, L, C]
        
        # 计算注意力权重
        attention = torch.bmm(query, key)  # [B, L, L]
        attention = self.softmax(attention)
        
        # 应用注意力权重到value
        attended_value = torch.bmm(attention, value)  # [B, L, C]
        attended_value = attended_value.permute(0, 2, 1).contiguous()  # [B, C, L]
        
        # 输出投影
        out = self.out_conv(attended_value)
        
        # 残差连接与可学习缩放
        out = self.gamma * out + x
        
        return out

class ResidualSelfAttentionBlock(nn.Module):
    """残差自注意力块 - PSA-GAN主要构建块
    
    结合卷积、自注意力、光谱归一化和残差连接
    """
    def __init__(self, in_channels, out_channels=None):
        super(ResidualSelfAttentionBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 卷积层 + 光谱归一化
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.spectral_norm = SpectralNorm(self.conv)
        
        # 激活函数
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        
        # 自注意力模块
        self.self_attention = SelfAttention(out_channels)
        
        # 残差连接的投影层（如果输入输出维度不同）
        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = None
            
    def forward(self, x):
        """前向传播"""
        # 保存残差连接
        residual = x
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
        
        # 卷积 + 激活 + 光谱归一化
        out = self.activation(self.spectral_norm(x))
        
        # 自注意力
        out = self.self_attention(out)
        
        # 残差连接
        out = out + residual
        
        return out

class ProgressiveFeatureExtractor(nn.Module):
    """渐进式特征提取器 - 模拟PSA-GAN的渐进式增长
    
    从粗粒度特征逐步向细粒度特征建模
    """
    def __init__(self, input_dim=6, hidden_channels=64, num_blocks=3):
        super(ProgressiveFeatureExtractor, self).__init__()
        self.input_dim = input_dim
        self.hidden_channels = hidden_channels
        self.num_blocks = num_blocks
        
        # 初始投影层
        self.input_projection = nn.Conv1d(input_dim, hidden_channels, kernel_size=1)
        
        # 渐进式特征提取块
        self.feature_blocks = nn.ModuleList()
        for i in range(num_blocks):
            block = ResidualSelfAttentionBlock(hidden_channels, hidden_channels)
            self.feature_blocks.append(block)
            
        # 上采样层（模拟渐进式增长中的分辨率提升）
        self.upsample_layers = nn.ModuleList()
        for i in range(num_blocks - 1):
            self.upsample_layers.append(nn.Upsample(scale_factor=1.0, mode='linear', align_corners=False))
        
    def forward(self, x):
        """
        Args:
            x: 输入特征 [batch_size, time_steps, input_dim]
        Returns:
            多尺度特征列表
        """
        # 转换为卷积格式: [batch_size, input_dim, time_steps]
        x = x.transpose(1, 2)
        
        # 输入投影
        x = self.input_projection(x)
        
        # 渐进式特征提取
        features = []
        for i, block in enumerate(self.feature_blocks):
            x = block(x)
            features.append(x)
            
            # 可选的上采样（这里保持维度不变，主要用于概念展示）
            if i < len(self.upsample_layers):
                x = self.upsample_layers[i](x)
        
        return features

class PSAGANODFlowPredictor(nn.Module):
    """基于PSA-GAN架构的OD流量预测模型
    
    主要创新点：
    1. 使用渐进式特征提取替代传统的单一编码器
    2. 集成自注意力机制捕捉长程时间依赖
    3. 采用光谱归一化稳定训练
    4. 多尺度特征融合提升预测精度
    """
    def __init__(self, input_dim=6, hidden_channels=64, time_steps=28, num_blocks=3):
        super(PSAGANODFlowPredictor, self).__init__()
        self.input_dim = input_dim
        self.hidden_channels = hidden_channels
        self.time_steps = time_steps
        self.num_blocks = num_blocks
        
        # PSA-GAN特征提取器
        self.feature_extractor = ProgressiveFeatureExtractor(
            input_dim=input_dim,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks
        )
        
        # 多尺度特征融合
        self.feature_fusion = nn.Conv1d(
            hidden_channels * num_blocks, hidden_channels, 
            kernel_size=3, padding=1
        )
        
        # 最终预测层
        self.predictor_head = nn.Sequential(
            ResidualSelfAttentionBlock(hidden_channels, hidden_channels),
            nn.Conv1d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_channels // 2, 2, kernel_size=1),  # 输出2维OD流量
        )
        
        # 损失函数相关
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        
    def forward(self, features, target_od=None, mode='train'):
        """
        前向传播
        Args:
            features: 输入特征 [batch_size, time_steps=28, input_dim=6]
            target_od: 目标OD流量 [batch_size, time_steps=28, 2]
            mode: 'train' 或 'eval'
        Returns:
            结果字典
        """
        batch_size = features.size(0)
        
        # PSA-GAN特征提取
        multi_scale_features = self.feature_extractor(features)
        
        # 多尺度特征融合
        fused_features = torch.cat(multi_scale_features, dim=1)  # [B, C*num_blocks, T]
        fused_features = self.feature_fusion(fused_features)  # [B, C, T]
        
        # 预测OD流量
        predicted_od = self.predictor_head(fused_features)  # [B, 2, T]
        predicted_od = predicted_od.transpose(1, 2)  # [B, T, 2]
        
        if mode == 'train' and target_od is not None:
            # 计算损失
            mse_loss = self.mse_loss(predicted_od, target_od)
            mae_loss = self.mae_loss(predicted_od, target_od)
            
            # PSA-GAN style loss combination (参考论文中的损失组合)
            total_loss = mse_loss + 0.5 * mae_loss
            
            return {
                'od_flows': predicted_od,
                'total_loss': total_loss,
                'mse_loss': mse_loss,
                'mae_loss': mae_loss
            }
        else:
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
        
        # 数据一致性验证 - 确保所有数据的节点维度匹配
        assert self.graph.shape[0] == self.graph.shape[1], f"图数据必须是方阵: {self.graph.shape}"
        assert self.io_flow.shape[0] == self.graph.shape[0], f"IO流量节点数与图节点数不匹配: {self.io_flow.shape[0]} vs {self.graph.shape[0]}"
        assert self.od_matrix.shape[0] == self.graph.shape[0], f"OD矩阵节点数与图节点数不匹配: {self.od_matrix.shape[0]} vs {self.graph.shape[0]}"
        assert self.od_matrix.shape[1] == self.graph.shape[0], f"OD矩阵节点数与图节点数不匹配: {self.od_matrix.shape[1]} vs {self.graph.shape[0]}"
        
        # 验证人口密度数据数量
        if self.station_data and len(self.station_data) != self.num_nodes:
            print(f"⚠️ 人口密度数据数量({len(self.station_data)})与节点数量({self.num_nodes})不匹配")
        
        print(f"✅ 数据一致性验证通过: {self.num_nodes}个节点, {self.time_steps}个时间步")
        
        # 站点对列表 - 与原版保持一致，使用所有站点对
        self.od_pairs = []
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                self.od_pairs.append((i, j))
        
        print(f"生成{len(self.od_pairs)}个站点对用于训练")
        
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

# ========== PSA-GAN训练函数 ==========
def train_psa_gan_model(args):
    """训练PSA-GAN OD流量预测模型"""
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
    
    # 创建PSA-GAN模型
    model = PSAGANODFlowPredictor(
        input_dim=6,
        hidden_channels=args.hidden_channels,
        time_steps=28,
        num_blocks=args.num_blocks
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PSA-GAN模型创建成功！")
    print(f"  总参数数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  隐藏通道数: {args.hidden_channels}")
    print(f"  特征提取块数: {args.num_blocks}")
    
    # 优化器 - 使用Adam优化器（论文推荐）
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        betas=(0.5, 0.999),  # PSA-GAN论文中使用的beta值
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.patience, verbose=True, min_lr=1e-6
    )
    
    # 训练循环变量
    best_val_loss = float('inf')
    best_model_path = os.path.join(args.output_dir, 'best_psa_gan_od_model.pth')
    epochs_without_improvement = 0
    train_history = []
    
    print(f"\n开始训练PSA-GAN OD流量预测模型...")
    print(f"模型将保存到: {best_model_path}")
    print(f"早停策略: 验证损失{args.early_stop_patience}轮无改善时停止训练")
    print("="*80)
    
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_losses = []
        train_mse_losses = []
        train_mae_losses = []
        
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{args.epochs} [训练]")
        for features, od_flows in train_progress:
            features = features.to(device)
            od_flows = od_flows.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(features, od_flows, mode='train')
            total_loss = outputs['total_loss']
            mse_loss = outputs['mse_loss']
            mae_loss = outputs['mae_loss']
            
            # 反向传播
            total_loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 记录损失
            train_losses.append(total_loss.item())
            train_mse_losses.append(mse_loss.item())
            train_mae_losses.append(mae_loss.item())
            
            # 更新进度条
            train_progress.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'MSE': f'{mse_loss.item():.4f}',
                'MAE': f'{mae_loss.item():.4f}'
            })
        
        # 计算训练指标
        avg_train_loss = np.mean(train_losses)
        avg_train_mse = np.mean(train_mse_losses)
        avg_train_mae = np.mean(train_mae_losses)
        
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
        print(f"   总损失: {avg_train_loss:.6f} | MSE: {avg_train_mse:.6f} | MAE: {avg_train_mae:.6f}")
        
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
            'train_mse': float(avg_train_mse),
            'train_mae': float(avg_train_mae),
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
                    f.write("PSA-GAN OD流量预测模型训练日志\n")
                    f.write("=" * 50 + "\n")
                
                f.write(f"Epoch {epoch+1}/{args.epochs}\n")
                f.write(f"   Training - Loss: {avg_train_loss:.6f}, MSE: {avg_train_mse:.6f}, MAE: {avg_train_mae:.6f}\n")
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
    print("🎉 PSA-GAN OD流量预测模型 - 最终测试结果")
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
    results_file = os.path.join(args.output_dir, "psa_gan_od_results.txt")
    with open(results_file, "w", encoding='utf-8') as f:
        f.write("基于PSA-GAN的OD流量预测模型测试结果\n")
        f.write("="*50 + "\n")
        f.write("论文: PSA-GAN: Progressive Self Attention GANs for Synthetic Time Series (ICLR 2022)\n")
        f.write("模型架构核心特点:\n")
        f.write("  - 渐进式特征提取 (Progressive Feature Extraction)\n")
        f.write("  - 自注意力机制 (Self-Attention Mechanism)\n")
        f.write("  - 光谱归一化 (Spectral Normalization)\n")
        f.write("  - 残差连接 (Residual Connections)\n")
        f.write("  - 多尺度特征融合 (Multi-scale Feature Fusion)\n")
        f.write("\n")
        f.write(f"模型参数:\n")
        f.write(f"  - 总参数数量: {total_params:,}\n")
        f.write(f"  - 可训练参数: {trainable_params:,}\n")
        f.write(f"  - 隐藏通道数: {args.hidden_channels}\n")
        f.write(f"  - 特征提取块数: {args.num_blocks}\n")
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
    parser = argparse.ArgumentParser(description="基于PSA-GAN的OD流量预测模型")
    
    # 数据参数 - 更新为52节点数据结构路径
    parser.add_argument("--io_flow_path", type=str, default="/private/od/data_NYTaxi/io_flow_daily.npy", 
                       help="IO流量数据路径")
    parser.add_argument("--graph_path", type=str, default="/private/od/data_NYTaxi/graph.npy", 
                       help="图结构数据路径")
    parser.add_argument("--od_matrix_path", type=str, default="/private/od/data_NYTaxi/od_matrix_daily.npy", 
                       help="OD矩阵数据路径")
    
    # PSA-GAN模型参数
    parser.add_argument("--hidden_channels", type=int, default=64, 
                       help="隐藏通道数 (PSA-GAN特征维度)")
    parser.add_argument("--num_blocks", type=int, default=3, 
                       help="渐进式特征提取块数量")
    
    # 训练参数  
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.0002, help="学习率 (PSA-GAN推荐)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="权重衰减")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="测试集比例 (固定8:1:1划分)")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例 (固定8:1:1划分)")
    parser.add_argument("--seed", type=int, default=82, help="随机种子")
    
    # 早停和学习率调整参数
    parser.add_argument("--early_stop_patience", type=int, default=15, help="早停策略：验证损失多少轮无改善时停止训练")
    parser.add_argument("--patience", type=int, default=8, help="学习率调整策略：验证损失多少轮无改善时降低学习率")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="/private/od/paper_ny/PSA_GAN", help="输出目录")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = create_dynamic_output_dir(args.output_dir)
    args.output_dir = output_dir
    
    print("="*60)
    print("🚀 PSA-GAN OD流量预测模型")
    print("="*60)
    print("📖 论文: PSA-GAN: Progressive Self Attention GANs for Synthetic Time Series")
    print("📖 会议: ICLR 2022")
    print("📖 作者: Paul Jeha, et al.")
    print()
    print("🔧 模型创新点:")
    print("  ✅ 渐进式特征提取 - 从粗粒度到细粒度建模")
    print("  ✅ 自注意力机制 - 捕捉长程时间依赖")
    print("  ✅ 光谱归一化 - 稳定训练过程")
    print("  ✅ 残差连接 - 改善梯度流动")
    print("  ✅ 多尺度特征融合 - 提升预测精度")
    print()
    print(f"📁 输出目录: {output_dir}")
    print("="*60)
    
    # 训练模型
    try:
        best_model_path = train_psa_gan_model(args)
        print("\n🎉 PSA-GAN模型训练完成!")
        print(f"📁 最佳模型保存位置: {best_model_path}")
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {str(e)}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)