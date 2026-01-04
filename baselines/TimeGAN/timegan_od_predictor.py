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
    dynamic_dir = os.path.join(base_dir, f"timegan_run_{timestamp}")
    os.makedirs(dynamic_dir, exist_ok=True)
    return dynamic_dir

# ========== TimeGAN 核心组件 ==========

class EmbeddingNetwork(nn.Module):
    """TimeGAN Embedding Network
    
    将原始时序特征映射到低维潜在空间
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(EmbeddingNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 使用GRU作为循环网络（论文中使用的是RNN，这里用GRU提升效果）
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # 输出投影层
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Sigmoid()  # 论文中使用sigmoid激活
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, input_dim]
        Returns:
            h: [batch_size, seq_len, hidden_dim] - 潜在表示
        """
        # RNN前向传播
        h, _ = self.rnn(x)  # [batch_size, seq_len, hidden_dim]
        
        # 输出投影和激活
        h = self.output_projection(h)
        h = self.activation(h)
        
        return h

class RecoveryNetwork(nn.Module):
    """TimeGAN Recovery Network
    
    将潜在表示映射回原始特征空间
    """
    def __init__(self, hidden_dim, output_dim):
        super(RecoveryNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 使用前馈网络进行恢复（论文推荐的架构）
        self.recovery_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # 确保输出在合理范围内
        )
        
    def forward(self, h):
        """
        Args:
            h: [batch_size, seq_len, hidden_dim] - 潜在表示
        Returns:
            x_reconstructed: [batch_size, seq_len, output_dim] - 重建的特征
        """
        return self.recovery_net(h)

class GeneratorNetwork(nn.Module):
    """TimeGAN Generator Network
    
    在潜在空间中生成序列表示
    """
    def __init__(self, noise_dim, hidden_dim, num_layers=2):
        super(GeneratorNetwork, self).__init__()
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 生成器RNN
        self.rnn = nn.GRU(
            input_size=noise_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # 输出投影层
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Sigmoid()
        
    def forward(self, z):
        """
        Args:
            z: [batch_size, seq_len, noise_dim] - 噪声序列
        Returns:
            h_synthetic: [batch_size, seq_len, hidden_dim] - 生成的潜在表示
        """
        h, _ = self.rnn(z)
        h = self.output_projection(h)
        h = self.activation(h)
        return h

class DiscriminatorNetwork(nn.Module):
    """TimeGAN Discriminator Network
    
    区分真实和合成的潜在表示序列
    """
    def __init__(self, hidden_dim, num_layers=2):
        super(DiscriminatorNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 双向RNN（论文推荐）
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2因为双向
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, h):
        """
        Args:
            h: [batch_size, seq_len, hidden_dim] - 潜在表示
        Returns:
            y: [batch_size, seq_len, 1] - 每个时间步的真实性概率
        """
        # 双向RNN
        rnn_out, _ = self.rnn(h)  # [batch_size, seq_len, hidden_dim*2]
        
        # 分类
        y = self.classifier(rnn_out)  # [batch_size, seq_len, 1]
        
        return y

class TimeGANODFlowPredictor(nn.Module):
    """基于TimeGAN架构的OD流量预测模型
    
    主要创新点：
    1. 四网络架构：embedding, recovery, generator, discriminator
    2. 联合训练：重建损失 + 监督损失 + 对抗损失
    3. 潜在空间学习：在低维空间进行对抗学习
    4. 适配预测任务：将生成任务转换为预测任务
    """
    def __init__(self, input_dim=6, hidden_dim=64, time_steps=28, noise_dim=32, num_layers=2):
        super(TimeGANODFlowPredictor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_steps = time_steps
        self.noise_dim = noise_dim
        self.num_layers = num_layers
        
        # TimeGAN四大网络组件
        self.embedding = EmbeddingNetwork(input_dim, hidden_dim, num_layers)
        self.recovery = RecoveryNetwork(hidden_dim, input_dim)
        self.generator = GeneratorNetwork(noise_dim, hidden_dim, num_layers)
        self.discriminator = DiscriminatorNetwork(hidden_dim, num_layers)
        
        # OD流量预测头（新增组件）
        self.od_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 2),  # 输出2维OD流量
            nn.Sigmoid()
        )
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
        # 损失权重（参考论文设定）
        self.lambda_recon = 1.0   # 重建损失权重
        self.eta_supervised = 10.0  # 监督损失权重（论文中的η）
        
    def forward(self, features, target_od=None, mode='train'):
        """
        前向传播
        Args:
            features: 输入特征 [batch_size, time_steps=28, input_dim=6]
            target_od: 目标OD流量 [batch_size, time_steps=28, 2]
            mode: 'train', 'eval', 'predict'
        Returns:
            结果字典
        """
        batch_size, seq_len = features.size(0), features.size(1)
        device = features.device
        
        if mode == 'train':
            return self._forward_train(features, target_od, device)
        elif mode == 'predict':
            return self._forward_predict(features)
        else:  # eval mode
            return self._forward_eval(features, target_od)
    
    def _forward_train(self, features, target_od, device):
        """训练模式前向传播 - TimeGAN联合训练"""
        batch_size, seq_len = features.size(0), features.size(1)
        
        # 1. Embedding: 真实特征 -> 潜在表示
        h_real = self.embedding(features)  # [B, T, H]
        
        # 2. Recovery: 潜在表示 -> 重建特征 (重建损失)
        x_reconstructed = self.recovery(h_real)
        recon_loss = self.mse_loss(x_reconstructed, features)
        
        # 3. Generator: 噪声 -> 合成潜在表示
        z = torch.randn(batch_size, seq_len, self.noise_dim, device=device)
        h_synthetic = self.generator(z)
        
        # 4. OD流量预测 (基于真实潜在表示)
        od_predicted = self.od_predictor(h_real)  # [B, T, 2]
        prediction_loss = self.mse_loss(od_predicted, target_od)
        
        # 5. 监督损失：让generator学习条件分布
        # 简化的监督损失计算，避免inplace操作问题
        z_supervised = torch.randn(batch_size, seq_len, self.noise_dim, device=device)
        h_supervised = self.generator(z_supervised)
        
        # 监督损失：合成的潜在表示应该接近真实的
        supervised_loss = self.mse_loss(h_supervised, h_real.detach())
        
        # 6. 对抗损失
        # 判别真实的潜在表示
        d_real = self.discriminator(h_real)
        d_real_loss = self.bce_loss(d_real, torch.ones_like(d_real))
        
        # 判别合成的潜在表示  
        d_synthetic = self.discriminator(h_synthetic.detach())
        d_synthetic_loss = self.bce_loss(d_synthetic, torch.zeros_like(d_synthetic))
        
        # 判别器总损失
        discriminator_loss = d_real_loss + d_synthetic_loss
        
        # 生成器对抗损失（让判别器认为合成的是真实的）
        d_synthetic_for_g = self.discriminator(h_synthetic)
        generator_adv_loss = self.bce_loss(d_synthetic_for_g, torch.ones_like(d_synthetic_for_g))
        
        # 7. 总损失组合（参考论文公式）
        # Embedding + Recovery: λ * L_S + L_R
        embedding_recovery_loss = self.lambda_recon * recon_loss + supervised_loss
        
        # Generator: η * L_S + L_U (对抗损失)
        generator_loss = self.eta_supervised * supervised_loss + generator_adv_loss
        
        # 主要的预测损失
        total_loss = prediction_loss + 0.1 * embedding_recovery_loss + 0.05 * generator_loss
        
        return {
            'od_flows': od_predicted,
            'total_loss': total_loss,
            'prediction_loss': prediction_loss,
            'recon_loss': recon_loss,
            'supervised_loss': supervised_loss,
            'discriminator_loss': discriminator_loss,
            'generator_loss': generator_loss,
            'embedding_recovery_loss': embedding_recovery_loss
        }
    
    def _forward_eval(self, features, target_od=None):
        """评估模式前向传播"""
        # 只进行预测，不计算训练相关损失
        h_real = self.embedding(features)
        od_predicted = self.od_predictor(h_real)
        
        result = {'od_flows': od_predicted}
        
        if target_od is not None:
            prediction_loss = self.mse_loss(od_predicted, target_od)
            result['prediction_loss'] = prediction_loss
        
        return result
    
    def _forward_predict(self, features):
        """纯预测模式"""
        with torch.no_grad():
            h_real = self.embedding(features)
            od_predicted = self.od_predictor(h_real)
            return {'od_flows': od_predicted}
    
    def generate(self, features):
        """生成OD流量预测 - 保持与原代码接口一致"""
        with torch.no_grad():
            result = self._forward_predict(features)
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

# ========== TimeGAN训练函数 ==========
def train_timegan_model(args):
    """训练TimeGAN OD流量预测模型"""
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
    
    # 创建TimeGAN模型
    model = TimeGANODFlowPredictor(
        input_dim=6,
        hidden_dim=args.hidden_dim,
        time_steps=28,
        noise_dim=args.noise_dim,
        num_layers=args.num_layers
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TimeGAN模型创建成功！")
    print(f"  总参数数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  隐藏维度: {args.hidden_dim}")
    print(f"  噪声维度: {args.noise_dim}")
    print(f"  网络层数: {args.num_layers}")
    
    # 优化器设置 - TimeGAN使用不同的优化器分别训练不同组件
    # Embedding + Recovery优化器
    embedding_recovery_params = list(model.embedding.parameters()) + list(model.recovery.parameters()) + list(model.od_predictor.parameters())
    optimizer_emb_rec = torch.optim.Adam(
        embedding_recovery_params,
        lr=args.lr,
        betas=(0.5, 0.9),  # TimeGAN论文推荐参数
        weight_decay=args.weight_decay
    )
    
    # Generator + Discriminator优化器
    generator_params = list(model.generator.parameters())
    discriminator_params = list(model.discriminator.parameters())
    
    optimizer_g = torch.optim.Adam(
        generator_params,
        lr=args.lr,
        betas=(0.5, 0.9),
        weight_decay=args.weight_decay
    )
    
    optimizer_d = torch.optim.Adam(
        discriminator_params,
        lr=args.lr,
        betas=(0.5, 0.9),
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler_emb_rec = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_emb_rec, mode='min', factor=0.7, patience=args.patience, verbose=True, min_lr=1e-6
    )
    
    # 训练循环变量
    best_val_loss = float('inf')
    best_model_path = os.path.join(args.output_dir, 'best_timegan_od_model.pth')
    epochs_without_improvement = 0
    train_history = []
    
    print(f"\n开始训练TimeGAN OD流量预测模型...")
    print(f"模型将保存到: {best_model_path}")
    print(f"早停策略: 验证损失{args.early_stop_patience}轮无改善时停止训练")
    print("="*80)
    
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_losses = []
        train_prediction_losses = []
        train_recon_losses = []
        train_supervised_losses = []
        train_discriminator_losses = []
        train_generator_losses = []
        
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{args.epochs} [训练]")
        for features, od_flows in train_progress:
            features = features.to(device)
            od_flows = od_flows.to(device)
            
            # TimeGAN的三阶段训练策略 - 分别计算避免计算图冲突
            
            # ===== 阶段1: 训练Embedding + Recovery + OD Predictor =====
            optimizer_emb_rec.zero_grad()
            outputs1 = model(features, od_flows, mode='train')
            emb_rec_loss = outputs1['prediction_loss'] + 0.1 * outputs1['embedding_recovery_loss']
            emb_rec_loss.backward()
            torch.nn.utils.clip_grad_norm_(embedding_recovery_params, max_norm=1.0)
            optimizer_emb_rec.step()
            
            # ===== 阶段2: 训练Discriminator =====
            optimizer_d.zero_grad()
            outputs2 = model(features, od_flows, mode='train')
            discriminator_loss = outputs2['discriminator_loss']
            discriminator_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator_params, max_norm=1.0)
            optimizer_d.step()
            
            # ===== 阶段3: 训练Generator =====
            optimizer_g.zero_grad()
            outputs3 = model(features, od_flows, mode='train')
            generator_loss = outputs3['generator_loss']
            generator_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator_params, max_norm=1.0)
            optimizer_g.step()
            
            # 使用最后一次前向传播的输出用于记录
            outputs = outputs3
            
            # 记录损失
            total_loss = outputs['total_loss']
            train_losses.append(total_loss.item())
            train_prediction_losses.append(outputs['prediction_loss'].item())
            train_recon_losses.append(outputs['recon_loss'].item())
            train_supervised_losses.append(outputs['supervised_loss'].item())
            train_discriminator_losses.append(discriminator_loss.item())
            train_generator_losses.append(generator_loss.item())
            
            # 更新进度条
            train_progress.set_postfix({
                'Total': f'{total_loss.item():.4f}',
                'Pred': f'{outputs["prediction_loss"].item():.4f}',
                'Disc': f'{discriminator_loss.item():.4f}',
                'Gen': f'{generator_loss.item():.4f}'
            })
        
        # 计算训练指标
        avg_train_loss = np.mean(train_losses)
        avg_train_prediction = np.mean(train_prediction_losses)
        avg_train_recon = np.mean(train_recon_losses)
        avg_train_supervised = np.mean(train_supervised_losses)
        avg_train_discriminator = np.mean(train_discriminator_losses)
        avg_train_generator = np.mean(train_generator_losses)
        
        # 验证阶段 - 计算详细指标
        print(f"  🔍 计算验证集指标...")
        val_metrics = calculate_metrics(model, val_loader, device, desc="验证集评估")
        
        # 学习率调整
        scheduler_emb_rec.step(val_metrics['loss'])
        current_lr = optimizer_emb_rec.param_groups[0]['lr']
        
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
        print(f"   总损失: {avg_train_loss:.6f} | 预测: {avg_train_prediction:.6f} | 重建: {avg_train_recon:.6f}")
        print(f"   监督: {avg_train_supervised:.6f} | 判别: {avg_train_discriminator:.6f} | 生成: {avg_train_generator:.6f}")
        
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
        
        # 保存训练历史
        epoch_history = {
            'epoch': int(epoch + 1),
            'train_loss': float(avg_train_loss),
            'train_prediction_loss': float(avg_train_prediction),
            'train_recon_loss': float(avg_train_recon),
            'train_supervised_loss': float(avg_train_supervised),
            'train_discriminator_loss': float(avg_train_discriminator),
            'train_generator_loss': float(avg_train_generator),
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
                    f.write("TimeGAN OD流量预测模型训练日志\n")
                    f.write("=" * 50 + "\n")
                
                f.write(f"Epoch {epoch+1}/{args.epochs}\n")
                f.write(f"   Training - Total: {avg_train_loss:.6f}, Pred: {avg_train_prediction:.6f}, Recon: {avg_train_recon:.6f}\n")
                f.write(f"            - Supervised: {avg_train_supervised:.6f}, Disc: {avg_train_discriminator:.6f}, Gen: {avg_train_generator:.6f}\n")
                f.write(f"   Validation - Loss: {val_metrics['loss']:.6f}, RMSE: {val_metrics['rmse']:.6f}, MAE: {val_metrics['mae']:.6f}, PCC: {val_metrics['pcc']:.6f}\n")
                
                if test_metrics:
                    f.write(f"   Test - Loss: {test_metrics.get('loss', 0):.6f}, RMSE: {test_metrics.get('rmse', 0):.6f}, MAE: {test_metrics.get('mae', 0):.6f}, PCC: {test_metrics.get('pcc', 0):.6f}\n")
                
                if is_best:
                    f.write(f"   New best model saved (Val Loss: {best_val_loss:.6f})\n")
                else:
                    f.write(f"   No improvement ({epochs_without_improvement}/{args.early_stop_patience} epochs)\n")
                
                f.write(f"   Learning Rate: {current_lr:.2e}\n")
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
        if is_best:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_emb_rec_state_dict': optimizer_emb_rec.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'scheduler_state_dict': scheduler_emb_rec.state_dict(),
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
    print("🎉 TimeGAN OD流量预测模型 - 最终测试结果")
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
    results_file = os.path.join(args.output_dir, "timegan_od_results.txt")
    with open(results_file, "w", encoding='utf-8') as f:
        f.write("基于TimeGAN的OD流量预测模型测试结果\n")
        f.write("="*50 + "\n")
        f.write("论文: Time-series Generative Adversarial Networks (NeurIPS 2019)\n")
        f.write("作者: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar\n")
        f.write("模型架构核心特点:\n")
        f.write("  - 四网络架构: Embedding, Recovery, Generator, Discriminator\n")
        f.write("  - 联合训练策略: 重建损失 + 监督损失 + 对抗损失\n")
        f.write("  - 潜在空间学习: 在低维空间进行对抗学习\n")
        f.write("  - 步进监督: 显式学习时间步进的条件分布\n")
        f.write("  - 混合目标: 结合无监督对抗学习和监督序列建模\n")
        f.write("\n")
        f.write(f"模型参数:\n")
        f.write(f"  - 总参数数量: {total_params:,}\n")
        f.write(f"  - 可训练参数: {trainable_params:,}\n")
        f.write(f"  - 隐藏维度: {args.hidden_dim}\n")
        f.write(f"  - 噪声维度: {args.noise_dim}\n")
        f.write(f"  - 网络层数: {args.num_layers}\n")
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
        f.write(f"\n")
        f.write("TimeGAN架构细节:\n")
        f.write("  1. Embedding Network: 将原始时序特征映射到低维潜在空间\n")
        f.write("  2. Recovery Network: 将潜在表示映射回原始特征空间\n")
        f.write("  3. Generator Network: 在潜在空间中生成合成序列\n")
        f.write("  4. Discriminator Network: 区分真实和合成的潜在表示\n")
        f.write("\n")
        f.write("训练策略:\n")
        f.write("  - 重建损失 (L_R): 确保embedding和recovery的可逆性\n")
        f.write("  - 监督损失 (L_S): 让生成器学习条件分布p(Xt|X_{1:t-1})\n") 
        f.write("  - 对抗损失 (L_U): 传统GAN对抗训练\n")
        f.write("  - 联合优化: min_{e,r} (λL_S + L_R), min_g (ηL_S + max_d L_U)\n")
    
    print(f"\n📁 详细结果已保存到: {results_file}")
    print(f"📁 最佳模型已保存到: {best_model_path}")
    print(f"📁 训练日志保存到: {log_file}")
    print(f"📁 训练历史保存到: {history_file}")
    
    return best_model_path

# ========== 主函数 ==========
def main():
    parser = argparse.ArgumentParser(description="基于TimeGAN的OD流量预测模型")
    
    # 数据参数 - 更新为52节点数据结构路径
    parser.add_argument("--io_flow_path", type=str, default="/private/od/data_NYTaxi/io_flow_daily.npy", 
                       help="IO流量数据路径")
    parser.add_argument("--graph_path", type=str, default="/private/od/data_NYTaxi/graph.npy", 
                       help="图结构数据路径")
    parser.add_argument("--od_matrix_path", type=str, default="/private/od/data_NYTaxi/od_matrix_daily.npy", 
                       help="OD矩阵数据路径")
    
    # TimeGAN模型参数
    parser.add_argument("--hidden_dim", type=int, default=64, 
                       help="隐藏维度 (潜在空间维度)")
    parser.add_argument("--noise_dim", type=int, default=32, 
                       help="噪声维度 (生成器输入维度)")
    parser.add_argument("--num_layers", type=int, default=2, 
                       help="RNN网络层数")
    
    # 训练参数  
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.0005, help="学习率 (TimeGAN推荐)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="权重衰减")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="测试集比例 (固定8:1:1划分)")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例 (固定8:1:1划分)")
    parser.add_argument("--seed", type=int, default=82, help="随机种子")
    
    # 早停和学习率调整参数
    parser.add_argument("--early_stop_patience", type=int, default=20, help="早停策略：验证损失多少轮无改善时停止训练")
    parser.add_argument("--patience", type=int, default=10, help="学习率调整策略：验证损失多少轮无改善时降低学习率")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="/private/od/paper_ny/TimeGAN", help="输出目录")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = create_dynamic_output_dir(args.output_dir)
    args.output_dir = output_dir
    
    print("="*60)
    print("🚀 TimeGAN OD流量预测模型")
    print("="*60)
    print("📖 论文: Time-series Generative Adversarial Networks")
    print("📖 会议: NeurIPS 2019")
    print("📖 作者: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar")
    print()
    print("🔧 模型创新点:")
    print("  ✅ 四网络架构 - Embedding, Recovery, Generator, Discriminator")
    print("  ✅ 联合训练策略 - 重建损失 + 监督损失 + 对抗损失")
    print("  ✅ 潜在空间学习 - 在低维空间进行对抗学习")
    print("  ✅ 步进监督 - 显式学习时间步进的条件分布")
    print("  ✅ 混合目标 - 结合无监督对抗学习和监督序列建模")
    print()
    print(f"📁 输出目录: {output_dir}")
    print("="*60)
    
    # 训练模型
    try:
        best_model_path = train_timegan_model(args)
        print("\n🎉 TimeGAN模型训练完成!")
        print(f"📁 最佳模型保存位置: {best_model_path}")
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {str(e)}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)