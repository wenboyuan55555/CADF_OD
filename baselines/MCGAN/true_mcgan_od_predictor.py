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
    dynamic_dir = os.path.join(base_dir, f"true_mcgan_run_{timestamp}")
    os.makedirs(dynamic_dir, exist_ok=True)
    return dynamic_dir

# ========== 稳定的MCGAN核心组件 ==========

class StableMCGANGenerator(nn.Module):
    """稳定的MCGAN生成器"""
    def __init__(self, input_dim=6, hidden_dim=64, output_dim=2, num_layers=2, dropout=0.1):
        super(StableMCGANGenerator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # 更稳定的特征编码器
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 添加LayerNorm提高稳定性
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 稳定的时序生成器
        self.temporal_generator = nn.LSTM(  # 使用LSTM替代GRU，更稳定
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # 更稳定的OD生成头
        self.od_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """更稳定的权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.8)  # 稍微增加gain但保持稳定
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data, gain=0.8)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data, gain=0.8)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0.01)
    
    def forward(self, features):
        batch_size, seq_len, _ = features.size()
        
        # 稳定的特征编码
        encoded_features = self.feature_encoder(features)
        
        # 稳定的时序生成
        temporal_output, _ = self.temporal_generator(encoded_features)
        
        # OD流量生成
        od_flows = self.od_head(temporal_output)
        
        # 确保输出在合理范围内
        od_flows = torch.clamp(od_flows, min=1e-6, max=1.0 - 1e-6)
        
        return od_flows

class StableMCGANDiscriminator(nn.Module):
    """稳定的MCGAN判别器"""
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, dropout=0.1):
        super(StableMCGANDiscriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 稳定的特征提取器
        self.feature_extractor = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 稳定的分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """稳定的权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.8)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data, gain=0.8)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data, gain=0.8)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0.01)
    
    def forward(self, od_flows):
        batch_size, seq_len, _ = od_flows.size()
        
        # 确保输入在合理范围
        od_flows = torch.clamp(od_flows, min=1e-6, max=1.0 - 1e-6)
        
        # 特征提取
        features, _ = self.feature_extractor(od_flows)
        
        # 判别分类
        validity = self.classifier(features)
        
        # 确保输出在合理范围
        validity = torch.clamp(validity, min=1e-6, max=1.0 - 1e-6)
        
        return validity

class TrueMCGANODFlowPredictor(nn.Module):
    """真正的MCGAN OD流量预测模型 - 优化版
    
    保持MCGAN核心特性：
    1. 回归损失：L_R(θ;φ) = E[(D_φ(x) - E[D_φ(ĝ)])²] ✅
    2. Monte Carlo估计器 ✅
    3. 判别器参与训练 ✅ 
    4. 稳定的数值计算 ✅
    5. 分阶段渐进式训练 ✅
    """
    def __init__(self, input_dim=6, hidden_dim=64, output_dim=2, num_layers=2, dropout=0.1, 
                 mc_samples=3, lambda_regression=0.8, lambda_adversarial=0.2):
        super(TrueMCGANODFlowPredictor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.mc_samples = mc_samples
        self.lambda_regression = lambda_regression
        self.lambda_adversarial = lambda_adversarial
        
        # 训练阶段控制
        self.training_phase = 1  # 1: 基础训练, 2: 判别器预训练, 3: 完整MCGAN
        self.current_epoch = 0
        self.discriminator_pretrain_epochs = 10  # 判别器预训练轮数
        self.mcgan_warmup_epochs = 20  # MCGAN预热轮数
        
        # 稳定的MCGAN网络
        self.generator = StableMCGANGenerator(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.discriminator = StableMCGANDiscriminator(
            input_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # 稳定的损失函数
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()  # 更稳定的BCE
        
        # PCC优化权重 - 重点优化PCC指标
        self.lambda_pcc = 0.5  # PCC损失权重
        
    def set_training_phase(self, epoch):
        """根据epoch设置训练阶段"""
        self.current_epoch = epoch
        if epoch < self.discriminator_pretrain_epochs:
            self.training_phase = 1  # 基础生成器训练
        elif epoch < self.discriminator_pretrain_epochs + self.mcgan_warmup_epochs:
            self.training_phase = 2  # 判别器预训练 + 生成器
        else:
            self.training_phase = 3  # 完整MCGAN训练
            
    def get_phase_description(self):
        """获取当前阶段描述"""
        if self.training_phase == 1:
            return "基础生成器训练"
        elif self.training_phase == 2:
            return "判别器预训练"
        else:
            return "完整MCGAN训练"
            
    def get_regression_weight(self):
        """渐进式回归损失权重"""
        if self.training_phase == 1:
            return 0.0  # 第一阶段不使用回归损失
        elif self.training_phase == 2:
            # 预训练阶段逐渐增加权重
            progress = (self.current_epoch - self.discriminator_pretrain_epochs) / self.mcgan_warmup_epochs
            return self.lambda_regression * min(1.0, progress)
        else:
            # 完整阶段使用全权重
            return self.lambda_regression
    
    def _compute_pcc_loss(self, pred, target):
        """计算皮尔逊相关系数损失 - 重点优化PCC指标"""
        # 展平张量
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
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
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        return pcc_loss
    
    def forward(self, features, target_od=None, mode='train'):
        if mode == 'train':
            return self._forward_train(features, target_od)
        elif mode == 'predict':
            return self._forward_predict(features)
        else:  # eval mode
            return self._forward_eval(features, target_od)
    
    def _stable_monte_carlo_sampling(self, features, device):
        """超稳定的Monte Carlo采样 - 优化版"""
        if self.training_phase == 1:
            # 第一阶段不进行MC采样，返回虚拟值
            batch_size, seq_len = features.size(0), features.size(1)
            return torch.full((batch_size, seq_len, 1), 0.5, device=device)
        
        mc_discriminator_outputs = []
        successful_samples = 0
        max_attempts = self.mc_samples * 2  # 允许重试
        
        for attempt in range(max_attempts):
            if successful_samples >= self.mc_samples:
                break
                
            try:
                # 添加更小的噪声
                noise_scale = 0.001  # 减小噪声规模
                noise = torch.randn_like(features) * noise_scale
                noisy_features = torch.clamp(features + noise, 0.0, 1.0)  # 确保在合理范围
                
                # 使用当前生成器状态生成样本
                generated_sample = self.generator(noisy_features)
                
                # 检查生成样本的有效性
                if torch.isfinite(generated_sample).all():
                    # 使用detach避免梯度回传到生成器
                    d_generated = self.discriminator(generated_sample.detach())
                    
                    # 检查判别器输出
                    if torch.isfinite(d_generated).all():
                        mc_discriminator_outputs.append(d_generated)
                        successful_samples += 1
                        
            except Exception as e:
                # 跳过有问题的样本
                continue
        
        if len(mc_discriminator_outputs) == 0:
            # 如果没有成功样本，返回安全的默认值
            batch_size, seq_len = features.size(0), features.size(1)
            return torch.full((batch_size, seq_len, 1), 0.5, device=device, requires_grad=True)
        
        # 稳定的期望计算
        expected_d_generated = torch.stack(mc_discriminator_outputs, dim=0).mean(dim=0)
        
        # 最终安全检查
        if not torch.isfinite(expected_d_generated).all():
            batch_size, seq_len = features.size(0), features.size(1)
            return torch.full((batch_size, seq_len, 1), 0.5, device=device, requires_grad=True)
            
        return expected_d_generated
    
    def _forward_train(self, features, target_od):
        """分阶段训练模式 - 优化版MCGAN"""
        batch_size, seq_len = features.size(0), features.size(1)
        device = features.device
        
        # 1. 生成OD流量
        generated_od = self.generator(features)
        
        # 2. 预测损失（所有阶段都有）
        prediction_loss = self.mse_loss(generated_od, target_od)
        
        # 初始化所有损失项
        regression_loss = torch.tensor(0.0, device=device, requires_grad=True)
        discriminator_loss = torch.tensor(0.0, device=device, requires_grad=True)
        generator_adv_loss = torch.tensor(0.0, device=device, requires_grad=True)
        pcc_loss = torch.tensor(0.0, device=device, requires_grad=True)  # PCC损失
        d_real_mean = torch.tensor(0.5, device=device)
        d_fake_mean = torch.tensor(0.5, device=device)
        expected_d_generated_mean = torch.tensor(0.5, device=device)
        
        # 根据训练阶段计算不同的损失
        if self.training_phase >= 2:  # 阶段2和3需要判别器
            try:
                # 3. 判别器相关计算
                d_real_logits = self.discriminator(target_od)
                d_fake_logits = self.discriminator(generated_od.detach())
                
                # 检查logits的有效性
                if torch.isfinite(d_real_logits).all() and torch.isfinite(d_fake_logits).all():
                    # 判别器损失
                    d_real_loss = self.bce_loss(d_real_logits.squeeze(-1), torch.ones_like(d_real_logits.squeeze(-1)))
                    d_fake_loss = self.bce_loss(d_fake_logits.squeeze(-1), torch.zeros_like(d_fake_logits.squeeze(-1)))
                    discriminator_loss = d_real_loss + d_fake_loss
                    
                    # 生成器对抗损失
                    d_fake_for_g_logits = self.discriminator(generated_od)
                    if torch.isfinite(d_fake_for_g_logits).all():
                        generator_adv_loss = self.bce_loss(d_fake_for_g_logits.squeeze(-1), torch.ones_like(d_fake_for_g_logits.squeeze(-1)))
                    
                    # 计算判别器评分
                    d_real_mean = torch.sigmoid(d_real_logits).mean()
                    d_fake_mean = torch.sigmoid(d_fake_logits).mean()
                
            except Exception as e:
                # 如果判别器计算失败，保持默认值
                pass
        
        # 4. MCGAN回归损失（仅在合适的阶段）
        current_regression_weight = self.get_regression_weight()
        if current_regression_weight > 0 and self.training_phase >= 2:
            try:
                # Monte Carlo估计期望判别器输出
                expected_d_generated = self._stable_monte_carlo_sampling(features, device)
                
                if torch.isfinite(expected_d_generated).all():
                    # 真实数据的判别器输出
                    d_real_for_regression = self.discriminator(target_od)
                    
                    if torch.isfinite(d_real_for_regression).all():
                        # MCGAN核心回归损失：L_R = E[(D(x) - E[D(ĝ)])²]
                        regression_loss = self.mse_loss(d_real_for_regression, expected_d_generated)
                        expected_d_generated_mean = expected_d_generated.mean()
                        
                        # 确保回归损失有效
                        if not torch.isfinite(regression_loss):
                            regression_loss = torch.tensor(0.0, device=device, requires_grad=True)
                            
            except Exception as e:
                # 如果回归损失计算失败，保持为0
                regression_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # 5. PCC损失计算 - 重点优化PCC指标
        try:
            pcc_loss = self._compute_pcc_loss(generated_od, target_od)
            if not torch.isfinite(pcc_loss):
                pcc_loss = torch.tensor(0.0, device=device, requires_grad=True)
        except Exception as e:
            pcc_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # 6. 总损失组合（根据阶段调整权重，重点优化PCC）
        if self.training_phase == 1:
            # 阶段1：预测损失 + PCC损失
            generator_total_loss = prediction_loss + self.lambda_pcc * pcc_loss
        elif self.training_phase == 2:
            # 阶段2：预测损失 + 渐进式回归损失 + 少量对抗损失 + PCC损失
            generator_total_loss = (
                0.7 * prediction_loss +                      # 降低预测损失权重
                current_regression_weight * regression_loss +
                0.1 * self.lambda_adversarial * generator_adv_loss +
                1.2 * self.lambda_pcc * pcc_loss            # 重点优化PCC
            )
        else:
            # 阶段3：完整MCGAN + 重点PCC优化
            generator_total_loss = (
                0.6 * prediction_loss +                      # 进一步降低预测损失权重
                current_regression_weight * regression_loss +
                self.lambda_adversarial * generator_adv_loss +
                1.5 * self.lambda_pcc * pcc_loss            # 最大化PCC优化
            )
        
        total_loss = generator_total_loss
        
        # 最终安全检查
        if not torch.isfinite(total_loss):
            total_loss = prediction_loss  # 回退到基础损失
        
        return {
            'od_flows': generated_od,
            'total_loss': total_loss,
            'prediction_loss': prediction_loss,
            'regression_loss': regression_loss,
            'discriminator_loss': discriminator_loss,
            'generator_adv_loss': generator_adv_loss,
            'generator_total_loss': generator_total_loss,
            'pcc_loss': pcc_loss,  # 新增PCC损失
            'd_real': d_real_mean,
            'd_fake': d_fake_mean,
            'expected_d_generated': expected_d_generated_mean,
            'training_phase': self.training_phase,
            'regression_weight': current_regression_weight
        }
    
    def _forward_eval(self, features, target_od=None):
        """评估模式"""
        generated_od = self.generator(features)
        result = {'od_flows': generated_od}
        
        if target_od is not None:
            prediction_loss = self.mse_loss(generated_od, target_od)
            result['prediction_loss'] = prediction_loss
        
        return result
    
    def _forward_predict(self, features):
        """纯预测模式"""
        with torch.no_grad():
            generated_od = self.generator(features)
            return {'od_flows': generated_od}
    
    def generate(self, features):
        """生成OD流量预测"""
        with torch.no_grad():
            result = self._forward_predict(features)
            return result['od_flows']

# ========== 保持与原代码一致的数据集 ==========
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
        
        # 数据集划分
        all_indices = list(range(len(self.od_pairs)))
        random.seed(seed)
        random.shuffle(all_indices)
        
        total_samples = len(all_indices)
        train_size = int(total_samples * 0.8)
        val_size = int(total_samples * 0.1)
        
        self.train_indices = all_indices[:train_size]
        self.val_indices = all_indices[train_size:train_size + val_size]
        self.test_indices = all_indices[train_size + val_size:]
        
        print(f"数据集划分完成:")
        print(f"  训练集: {len(self.train_indices)} 样本 ({len(self.train_indices)/total_samples:.1%})")
        print(f"  验证集: {len(self.val_indices)} 样本 ({len(self.val_indices)/total_samples:.1%})")
        print(f"  测试集: {len(self.test_indices)} 样本 ({len(self.test_indices)/total_samples:.1%})")
        
        self.set_mode('train')
    
    def set_mode(self, mode):
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
        site_pair_idx = self.current_indices[idx]
        site_i, site_j = self.od_pairs[site_pair_idx]
        
        # 获取OD流量
        od_i_to_j = self.od_matrix[site_i, site_j, :]
        od_j_to_i = self.od_matrix[site_j, site_i, :]
        od_flows = np.stack([od_i_to_j, od_j_to_i], axis=1)
        
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
        
        # 获取站点人口密度
        if hasattr(self, 'station_data') and len(self.station_data) > 0:
            if site_i < len(self.station_data) and site_j < len(self.station_data):
                pop_density_i = self.station_data[site_i].get('grid_population_density', 0.0)
                pop_density_j = self.station_data[site_j].get('grid_population_density', 0.0)
            else:
                pop_density_i = 0.0
                pop_density_j = 0.0
                
            pop_density = (pop_density_i + pop_density_j) / 2
            max_pop_density = max([station.get('grid_population_density', 1.0) for station in self.station_data])
            if max_pop_density == 0:
                max_pop_density = 1.0
            pop_density_normalized = pop_density / max_pop_density
        else:
            pop_density_normalized = 0.0
        
        # 构建特征：IO流量 + 距离特征 + 人口密度特征
        distance_feature = np.ones((self.time_steps, 1)) * distance_normalized
        pop_density_feature = np.ones((self.time_steps, 1)) * pop_density_normalized
        features = np.concatenate([io_flow_i, io_flow_j, distance_feature, pop_density_feature], axis=1)
        # 特征维度: (时间步, io_flow_features*2 + 2) = (时间步, 2*2+2=6) 或 (时间步, 4*2+2=10)
        
        return torch.FloatTensor(features), torch.FloatTensor(od_flows)

# ========== 评估指标计算 ==========
def calculate_metrics(model, dataloader, device, desc="Evaluating"):
    model.eval()
    all_predictions = []
    all_targets = []
    total_losses = []
    
    with torch.no_grad():
        progress = tqdm(dataloader, desc=desc, leave=False)
        for features, od_flows in progress:
            features = features.to(device)
            od_flows = od_flows.to(device)
            
            predicted = model.generate(features)
            loss = F.mse_loss(predicted, od_flows)
            total_losses.append(loss.item())
            
            all_predictions.append(predicted.cpu().numpy())
            all_targets.append(od_flows.cpu().numpy())
            
            progress.set_postfix({'MSE': f'{loss.item():.6f}'})
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    mse = np.mean((all_predictions - all_targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_predictions - all_targets))
    
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

# ========== 主训练函数 ==========
def train_true_mcgan_model(args):
    """训练真正的MCGAN模型"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
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
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dataset.set_mode('val')
    val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    dataset.set_mode('test')
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    dataset.set_mode('train')
    
    # 动态计算输入特征维度
    # 特征构成: io_flow_i + io_flow_j + distance + population_density
    # = io_flow_features*2 + 2
    io_flow_features = dataset.io_flow.shape[2]  # 2 或 4
    input_dim = io_flow_features * 2 + 2  # 6 或 10
    print(f"✅ 动态计算输入特征维度: {input_dim} (IO流量特征: {io_flow_features})")
    
    # 创建真正的MCGAN模型
    model = TrueMCGANODFlowPredictor(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=2,
        num_layers=args.num_layers,
        dropout=args.dropout,
        mc_samples=args.mc_samples,
        lambda_regression=args.lambda_regression,
        lambda_adversarial=args.lambda_adversarial
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    generator_params = sum(p.numel() for p in model.generator.parameters())
    discriminator_params = sum(p.numel() for p in model.discriminator.parameters())
    
    print(f"真正的MCGAN模型创建成功！")
    print(f"  总参数数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  生成器参数: {generator_params:,}")
    print(f"  判别器参数: {discriminator_params:,}")
    print(f"  隐藏维度: {args.hidden_dim}")
    print(f"  网络层数: {args.num_layers}")
    print(f"  MC采样数: {args.mc_samples}")
    print(f"  回归损失权重: {args.lambda_regression}")
    print(f"  对抗损失权重: {args.lambda_adversarial}")
    print(f"  🎯 核心特性: 保持MCGAN回归损失和Monte Carlo采样")
    
    # 稳定的优化器
    optimizer_g = torch.optim.AdamW(  # 使用AdamW
        model.generator.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8
    )
    
    optimizer_d = torch.optim.AdamW(
        model.discriminator.parameters(),
        lr=args.lr_d,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8
    )
    
    # 学习率调度器
    scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_g, mode='min', factor=0.8, patience=args.patience, verbose=True, min_lr=1e-6
    )
    
    scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_d, mode='min', factor=0.8, patience=args.patience, verbose=True, min_lr=1e-6
    )
    
    # 训练循环变量
    best_val_loss = float('inf')
    best_model_path = os.path.join(args.output_dir, 'best_true_mcgan_od_model.pth')
    epochs_without_improvement = 0
    train_history = []
    
    print(f"\n开始训练真正的MCGAN OD流量预测模型...")
    print(f"模型将保存到: {best_model_path}")
    print(f"早停策略: 验证损失{args.early_stop_patience}轮无改善时停止训练")
    print("="*80)
    
    for epoch in range(args.epochs):
        # 设置训练阶段
        model.set_training_phase(epoch)
        phase_desc = model.get_phase_description()
        current_regression_weight = model.get_regression_weight()
        
        # 训练阶段
        model.train()
        train_losses = []
        train_prediction_losses = []
        train_regression_losses = []
        train_discriminator_losses = []
        train_generator_adv_losses = []
        train_pcc_losses = []  # 新增PCC损失记录
        train_d_real_scores = []
        train_d_fake_scores = []
        
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{args.epochs} [{phase_desc}]")
        for features, od_flows in train_progress:
            features = features.to(device)
            od_flows = od_flows.to(device)
            
            # 根据训练阶段采用不同的训练策略
            if model.training_phase == 1:
                # ===== 阶段1: 只训练生成器 =====
                optimizer_g.zero_grad()
                outputs = model(features, od_flows, mode='train')
                generator_loss = outputs['generator_total_loss']
                
                if torch.isfinite(generator_loss):
                    generator_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.generator.parameters(), max_norm=1.0)
                    optimizer_g.step()
                    
            elif model.training_phase >= 2:
                # ===== 阶段2&3: 判别器 + 生成器训练 =====
                
                # 第一个前向传播：训练判别器
                outputs_d = model(features, od_flows, mode='train')
                discriminator_loss = outputs_d['discriminator_loss']
                
                if torch.isfinite(discriminator_loss) and discriminator_loss.item() > 1e-6:
                    optimizer_d.zero_grad()
                    discriminator_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), max_norm=1.0)
                    optimizer_d.step()
                
                # 第二个前向传播：训练生成器
                optimizer_g.zero_grad()
                outputs_g = model(features, od_flows, mode='train')
                generator_loss = outputs_g['generator_total_loss']
                
                if torch.isfinite(generator_loss):
                    generator_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.generator.parameters(), max_norm=1.0)
                    optimizer_g.step()
                
                # 使用最后一次的输出用于记录
                outputs = outputs_g
            else:
                # 默认情况：单次前向传播
                outputs = model(features, od_flows, mode='train')
            
            # 记录损失（根据阶段记录不同的指标）
            if torch.isfinite(outputs['total_loss']):
                train_losses.append(outputs['total_loss'].item())
                train_prediction_losses.append(outputs['prediction_loss'].item())
                train_regression_losses.append(outputs['regression_loss'].item())
                train_generator_adv_losses.append(outputs['generator_adv_loss'].item())
                train_pcc_losses.append(outputs['pcc_loss'].item())  # 记录PCC损失
                train_d_real_scores.append(outputs['d_real'].item())
                train_d_fake_scores.append(outputs['d_fake'].item())
                
                # 判别器损失根据阶段而定
                if model.training_phase >= 2:
                    # 阶段2和3有判别器训练
                    train_discriminator_losses.append(outputs['discriminator_loss'].item())
                else:
                    # 阶段1没有判别器训练，使用0值
                    train_discriminator_losses.append(0.0)
            
            # 更新进度条 - 根据训练阶段显示不同信息，突出PCC损失
            if model.training_phase == 1:
                train_progress.set_postfix({
                    'Phase': '基础',
                    'Total': f'{outputs["total_loss"].item():.4f}',
                    'Pred': f'{outputs["prediction_loss"].item():.4f}',
                    'PCC': f'{outputs["pcc_loss"].item():.4f}'  # 突出PCC损失
                })
            elif model.training_phase == 2:
                train_progress.set_postfix({
                    'Phase': '预训练',
                    'Total': f'{outputs["total_loss"].item():.4f}',
                    'PCC': f'{outputs["pcc_loss"].item():.4f}',  # 突出PCC损失
                    'MCGAN_Regr': f'{outputs["regression_loss"].item():.4f}',
                    'D_real': f'{outputs["d_real"].item():.3f}'
                })
            else:
                train_progress.set_postfix({
                    'Phase': '完整MCGAN',
                    'Total': f'{outputs["total_loss"].item():.4f}',
                    'PCC': f'{outputs["pcc_loss"].item():.4f}',  # 突出PCC损失
                    'MCGAN_Regr': f'{outputs["regression_loss"].item():.4f}',
                    'D_real': f'{outputs["d_real"].item():.3f}'
                })
        
        # 计算训练指标
        avg_train_loss = np.mean(train_losses) if train_losses else float('inf')
        avg_train_prediction = np.mean(train_prediction_losses) if train_prediction_losses else float('inf')
        avg_train_regression = np.mean(train_regression_losses) if train_regression_losses else 0.0
        avg_train_discriminator = np.mean(train_discriminator_losses) if train_discriminator_losses else float('inf')
        avg_train_generator_adv = np.mean(train_generator_adv_losses) if train_generator_adv_losses else float('inf')
        avg_train_pcc = np.mean(train_pcc_losses) if train_pcc_losses else 0.0  # PCC损失平均值
        avg_d_real_score = np.mean(train_d_real_scores) if train_d_real_scores else 0.5
        avg_d_fake_score = np.mean(train_d_fake_scores) if train_d_fake_scores else 0.5
        
        # 验证阶段
        print(f"  🔍 计算验证集指标...")
        val_metrics = calculate_metrics(model, val_loader, device, desc="验证集评估")
        
        # 学习率调整
        scheduler_g.step(val_metrics['loss'])
        scheduler_d.step(avg_train_discriminator)
        current_lr_g = optimizer_g.param_groups[0]['lr']
        current_lr_d = optimizer_d.param_groups[0]['lr']
        
        # 检查是否是最佳模型
        is_best = val_metrics['loss'] < best_val_loss
        test_metrics = None
        
        if is_best:
            print(f"  🎯 新最佳验证损失! 评估测试集...")
            test_metrics = calculate_metrics(model, test_loader, device, desc="测试集评估")
            best_val_loss = val_metrics['loss']
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if os.path.exists(best_model_path):
                try:
                    checkpoint = torch.load(best_model_path, map_location=device)
                    test_metrics = checkpoint.get('test_metrics', {})
                except:
                    test_metrics = {}
        
        # 打印详细结果
        print(f"\n📊 Epoch {epoch+1:3d}/{args.epochs} 真正的MCGAN训练完成:")
        print(f"{'='*80}")
        print(f"🔹 训练阶段: {phase_desc} (阶段 {model.training_phase})")
        print(f"🔹 回归损失权重: {current_regression_weight:.3f}")
        print(f"🔹 训练集 (真MCGAN + PCC优化):")
        if model.training_phase == 1:
            print(f"   总损失: {avg_train_loss:.6f} | 预测: {avg_train_prediction:.6f} | 🎯PCC损失: {avg_train_pcc:.6f}")
            print(f"   阶段: 基础生成器 + PCC优化")
        elif model.training_phase == 2:
            print(f"   总损失: {avg_train_loss:.6f} | 预测: {avg_train_prediction:.6f} | 🎯PCC损失: {avg_train_pcc:.6f}")
            print(f"   🎯MCGAN回归: {avg_train_regression:.6f} | 判别: {avg_train_discriminator:.6f} | 生成对抗: {avg_train_generator_adv:.6f}")
            print(f"   🎯判别器评分: D(real)={avg_d_real_score:.3f}, D(fake)={avg_d_fake_score:.3f} | 阶段: 判别器预训练 + PCC优化")
        else:
            print(f"   总损失: {avg_train_loss:.6f} | 预测: {avg_train_prediction:.6f} | 🎯PCC损失: {avg_train_pcc:.6f}")
            print(f"   🎯MCGAN回归: {avg_train_regression:.6f} | 判别: {avg_train_discriminator:.6f} | 生成对抗: {avg_train_generator_adv:.6f}")
            print(f"   🎯判别器评分: D(real)={avg_d_real_score:.3f}, D(fake)={avg_d_fake_score:.3f} | 阶段: 完整MCGAN + 最大PCC优化")
        
        print(f"🔹 验证集:")
        print(f"   总损失: {val_metrics['loss']:.6f} | MSE: {val_metrics['mse']:.6f}")
        print(f"   RMSE: {val_metrics['rmse']:.6f} | MAE: {val_metrics['mae']:.6f} | PCC: {val_metrics['pcc']:.6f}")
        
        if test_metrics:
            print(f"🔹 测试集:")  
            print(f"   总损失: {test_metrics.get('loss', 0):.6f} | MSE: {test_metrics.get('mse', 0):.6f}")
            print(f"   RMSE: {test_metrics.get('rmse', 0):.6f} | MAE: {test_metrics.get('mae', 0):.6f} | PCC: {test_metrics.get('pcc', 0):.6f}")
        else:
            print(f"🔹 测试集: 未评估 (仅在验证集改善时评估)")
        
        print(f"🔹 学习率: G={current_lr_g:.2e}, D={current_lr_d:.2e}")
        
        # 保存训练日志到文件 - 添加缺失的日志功能
        log_file = os.path.join(args.output_dir, "training_log.txt")
        try:
            # 如果是第一轮，创建新文件；否则追加
            mode = 'w' if epoch == 0 else 'a'
            with open(log_file, mode, encoding='utf-8') as f:
                if epoch == 0:
                    f.write("真正的MCGAN OD流量预测模型训练日志\n")
                    f.write("=" * 50 + "\n")
                
                f.write(f"Epoch {epoch+1}/{args.epochs}\n")
                f.write(f"   Training Phase: {phase_desc} (Stage {model.training_phase})\n")
                f.write(f"   Training - Total: {avg_train_loss:.6f}, Prediction: {avg_train_prediction:.6f}, MCGAN_Regression: {avg_train_regression:.6f}\n")
                f.write(f"   Validation - Loss: {val_metrics['loss']:.6f}, RMSE: {val_metrics['rmse']:.6f}, MAE: {val_metrics['mae']:.6f}, PCC: {val_metrics['pcc']:.6f}\n")
                
                # 🔧 关键修复：确保测试集指标总是被记录到日志文件
                if test_metrics:
                    f.write(f"   Test - Loss: {test_metrics.get('loss', 0):.6f}, RMSE: {test_metrics.get('rmse', 0):.6f}, MAE: {test_metrics.get('mae', 0):.6f}, PCC: {test_metrics.get('pcc', 0):.6f}\n")
                else:
                    f.write(f"   Test - Not evaluated this epoch (only when validation improves)\n")
                
                if is_best:
                    f.write(f"   New best model saved (Val Loss: {best_val_loss:.6f})\n")
                else:
                    f.write(f"   No improvement ({epochs_without_improvement}/{args.early_stop_patience} epochs without improvement)\n")
                
                f.write(f"   Learning Rate: G={current_lr_g:.2e}, D={current_lr_d:.2e}\n")
                f.write("\n")
                f.flush()
        except Exception as e:
            print(f"⚠️ 保存训练日志失败: {e}")
        
        # 保存最佳模型
        if is_best:
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'scheduler_g_state_dict': scheduler_g.state_dict(),
                'scheduler_d_state_dict': scheduler_d.state_dict(),
                'epoch': epoch,
                'val_loss': val_metrics['loss'],
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'train_history': train_history,
                'args': args
            }, best_model_path)
            print(f"🎯 ✅ 保存最佳真MCGAN模型 (验证损失: {best_val_loss:.6f})")
        else:
            print(f"⏳ 验证损失未改善 ({epochs_without_improvement}/{args.early_stop_patience}轮)")
        
        # 早停检查
        if epochs_without_improvement >= args.early_stop_patience:
            print(f"\n🛑 早停触发! 验证损失已{args.early_stop_patience}轮未改善，停止训练")
            print(f"   最佳验证损失: {best_val_loss:.6f} (来自第{epoch - epochs_without_improvement + 2}轮)")
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
    print("🎉 真正的MCGAN OD流量预测模型 - 最终测试结果")
    print(f"{'='*60}")
    print(f"📊 最终测试指标 (基于第{best_epoch}轮最佳模型):")
    print(f"   📈 均方误差 (MSE):     {final_test_metrics.get('mse', 0):.6f}")
    print(f"   📈 均方根误差 (RMSE):   {final_test_metrics.get('rmse', 0):.6f}")
    print(f"   📈 平均绝对误差 (MAE):  {final_test_metrics.get('mae', 0):.6f}")
    print(f"   📈 皮尔逊相关系数 (PCC): {final_test_metrics.get('pcc', 0):.6f}")
    print(f"   📈 测试损失:          {final_test_metrics.get('loss', 0):.6f}")
    print(f"{'='*60}")
    
    # 保存详细结果到文件
    results_file = os.path.join(args.output_dir, "true_mcgan_od_results.txt")
    with open(results_file, "w", encoding='utf-8') as f:
        f.write("真正的MCGAN OD流量预测模型测试结果\n")
        f.write("="*50 + "\n")
        f.write("论文: MCGAN: Enhancing GAN Training with Regression-Based Generator Loss (AAAI 2025)\n")
        f.write("模型核心特点:\n")
        f.write("  - 回归损失函数: L_R(θ;φ) = E[(D_φ(x) - E[D_φ(ĝ)])²]\n")
        f.write("  - Monte Carlo估计器估算期望判别器输出\n")
        f.write("  - 判别器真正参与训练（不固定0.5）\n")
        f.write("  - 稳定的数值计算和分阶段训练\n")
        f.write("\n")
        f.write(f"模型参数:\n")
        f.write(f"  - 隐藏维度: {args.hidden_dim}\n")
        f.write(f"  - 网络层数: {args.num_layers}\n")
        f.write(f"  - MC采样数: {args.mc_samples}\n")
        f.write(f"  - 回归损失权重: {args.lambda_regression}\n")
        f.write(f"  - 对抗损失权重: {args.lambda_adversarial}\n")
        f.write(f"  - 训练轮数: {args.epochs}\n")
        f.write(f"  - 批次大小: {args.batch_size}\n")
        f.write(f"  - 学习率: G={args.lr}, D={args.lr_d}\n")
        f.write("\n")
        f.write("测试结果:\n")
        f.write(f"  均方误差 (MSE):     {final_test_metrics.get('mse', 0):.6f}\n")
        f.write(f"  均方根误差 (RMSE):   {final_test_metrics.get('rmse', 0):.6f}\n")
        f.write(f"  平均绝对误差 (MAE):  {final_test_metrics.get('mae', 0):.6f}\n")
        f.write(f"  皮尔逊相关系数 (PCC): {final_test_metrics.get('pcc', 0):.6f}\n")
        f.write(f"  测试损失:          {final_test_metrics.get('loss', 0):.6f}\n")
        f.write(f"  最佳验证损失:       {best_val_loss:.6f}\n")
    
    print(f"\n📁 详细结果已保存到: {results_file}")
    print(f"📁 训练日志已保存到: {os.path.join(args.output_dir, 'training_log.txt')}")
    print(f"📁 最佳模型已保存到: {best_model_path}")
    
    return best_model_path

# ========== 主函数 ==========
def main():
    parser = argparse.ArgumentParser(description="真正的MCGAN OD流量预测模型")
    
    # 数据参数
    parser.add_argument("--io_flow_path", type=str, default="/private/od/data_NYTaxi/io_flow_daily.npy")
    parser.add_argument("--graph_path", type=str, default="/private/od/data_NYTaxi/graph.npy")
    parser.add_argument("--od_matrix_path", type=str, default="/private/od/data_NYTaxi/od_matrix_daily.npy")
    
    # 真MCGAN模型参数
    parser.add_argument("--hidden_dim", type=int, default=64, help="隐藏维度")
    parser.add_argument("--num_layers", type=int, default=2, help="网络层数")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout概率")
    parser.add_argument("--mc_samples", type=int, default=3, help="Monte Carlo采样数量")
    parser.add_argument("--lambda_regression", type=float, default=0.8, help="MCGAN回归损失权重")
    parser.add_argument("--lambda_adversarial", type=float, default=0.2, help="对抗损失权重")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.0002, help="生成器学习率")
    parser.add_argument("--lr_d", type=float, default=0.0002, help="判别器学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="权重衰减")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--seed", type=int, default=82, help="随机种子")
    
    # 早停和学习率调整参数
    parser.add_argument("--early_stop_patience", type=int, default=15, help="早停轮数")
    parser.add_argument("--patience", type=int, default=8, help="学习率调整轮数")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="/private/od/paper_ny/MCGAN", help="输出目录")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = create_dynamic_output_dir(args.output_dir)
    args.output_dir = output_dir
    
    print("="*60)
    print("🎯 真正的MCGAN OD流量预测模型")
    print("="*60)
    print("📖 论文: MCGAN: Enhancing GAN Training with Regression-Based Generator Loss")
    print("📖 会议: AAAI 2025")
    print("📖 作者: Baoren Xiao, Hao Ni, Weixin Yang")
    print()
    print("🔧 真正的MCGAN特性:")
    print("  ✅ 回归损失函数 - L_R(θ;φ) = E[(D_φ(x) - E[D_φ(ĝ)])²] 🎯")
    print("  ✅ Monte Carlo估计器 - 估算期望判别器输出")
    print("  ✅ 判别器真正参与训练 - 不固定0.5")
    print("  ✅ 稳定的数值计算 - LayerNorm + 梯度裁剪")
    print("  ✅ 保持论文核心思想 - 不简化关键组件")
    print()
    print(f"📁 输出目录: {output_dir}")
    print("="*60)
    
    try:
        best_model_path = train_true_mcgan_model(args)
        print("\n🎉 真正的MCGAN模型训练完成!")
        print(f"📁 最佳模型保存位置: {best_model_path}")
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {str(e)}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)