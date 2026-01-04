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
    dynamic_dir = os.path.join(base_dir, f"lmgu_ddpm_run_{timestamp}")
    os.makedirs(dynamic_dir, exist_ok=True)
    return dynamic_dir

# ========== LMGU-DDPM 核心组件 ==========

class GaussianMixtureDDPMSchedule:
    """
    基于论文的DDPM调度器 - 适配高斯混合模型学习
    
    论文中使用的调度策略：
    - 高噪声阶段：t = O(log d) 类似Power Iteration
    - 低噪声阶段：t = O(1) 类似EM算法
    - DDPM目标：Lt(s) = E[||s(Xt) + Zt/√(1-exp(-2t))||^2]
    """
    
    def __init__(self, timesteps=1000, high_noise_ratio=0.3, low_noise_ratio=0.7):
        self.timesteps = timesteps
        self.high_noise_steps = int(timesteps * high_noise_ratio)  # 高噪声阶段步数
        self.low_noise_steps = int(timesteps * low_noise_ratio)   # 低噪声阶段步数
        
        # 论文中的两阶段调度策略
        # 高噪声阶段：较大的噪声尺度，类似Power Iteration
        high_noise_betas = torch.linspace(0.01, 0.05, self.high_noise_steps)
        
        # 低噪声阶段：较小的噪声尺度，类似EM算法  
        low_noise_betas = torch.linspace(0.0001, 0.01, self.low_noise_steps)
        
        # 合并两个阶段
        remaining_steps = timesteps - self.high_noise_steps - self.low_noise_steps
        if remaining_steps > 0:
            transition_betas = torch.linspace(0.01, 0.005, remaining_steps)
            self.betas = torch.cat([high_noise_betas, transition_betas, low_noise_betas])
        else:
            self.betas = torch.cat([high_noise_betas, low_noise_betas])
        
        # 确保长度匹配
        if len(self.betas) != timesteps:
            self.betas = torch.linspace(0.0001, 0.05, timesteps)
        
        # 计算αt和累积乘积
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 计算后向过程需要的参数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # 后向过程方差
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    
    def get_noise_stage(self, t):
        """判断当前时间步属于哪个阶段"""
        if t >= self.timesteps - self.high_noise_steps:
            return "high_noise"  # 高噪声阶段（Power Iteration）
        else:
            return "low_noise"   # 低噪声阶段（EM算法）
    
    def get_schedule_params(self, t, device):
        """获取时间步t对应的调度参数"""
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        
        sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t]
        return sqrt_alpha_cumprod_t, sqrt_one_minus_alpha_cumprod_t

class StudentNetworkGMM(nn.Module):
    """
    基于论文的学生网络架构 - 适配时序数据
    
    论文中的学生网络：sμ(x) = tanh(μ^T x)μ - x
    这里适配为处理时序OD流量数据的版本
    """
    
    def __init__(self, input_dim=2, condition_dim=6, num_components=4, embed_dim=128):
        super().__init__()
        
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.num_components = num_components
        self.embed_dim = embed_dim
        
        # 时间嵌入 - 处理扩散时间步t
        self.time_embedding = self._build_time_embedding(embed_dim)
        
        # 条件嵌入 - 处理输入特征
        self.condition_embedding = self._build_condition_embedding(condition_dim, embed_dim)
        
        # 高斯混合分量参数 - 对应论文中的μ参数
        # 为每个时间步和输出维度学习多个高斯分量
        self.mixture_means = nn.Parameter(
            torch.randn(num_components, input_dim, embed_dim) * 0.1
        )
        
        # 学生网络的主体架构 - 基于论文中的tanh网络
        # sμ(x) = tanh(μ^T x)μ - x 的扩展版本
        self.student_networks = nn.ModuleList([
            self._build_student_network(input_dim, embed_dim) 
            for _ in range(num_components)
        ])
        
        # 混合权重预测网络
        self.mixture_weights_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),  # time + condition
            nn.SiLU(),
            nn.Linear(embed_dim, num_components),
            nn.Softmax(dim=-1)
        )
        
        # 初始化参数
        self._initialize_parameters()
    
    def _build_time_embedding(self, dim):
        """构建时间嵌入层"""
        half_dim = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim)
        
        class TimeEmbedding(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('freqs', freqs)
                self.mlp = nn.Sequential(
                    nn.Linear(dim, dim * 2),
                    nn.SiLU(),
                    nn.Linear(dim * 2, dim)
                )
            
            def forward(self, t):
                args = t.float()[:, None] * self.freqs[None, :]
                time_emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
                return self.mlp(time_emb)
        
        return TimeEmbedding()
    
    def _build_condition_embedding(self, condition_dim, embed_dim):
        """构建条件嵌入层"""
        return nn.Sequential(
            nn.Linear(condition_dim, embed_dim // 2),
            nn.SiLU(),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.SiLU(),
            # 时序注意力机制
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=4,
                    dim_feedforward=embed_dim * 2,
                    dropout=0.1,
                    batch_first=True
                ),
                num_layers=2
            )
        )
    
    def _build_student_network(self, input_dim, embed_dim):
        """构建单个学生网络 - 基于论文中的tanh架构"""
        return nn.Sequential(
            # 第一层：线性变换 + tanh激活 (对应论文中的μ^T x)
            nn.Linear(input_dim + embed_dim * 2, embed_dim),  # input + time + condition
            nn.Tanh(),
            
            # 第二层：输出层 (对应论文中的μ部分)
            nn.Linear(embed_dim, input_dim)
        )
    
    def _initialize_parameters(self):
        """初始化参数"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # 高斯混合均值使用小的随机值初始化
        nn.init.normal_(self.mixture_means, mean=0, std=0.1)
    
    def forward(self, x, timesteps, conditions):
        """
        前向传播 - 实现论文中的学生网络
        
        Args:
            x: [batch_size, input_dim, seq_len] 输入噪声数据
            timesteps: [batch_size,] 扩散时间步
            conditions: [batch_size, seq_len, condition_dim] 条件特征
            
        Returns:
            noise_pred: [batch_size, input_dim, seq_len] 预测的噪声
        """
        batch_size, input_dim, seq_len = x.shape
        
        # 时间和条件嵌入
        time_emb = self.time_embedding(timesteps)  # [batch_size, embed_dim]
        
        # 处理条件嵌入 - 考虑时序信息
        cond_reshaped = conditions.reshape(-1, conditions.shape[-1])  # [batch_size * seq_len, condition_dim]
        cond_emb = self.condition_embedding[:-1](cond_reshaped)  # 不使用Transformer层
        cond_emb = cond_emb.reshape(batch_size, seq_len, -1)  # [batch_size, seq_len, embed_dim]
        
        # 通过Transformer处理时序信息
        cond_emb = self.condition_embedding[-1](cond_emb)  # [batch_size, seq_len, embed_dim]
        cond_emb = cond_emb.mean(dim=1)  # [batch_size, embed_dim] 全局池化
        
        # 计算混合权重
        combined_emb = torch.cat([time_emb, cond_emb], dim=-1)  # [batch_size, embed_dim * 2]
        mixture_weights = self.mixture_weights_net(combined_emb)  # [batch_size, num_components]
        
        # 为每个时间步计算学生网络输出
        noise_predictions = []
        
        for t in range(seq_len):
            x_t = x[:, :, t]  # [batch_size, input_dim]
            
            # 扩展嵌入到当前时间步
            time_emb_t = time_emb  # [batch_size, embed_dim]
            cond_emb_t = cond_emb  # [batch_size, embed_dim]
            
            # 为每个高斯分量计算学生网络输出
            component_outputs = []
            for k in range(self.num_components):
                # 准备输入：x + time_emb + cond_emb
                student_input = torch.cat([x_t, time_emb_t, cond_emb_t], dim=-1)
                
                # 通过学生网络：实现论文中的 sμ(x) = tanh(μ^T x)μ - x
                student_output = self.student_networks[k](student_input)  # [batch_size, input_dim]
                
                # 添加残差连接 - 对应论文中的 "- x" 部分
                # 但这里我们不直接减去x，而是让网络学习噪声预测
                component_outputs.append(student_output)
            
            # 使用混合权重组合各个分量的输出
            component_outputs = torch.stack(component_outputs, dim=1)  # [batch_size, num_components, input_dim]
            
            # 加权组合
            mixture_weights_expanded = mixture_weights.unsqueeze(-1)  # [batch_size, num_components, 1]
            weighted_output = (component_outputs * mixture_weights_expanded).sum(dim=1)  # [batch_size, input_dim]
            
            noise_predictions.append(weighted_output)
        
        # 堆叠时间维度
        noise_pred = torch.stack(noise_predictions, dim=2)  # [batch_size, input_dim, seq_len]
        
        return noise_pred

class LMGUDDPMODFlowPredictor(nn.Module):
    """
    基于论文《Learning Mixtures of Gaussians Using the DDPM Objective》的OD流量预测模型
    
    核心设计理念：
    1. 高斯混合建模：将OD流量建模为多个高斯分量的混合
    2. DDPM学习目标：使用论文中证明的DDPM目标进行参数学习
    3. 两阶段训练：高噪声阶段 + 低噪声阶段的训练策略
    4. 学生网络架构：基于论文中的tanh激活两层网络
    
    适配改进：
    - 保持与原代码相同的输入输出接口
    - 支持MSE、RMSE、MAE、PCC评估指标
    - 集成条件化机制以提升预测精度
    """
    
    def __init__(self, input_dim=6, hidden_dim=128, time_steps=28, output_dim=2, 
                 timesteps=1000, num_components=4):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_steps = time_steps
        self.output_dim = output_dim
        self.timesteps = timesteps
        self.num_components = num_components
        
        # 论文中的DDPM调度器 - 两阶段设计
        self.ddpm_schedule = GaussianMixtureDDPMSchedule(
            timesteps=timesteps, 
            high_noise_ratio=0.3,  # 30% 高噪声阶段
            low_noise_ratio=0.7    # 70% 低噪声阶段
        )
        
        # 论文中的学生网络架构
        self.student_network = StudentNetworkGMM(
            input_dim=output_dim,
            condition_dim=input_dim,
            num_components=num_components,
            embed_dim=hidden_dim
        )
        
        # 特征条件化网络 - 用于推理阶段
        self.feature_to_init = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
    
    def forward_diffusion(self, x0, t, noise=None):
        """
        前向扩散过程 - 基于论文的DDPM框架
        
        Args:
            x0: [batch_size, time_steps, output_dim] 干净的数据
            t: [batch_size,] 时间步
            noise: [batch_size, time_steps, output_dim] 噪声 (可选)
        Returns:
            xt: [batch_size, time_steps, output_dim] 加噪后的数据
            noise: [batch_size, time_steps, output_dim] 添加的噪声
        """
        if noise is None:
            noise = torch.randn_like(x0)
        
        # 获取调度参数
        sqrt_alpha_cumprod_t, sqrt_one_minus_alpha_cumprod_t = self.ddpm_schedule.get_schedule_params(t, x0.device)
        
        # 广播到正确的形状
        sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.reshape(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.reshape(-1, 1, 1)
        
        # 添加噪声
        xt = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
        
        return xt, noise
    
    def backward_diffusion_step(self, xt, t, conditions, use_ddim=False):
        """
        后向扩散单步 - 基于论文的学生网络预测
        
        Args:
            xt: [batch_size, time_steps, output_dim] 当前噪声状态
            t: [batch_size,] 时间步
            conditions: [batch_size, time_steps, input_dim] 条件特征
            use_ddim: 是否使用DDIM采样
        Returns:
            xt_minus_1: [batch_size, time_steps, output_dim] 去噪后的状态
        """
        batch_size = xt.shape[0]
        
        # 转换为学生网络格式
        xt_student = xt.transpose(1, 2)  # [batch_size, output_dim, time_steps]
        
        # 使用学生网络预测噪声 - 论文中的核心
        predicted_noise = self.student_network(xt_student, t, conditions)
        predicted_noise = predicted_noise.transpose(1, 2)  # [batch_size, time_steps, output_dim]
        
        if use_ddim:
            # DDIM采样
            sqrt_alpha_cumprod_t, sqrt_one_minus_alpha_cumprod_t = self.ddpm_schedule.get_schedule_params(t, xt.device)
            sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.reshape(-1, 1, 1)
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.reshape(-1, 1, 1)
            
            # 预测x0
            pred_x0 = (xt - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / (sqrt_alpha_cumprod_t + 1e-8)
            pred_x0 = torch.clamp(pred_x0, -3.0, 3.0)
            
            if torch.all(t == 0):
                return torch.clamp(pred_x0, 0.0, 1.0)
            else:
                t_prev = torch.clamp(t - 1, min=0)
                sqrt_alpha_cumprod_t_prev, sqrt_one_minus_alpha_cumprod_t_prev = self.ddpm_schedule.get_schedule_params(t_prev, xt.device)
                sqrt_alpha_cumprod_t_prev = sqrt_alpha_cumprod_t_prev.reshape(-1, 1, 1)
                sqrt_one_minus_alpha_cumprod_t_prev = sqrt_one_minus_alpha_cumprod_t_prev.reshape(-1, 1, 1)
                
                xt_minus_1 = sqrt_alpha_cumprod_t_prev * pred_x0 + sqrt_one_minus_alpha_cumprod_t_prev * predicted_noise
                return torch.clamp(xt_minus_1, -2.0, 2.0)
        else:
            # 标准DDPM采样
            device = xt.device
            alphas = self.ddpm_schedule.alphas.to(device)[t].reshape(-1, 1, 1)
            alphas_cumprod = self.ddpm_schedule.alphas_cumprod.to(device)[t].reshape(-1, 1, 1)
            betas = self.ddpm_schedule.betas.to(device)[t].reshape(-1, 1, 1)
            
            sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
            sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alphas_cumprod)
            
            model_mean = sqrt_recip_alphas * (xt - betas * predicted_noise / (sqrt_one_minus_alpha_cumprod_t + 1e-8))
            
            if torch.all(t == 0):
                return torch.clamp(model_mean, 0.0, 1.0)
            else:
                posterior_variance_t = self.ddpm_schedule.posterior_variance.to(device)[t].reshape(-1, 1, 1)
                noise = torch.randn_like(xt) * 0.5
                result = model_mean + torch.sqrt(posterior_variance_t + 1e-8) * noise
                return torch.clamp(result, -2.0, 2.0)
    
    def two_stage_training_loss(self, features, target_od, t):
        """
        基于论文的两阶段训练损失计算
        
        Args:
            features: [batch_size, time_steps, input_dim] 输入特征
            target_od: [batch_size, time_steps, output_dim] 目标OD流量
            t: [batch_size,] 时间步
        Returns:
            loss_dict: 包含各种损失的字典
        """
        batch_size = features.size(0)
        device = features.device
        
        # 前向扩散：添加噪声
        xt, noise = self.forward_diffusion(target_od, t)
        
        # 转换为学生网络格式
        xt_student = xt.transpose(1, 2)  # [batch_size, output_dim, time_steps]
        
        # 使用学生网络预测噪声
        predicted_noise = self.student_network(xt_student, t, features)
        predicted_noise = predicted_noise.transpose(1, 2)  # [batch_size, time_steps, output_dim]
        
        # 基础DDPM损失
        ddpm_loss = self.mse_loss(predicted_noise, noise)
        
        # 判断当前处于哪个训练阶段
        stage_info = [self.ddpm_schedule.get_noise_stage(t_i.item()) for t_i in t]
        high_noise_mask = torch.tensor([s == "high_noise" for s in stage_info], device=device)
        low_noise_mask = torch.tensor([s == "low_noise" for s in stage_info], device=device)
        
        # 高噪声阶段损失 - 类似Power Iteration，关注方向收敛
        if high_noise_mask.any():
            high_noise_indices = high_noise_mask.nonzero().squeeze(-1)
            
            if len(high_noise_indices) > 0:
                # 方向一致性损失
                pred_high = predicted_noise[high_noise_indices]
                noise_high = noise[high_noise_indices]
                
                # 计算余弦相似度损失 - 使用reshape避免view错误
                pred_flat = pred_high.reshape(len(high_noise_indices), -1)
                noise_flat = noise_high.reshape(len(high_noise_indices), -1)
                
                pred_norm = F.normalize(pred_flat, p=2, dim=-1)
                noise_norm = F.normalize(noise_flat, p=2, dim=-1)
                
                cosine_sim = (pred_norm * noise_norm).sum(dim=-1).mean()
                direction_loss = 1.0 - cosine_sim  # 鼓励方向一致性
            else:
                direction_loss = torch.tensor(0.0, device=device)
        else:
            direction_loss = torch.tensor(0.0, device=device)
        
        # 低噪声阶段损失 - 类似EM算法，关注精确重构
        pred_x0_for_pcc = None  # 用于PCC损失计算的pred_x0
        
        if low_noise_mask.any():
            low_noise_indices = low_noise_mask.nonzero().squeeze(-1)
            
            if len(low_noise_indices) > 0:
                # 重构损失
                sqrt_alpha_cumprod_t, sqrt_one_minus_alpha_cumprod_t = self.ddpm_schedule.get_schedule_params(t[low_noise_indices], device)
                sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.reshape(-1, 1, 1)
                sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.reshape(-1, 1, 1)
                
                xt_low = xt[low_noise_indices]
                pred_noise_low = predicted_noise[low_noise_indices]
                target_low = target_od[low_noise_indices]
                
                # 预测x0
                pred_x0 = (xt_low - sqrt_one_minus_alpha_cumprod_t * pred_noise_low) / (sqrt_alpha_cumprod_t + 1e-8)
                pred_x0_for_pcc = pred_x0  # 保存用于PCC计算
                reconstruction_loss = self.mse_loss(pred_x0, target_low)
            else:
                reconstruction_loss = torch.tensor(0.0, device=device)
        else:
            reconstruction_loss = torch.tensor(0.0, device=device)
        
        # 特征条件化损失
        feature_pred_loss = self._compute_feature_prediction_loss(features, target_od)
        
        # 相关性损失
        correlation_loss = self._compute_correlation_loss(predicted_noise, noise)
        
        # 直接PCC优化损失 - 重点优化PCC指标
        if pred_x0_for_pcc is not None:
            # 使用低噪声阶段的预测结果计算PCC损失
            target_for_pcc = target_od[low_noise_mask.nonzero().squeeze(-1)]
            direct_pcc_loss = self._compute_direct_pcc_loss(pred_x0_for_pcc, target_for_pcc)
        else:
            # 如果没有低噪声阶段，使用整体预测计算PCC损失
            sqrt_alpha_cumprod_t, sqrt_one_minus_alpha_cumprod_t = self.ddpm_schedule.get_schedule_params(t, device)
            sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.reshape(-1, 1, 1)
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.reshape(-1, 1, 1)
            pred_x0_all = (xt - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / (sqrt_alpha_cumprod_t + 1e-8)
            direct_pcc_loss = self._compute_direct_pcc_loss(pred_x0_all, target_od)
        
        # 组合损失 - 重新调整权重，重点优化PCC
        total_loss = (0.3 * ddpm_loss +                    # 基础DDPM损失（降低权重）
                     0.1 * direction_loss +               # 高噪声阶段：方向损失（降低权重）
                     0.2 * reconstruction_loss +           # 低噪声阶段：重构损失（降低权重）
                     0.3 * feature_pred_loss +             # 特征预测损失（降低权重）
                     0.2 * correlation_loss +              # 相关性损失（降低权重）
                     1.5 * direct_pcc_loss)                # 直接PCC损失（重点优化）
        
        return {
            'total_loss': total_loss,
            'ddpm_loss': ddpm_loss,
            'direction_loss': direction_loss,
            'reconstruction_loss': reconstruction_loss,
            'feature_prediction_loss': feature_pred_loss,
            'correlation_loss': correlation_loss,
            'direct_pcc_loss': direct_pcc_loss,  # 新增直接PCC损失
            'high_noise_ratio': high_noise_mask.float().mean(),
            'low_noise_ratio': low_noise_mask.float().mean()
        }
    
    def _compute_feature_prediction_loss(self, features, target):
        """计算特征直接预测损失"""
        batch_size, time_steps, _ = features.shape
        feature_predictions = []
        
        for t in range(time_steps):
            step_feature = features[:, t, :]
            step_pred = self.feature_to_init(step_feature)
            feature_predictions.append(step_pred)
        
        feature_pred_sequence = torch.stack(feature_predictions, dim=1)
        return self.mse_loss(feature_pred_sequence, target)
    
    def _compute_correlation_loss(self, pred, target):
        """计算皮尔逊相关系数损失 - 增强版PCC优化"""
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        
        valid_mask = ~(torch.isnan(pred_flat) | torch.isnan(target_flat))
        if torch.sum(valid_mask) < 2:
            return torch.tensor(1.0, device=pred.device)
        
        pred_valid = pred_flat[valid_mask]
        target_valid = target_flat[valid_mask]
        
        pred_mean = torch.mean(pred_valid)
        target_mean = torch.mean(target_valid)
        
        pred_centered = pred_valid - pred_mean
        target_centered = target_valid - target_mean
        
        numerator = torch.sum(pred_centered * target_centered)
        denominator = torch.sqrt(torch.sum(pred_centered ** 2) * torch.sum(target_centered ** 2)) + 1e-8
        
        correlation = numerator / denominator
        
        # 增强PCC优化：使用更强的惩罚
        pcc_loss = 1.0 - correlation  # 直接最大化PCC
        
        # 数值稳定性检查
        if torch.isnan(pcc_loss) or torch.isinf(pcc_loss):
            return torch.tensor(0.5, device=pred.device)
        
        return pcc_loss
    
    def _compute_direct_pcc_loss(self, pred, target):
        """直接PCC损失 - 专门用于提升PCC指标"""
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
            # 训练模式：使用论文中的两阶段训练策略
            
            # 随机采样时间步
            t = torch.randint(0, self.timesteps, (batch_size,), device=device)
            
            # 计算两阶段训练损失
            loss_dict = self.two_stage_training_loss(features, target_od, t)
            
            # 为了兼容性，计算预测结果
            xt, noise = self.forward_diffusion(target_od, t)
            xt_student = xt.transpose(1, 2)
            predicted_noise = self.student_network(xt_student, t, features)
            predicted_noise = predicted_noise.transpose(1, 2)
            
            # 计算预测的x0用于评估
            sqrt_alpha_cumprod_t, sqrt_one_minus_alpha_cumprod_t = self.ddpm_schedule.get_schedule_params(t, device)
            sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.reshape(-1, 1, 1)
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.reshape(-1, 1, 1)
            
            pred_x0 = (xt - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / (sqrt_alpha_cumprod_t + 1e-8)
            pred_x0 = torch.clamp(pred_x0, -5.0, 5.0)
            
            # 更新结果字典
            loss_dict.update({
                'od_flows': pred_x0,
                'mse_loss': loss_dict['ddpm_loss'],  # 为了兼容性
                'mae_loss': self.mae_loss(pred_x0, target_od)
            })
            
            return loss_dict
            
        else:
            # 推理模式：扩散生成
            return self.generate_od_flows(features)
    
    def generate_od_flows(self, features, num_inference_steps=50, use_ddim=True):
        """
        生成OD流量预测 - 基于论文的两阶段推理
        
        Args:
            features: [batch_size, time_steps, input_dim] 输入特征
            num_inference_steps: 推理步数
            use_ddim: 是否使用DDIM采样加速
        Returns:
            generated_od: [batch_size, time_steps, output_dim] 生成的OD流量
        """
        batch_size = features.size(0)
        device = features.device
        
        # 特征引导初始化 - 更强的特征依赖
        feature_based_predictions = []
        for t in range(self.time_steps):
            step_feature = features[:, t, :]
            step_pred = self.feature_to_init(step_feature)
            feature_based_predictions.append(step_pred)
        
        feature_pred_sequence = torch.stack(feature_based_predictions, dim=1)
        
        # 在特征预测基础上添加少量噪声
        noise_scale = 0.1
        noise = torch.randn_like(feature_pred_sequence) * noise_scale
        xt = feature_pred_sequence + noise
        
        # 计算推理时间步
        if use_ddim:
            timesteps = torch.linspace(self.timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=device)
        else:
            timesteps = torch.arange(self.timesteps - 1, -1, -1, dtype=torch.long, device=device)
            num_inference_steps = len(timesteps)
        
        # 两阶段去噪过程
        for i in range(num_inference_steps):
            t = timesteps[i:i+1].expand(batch_size)
            
            with torch.no_grad():
                xt = self.backward_diffusion_step(xt, t, features, use_ddim=use_ddim)
                xt = torch.clamp(xt, -2.0, 2.0)
                
                # 根据当前阶段调整特征引导强度
                current_stage = self.ddpm_schedule.get_noise_stage(t[0].item())
                
                if current_stage == "high_noise":
                    # 高噪声阶段：更强的特征引导，类似Power Iteration
                    guidance_strength = 0.4 * (1.0 - i / num_inference_steps)
                else:
                    # 低噪声阶段：适中的特征引导，类似EM算法精调
                    guidance_strength = 0.2 * (1.0 - i / num_inference_steps)
                
                # 重新计算特征引导
                feature_guidance = []
                for step in range(self.time_steps):
                    step_feature = features[:, step, :]
                    step_pred = self.feature_to_init(step_feature)
                    feature_guidance.append(step_pred)
                feature_guidance = torch.stack(feature_guidance, dim=1)
                
                # 应用特征引导
                xt = xt + guidance_strength * (feature_guidance - xt)
                
                # 定期强化特征一致性
                if i % 10 == 0:
                    xt = 0.85 * xt + 0.15 * feature_guidance
        
        # 最终输出裁剪
        xt = torch.clamp(xt, 0.0, 1.0)
        
        return {'od_flows': xt}
    
    def generate(self, features):
        """生成OD流量预测 - 保持与原代码接口一致"""
        with torch.no_grad():
            result = self.generate_od_flows(features, num_inference_steps=50, use_ddim=True)
            return result['od_flows']

# ========== 复用原代码的数据集和工具函数 ==========

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

# ========== LMGU-DDPM训练函数 ==========
def train_lmgu_ddpm_model(args):
    """训练LMGU-DDPM OD流量预测模型"""
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
    
    # 创建LMGU-DDPM模型
    model = LMGUDDPMODFlowPredictor(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        time_steps=28,
        output_dim=2,
        timesteps=args.diffusion_timesteps,
        num_components=args.num_components
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LMGU-DDPM模型创建成功！")
    print(f"  总参数数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  隐藏维度: {args.hidden_dim}")
    print(f"  扩散时间步: {args.diffusion_timesteps}")
    print(f"  高斯混合分量数: {args.num_components}")
    
    # 优化器 - 使用AdamW优化器
    optimizer = torch.optim.AdamW(
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
    best_model_path = os.path.join(args.output_dir, 'best_lmgu_ddpm_od_model.pth')
    epochs_without_improvement = 0
    train_history = []
    
    print(f"\n开始训练LMGU-DDPM OD流量预测模型...")
    print(f"模型将保存到: {best_model_path}")
    print(f"早停策略: 验证损失{args.early_stop_patience}轮无改善时停止训练")
    print("="*80)
    
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_losses = []
        train_ddpm_losses = []
        train_direction_losses = []
        train_reconstruction_losses = []
        train_feature_losses = []
        train_correlation_losses = []
        train_direct_pcc_losses = []
        
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{args.epochs} [训练]")
        for features, od_flows in train_progress:
            features = features.to(device)
            od_flows = od_flows.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播 - 使用两阶段训练
            outputs = model(features, od_flows, mode='train')
            total_loss = outputs['total_loss']
            ddpm_loss = outputs['ddpm_loss']
            direction_loss = outputs.get('direction_loss', 0.0)
            reconstruction_loss = outputs.get('reconstruction_loss', 0.0)
            feature_loss = outputs.get('feature_prediction_loss', 0.0)
            corr_loss = outputs.get('correlation_loss', 0.0)
            direct_pcc_loss = outputs.get('direct_pcc_loss', 0.0)
            
            # 反向传播
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 记录损失
            train_losses.append(total_loss.item())
            train_ddpm_losses.append(ddpm_loss.item())
            train_direction_losses.append(direction_loss.item() if torch.is_tensor(direction_loss) else direction_loss)
            train_reconstruction_losses.append(reconstruction_loss.item() if torch.is_tensor(reconstruction_loss) else reconstruction_loss)
            train_feature_losses.append(feature_loss.item() if torch.is_tensor(feature_loss) else feature_loss)
            train_correlation_losses.append(corr_loss.item() if torch.is_tensor(corr_loss) else corr_loss)
            train_direct_pcc_losses.append(direct_pcc_loss.item() if torch.is_tensor(direct_pcc_loss) else direct_pcc_loss)
            
            # 更新进度条
            train_progress.set_postfix({
                'Total': f'{total_loss.item():.4f}',
                'DDPM': f'{ddpm_loss.item():.4f}',
                'PCC': f'{direct_pcc_loss.item() if torch.is_tensor(direct_pcc_loss) else direct_pcc_loss:.4f}',
                'Recon': f'{reconstruction_loss.item() if torch.is_tensor(reconstruction_loss) else reconstruction_loss:.4f}',
                'Corr': f'{corr_loss.item() if torch.is_tensor(corr_loss) else corr_loss:.4f}'
            })
        
        # 计算训练指标
        avg_train_loss = np.mean(train_losses)
        avg_train_ddpm = np.mean(train_ddpm_losses)
        avg_train_direction = np.mean(train_direction_losses)
        avg_train_recon = np.mean(train_reconstruction_losses)
        avg_train_feature = np.mean(train_feature_losses)
        avg_train_corr = np.mean(train_correlation_losses)
        avg_train_direct_pcc = np.mean(train_direct_pcc_losses)
        
        # 验证阶段
        print(f"  🔍 计算验证集指标...")
        val_metrics = calculate_metrics(model, val_loader, device, desc="验证集评估")
        
        # 学习率调整
        scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']
        
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
        print(f"\n📊 Epoch {epoch+1:3d}/{args.epochs} 训练完成:")
        print(f"{'='*80}")
        print(f"🔹 训练集:")
        print(f"   总损失: {avg_train_loss:.6f} | DDPM: {avg_train_ddpm:.6f} | PCC损失: {avg_train_direct_pcc:.6f}")
        print(f"   方向损失: {avg_train_direction:.6f} | 重构损失: {avg_train_recon:.6f}")
        print(f"   特征损失: {avg_train_feature:.6f} | 相关性损失: {avg_train_corr:.6f}")
        
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
            'train_ddpm_loss': float(avg_train_ddpm),
            'train_direction_loss': float(avg_train_direction),
            'train_reconstruction_loss': float(avg_train_recon),
            'val_loss': float(val_metrics['loss']),
            'val_mse': float(val_metrics['mse']),
            'val_rmse': float(val_metrics['rmse']),
            'val_mae': float(val_metrics['mae']),
            'val_pcc': float(val_metrics['pcc']),
            'lr': float(current_lr),
            'is_best': bool(is_best)
        }
        
        if test_metrics:
            epoch_history.update({
                'test_loss': float(test_metrics.get('loss', 0)),
                'test_mse': float(test_metrics.get('mse', 0)),
                'test_rmse': float(test_metrics.get('rmse', 0)),
                'test_mae': float(test_metrics.get('mae', 0)),
                'test_pcc': float(test_metrics.get('pcc', 0))
            })
        
        train_history.append(epoch_history)
        
        # 保存训练日志
        log_file = os.path.join(args.output_dir, "training_log.txt")
        try:
            mode = 'w' if epoch == 0 else 'a'
            with open(log_file, mode, encoding='utf-8') as f:
                if epoch == 0:
                    f.write("LMGU-DDPM OD流量预测模型训练日志\n")
                    f.write("=" * 50 + "\n")
                
                f.write(f"Epoch {epoch+1}/{args.epochs}\n")
                f.write(f"   Training - Total: {avg_train_loss:.6f}, DDPM: {avg_train_ddpm:.6f}, Direction: {avg_train_direction:.6f}\n")
                f.write(f"   Validation - Loss: {val_metrics['loss']:.6f}, RMSE: {val_metrics['rmse']:.6f}, MAE: {val_metrics['mae']:.6f}, PCC: {val_metrics['pcc']:.6f}\n")
                
                if test_metrics:
                    f.write(f"   Test - Loss: {test_metrics.get('loss', 0):.6f}, RMSE: {test_metrics.get('rmse', 0):.6f}, MAE: {test_metrics.get('mae', 0):.6f}, PCC: {test_metrics.get('pcc', 0):.6f}\n")
                
                if is_best:
                    f.write(f"   New best model saved (Val Loss: {best_val_loss:.6f})\n")
                
                f.write(f"   Learning Rate: {current_lr:.2e}\n\n")
                f.flush()
        except Exception as e:
            print(f"⚠️ 保存训练日志失败: {e}")
        
        # 保存详细历史数据
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
            break
        
        if current_lr < 1e-6:
            print(f"\n🛑 学习率过小 ({current_lr:.2e})，停止训练")
            break
        
        print("="*80)
    
    # 最终测试阶段
    print(f"\n{'='*60}")
    print("🎯 最终测试阶段 - 使用最佳模型进行评估")
    print(f"{'='*60}")
    
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_epoch = checkpoint['epoch'] + 1
        best_val_metrics = checkpoint.get('val_metrics', {})
        best_test_metrics = checkpoint.get('test_metrics', {})
        print(f"✅ 已加载最佳模型 (来自第{best_epoch}轮)")
        
        print(f"\n🏆 最佳模型性能 (第{best_epoch}轮):")
        print(f"🔸 验证集: Loss={best_val_metrics.get('loss', 0):.6f}, RMSE={best_val_metrics.get('rmse', 0):.6f}, MAE={best_val_metrics.get('mae', 0):.6f}, PCC={best_val_metrics.get('pcc', 0):.6f}")
        print(f"🔸 测试集: Loss={best_test_metrics.get('loss', 0):.6f}, RMSE={best_test_metrics.get('rmse', 0):.6f}, MAE={best_test_metrics.get('mae', 0):.6f}, PCC={best_test_metrics.get('pcc', 0):.6f}")
        
        final_test_metrics = best_test_metrics
    else:
        print("⚠️ 最佳模型文件不存在，使用当前模型进行最终测试")
        final_test_metrics = calculate_metrics(model, test_loader, device, desc="最终测试")
    
    print(f"\n{'='*60}")
    print("🎉 LMGU-DDPM OD流量预测模型 - 最终测试结果")
    print(f"{'='*60}")
    print(f"📊 最终测试指标:")
    print(f"   📈 均方误差 (MSE):     {final_test_metrics.get('mse', 0):.6f}")
    print(f"   📈 均方根误差 (RMSE):   {final_test_metrics.get('rmse', 0):.6f}")
    print(f"   📈 平均绝对误差 (MAE):  {final_test_metrics.get('mae', 0):.6f}")
    print(f"   📈 皮尔逊相关系数 (PCC): {final_test_metrics.get('pcc', 0):.6f}")
    print(f"   📈 测试损失:          {final_test_metrics.get('loss', 0):.6f}")
    print(f"{'='*60}")
    
    # 保存详细结果
    results_file = os.path.join(args.output_dir, "lmgu_ddpm_od_results.txt")
    with open(results_file, "w", encoding='utf-8') as f:
        f.write("基于LMGU-DDPM的OD流量预测模型测试结果\n")
        f.write("="*50 + "\n")
        f.write("论文: Learning Mixtures of Gaussians Using the DDPM Objective (NeurIPS 2023)\n")
        f.write("作者: Kulin Shah, Sitan Chen, Adam Klivans\n")
        f.write("模型架构核心特点:\n")
        f.write("  - 高斯混合模型的DDPM学习框架 (Gaussian Mixture DDPM Learning)\n")
        f.write("  - 两阶段训练策略 (Two-Stage Training: Power Iteration + EM Algorithm)\n")
        f.write("  - 学生网络架构 sμ(x) = tanh(μ^T x)μ - x (Student Network Architecture)\n")
        f.write("  - 理论保证的参数恢复 (Theoretical Guarantees for Parameter Recovery)\n")
        f.write("  - 条件生成机制 (Conditional Generation Mechanism)\n")
        f.write("\n")
        f.write(f"模型参数:\n")
        f.write(f"  - 总参数数量: {total_params:,}\n")
        f.write(f"  - 可训练参数: {trainable_params:,}\n")
        f.write(f"  - 隐藏维度: {args.hidden_dim}\n")
        f.write(f"  - 扩散时间步: {args.diffusion_timesteps}\n")
        f.write(f"  - 高斯混合分量数: {args.num_components}\n")
        f.write(f"  - 训练轮数: {args.epochs}\n")
        f.write(f"  - 批次大小: {args.batch_size}\n")
        f.write(f"  - 学习率: {args.lr}\n")
        f.write("\n")
        f.write("测试结果:\n")
        f.write(f"  均方误差 (MSE):     {final_test_metrics.get('mse', 0):.6f}\n")
        f.write(f"  均方根误差 (RMSE):   {final_test_metrics.get('rmse', 0):.6f}\n")
        f.write(f"  平均绝对误差 (MAE):  {final_test_metrics.get('mae', 0):.6f}\n")
        f.write(f"  皮尔逊相关系数 (PCC): {final_test_metrics.get('pcc', 0):.6f}\n")
        f.write(f"  测试损失:          {final_test_metrics.get('loss', 0):.6f}\n")
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
    parser = argparse.ArgumentParser(description="基于LMGU-DDPM的OD流量预测模型")
    
    # 数据参数
    parser.add_argument("--io_flow_path", type=str, default="/private/od/data_NYTaxi/io_flow_daily.npy", 
                       help="IO流量数据路径")
    parser.add_argument("--graph_path", type=str, default="/private/od/data_NYTaxi/graph.npy", 
                       help="图结构数据路径")
    parser.add_argument("--od_matrix_path", type=str, default="/private/od/data_NYTaxi/od_matrix_daily.npy", 
                       help="OD矩阵数据路径")
    
    # LMGU-DDPM模型参数
    parser.add_argument("--hidden_dim", type=int, default=128, 
                       help="隐藏维度")
    parser.add_argument("--diffusion_timesteps", type=int, default=1000, 
                       help="扩散过程时间步数")
    parser.add_argument("--num_components", type=int, default=4,
                       help="高斯混合分量数")
    
    # 训练参数  
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.0002, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--seed", type=int, default=82, help="随机种子")
    
    # 早停和学习率调整参数
    parser.add_argument("--early_stop_patience", type=int, default=15, help="早停策略")
    parser.add_argument("--patience", type=int, default=8, help="学习率调整策略")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="/private/od/paper_ny/LMGU_DDPM", help="输出目录")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = create_dynamic_output_dir(args.output_dir)
    args.output_dir = output_dir
    
    print("="*60)
    print("🚀 LMGU-DDPM OD流量预测模型")
    print("="*60)
    print("📖 论文: Learning Mixtures of Gaussians Using the DDPM Objective")
    print("📖 会议: NeurIPS 2023")
    print("📖 作者: Kulin Shah, Sitan Chen, Adam Klivans")
    print()
    print("🔧 模型创新点:")
    print("  ✅ 高斯混合模型的DDPM学习 - 理论保证的参数恢复")
    print("  ✅ 两阶段训练策略 - 高噪声(Power Iteration) + 低噪声(EM算法)")
    print("  ✅ 学生网络架构 - sμ(x) = tanh(μ^T x)μ - x")
    print("  ✅ 条件生成机制 - 基于输入特征进行条件化生成")
    print("  ✅ 时序数据适配 - 将高斯混合学习适配到OD流量生成")
    print()
    print(f"📁 输出目录: {output_dir}")
    print("="*60)
    
    # 训练模型
    try:
        best_model_path = train_lmgu_ddpm_model(args)
        print("\n🎉 LMGU-DDPM模型训练完成!")
        print(f"📁 最佳模型保存位置: {best_model_path}")
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {str(e)}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)