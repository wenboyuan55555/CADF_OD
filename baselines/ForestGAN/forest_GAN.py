import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import re
from datetime import datetime
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from contextlib import nullcontext
import math
import argparse
import random
import traceback
import pandas as pd
import matplotlib.pyplot as plt
import time
import json
import warnings
warnings.filterwarnings("ignore")

# 尝试导入transformers库
try:
    from transformers import AutoTokenizer, AutoModel
    print("成功导入transformers库")
except ImportError:
    print("transformers库将在需要时安装")

# ========== Informer模型定义 ==========
class ProbAttention(nn.Module):
    """概率稀疏自注意力机制"""
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def _prob_QK(self, Q, K, sample_k, n_top):
        # Q [B, H, L, D]
        B, H, L_Q, D = Q.shape
        _, _, L_K, _ = K.shape
        
        # 计算稀疏样本注意力得分
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, D)  # B, H, L_Q, L_K, D
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # L_Q, sample_k
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]  # B, H, L_Q, sample_k, D
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)  # B, H, L_Q, sample_k
        
        # 找出最大的M个标记的索引
        M = int(math.ceil(L_K * self.factor))
        M = min(M, L_K)
        
        # 找出具有较高贡献度的键
        _, top_indices = torch.topk(Q_K_sample, n_top, dim=-1)  # B, H, L_Q, n_top
        
        # 使用前n_top个稀疏样本得到的索引
        index_mask = torch.ones(B, H, L_Q, L_K, device=Q.device)
        index_mask.scatter_(-1, top_indices, 0)  # B, H, L_Q, L_K
        
        return index_mask

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape
        
        queries = queries.transpose(1, 2)  # B, H, L_Q, D
        keys = keys.transpose(1, 2)  # B, H, L_K, D
        values = values.transpose(1, 2)  # B, H, L_K, D

        # 设置稀疏采样的样本数和保留的top数
        U_part = self.factor * math.ceil(math.log(L_K))
        u = self.factor * math.ceil(math.log(L_Q))
        
        U_part = int(min(U_part, L_K))
        u = int(min(u, L_Q))

        # 计算概率稀疏注意力
        index_mask = self._prob_QK(queries, keys, U_part, u)  # B, H, L_Q, L_K
        
        # 计算注意力得分，仅保留索引掩码中的值为True的部分
        scale = self.scale or 1. / math.sqrt(D)
        if self.mask_flag:
            if attn_mask is not None:
                index_mask = index_mask * attn_mask
        
        # 计算注意力
        attention = torch.matmul(queries, keys.transpose(2, 3)) * scale  # B, H, L_Q, L_K
        if self.mask_flag and attn_mask is not None:
            attention = attention.masked_fill(attn_mask == 0, -1e9)
        
        # 处理概率注意力
        attention = torch.where(index_mask > 0, attention, torch.tensor(-1e9, device=attention.device))
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        
        # 计算注意力输出
        out = torch.matmul(attention, values)  # B, H, L_Q, D
        out = out.transpose(1, 2)  # B, L_Q, H, D
        
        if self.output_attention:
            return out, attention
        else:
            return out, None

class AttentionLayer(nn.Module):
    """注意力层封装"""
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()
        
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        
    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        
        # 投影查询、键、值
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        
        # 处理掩码
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, H, 1, 1)  # B, H, L, S
        
        # 计算注意力
        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        
        # 输出投影
        out = out.contiguous().view(B, L, -1)
        out = self.out_projection(out)
        
        return out, attn

class EncoderLayer(nn.Module):
    """Informer编码器层"""
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        
    def forward(self, x, attn_mask=None):
        # 自注意力机制
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        
        # 残差连接和层归一化
        x = x + self.dropout(new_x)
        x = self.norm1(x)
        
        # 前馈网络
        y = x.transpose(1, 2)
        y = self.conv1(y)
        y = self.activation(y)
        y = self.conv2(y)
        y = y.transpose(1, 2)
        
        # 再次进行残差连接和层归一化
        x = x + self.dropout(y)
        x = self.norm2(x)
        
        return x, attn

class Encoder(nn.Module):
    """Informer编码器"""
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
        
    def forward(self, x, attn_mask=None):
        attns = []
        
        # 通过所有编码器层
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask)
            attns.append(attn)
        
        if self.norm is not None:
            x = self.norm(x)
            
        return x, attns

class Informer(nn.Module):
    """Informer编码器模型"""
    def __init__(self, enc_in=5, d_model=128, c_out=128, factor=5, n_heads=8, 
                 e_layers=3, d_ff=512, dropout=0.1, activation='gelu'):
        super(Informer, self).__init__()
        
        # 编码输入特征
        self.enc_embedding = nn.Linear(enc_in, d_model)
        
        # Informer编码器 - 使用ProbAttention
        attn_enc_layers = []
        for _ in range(e_layers):
            attn = ProbAttention(mask_flag=False, factor=factor, attention_dropout=dropout)
            attention_layer = AttentionLayer(
                attention=attn,
                d_model=d_model,
                n_heads=n_heads
            )
            layer = EncoderLayer(
                attention=attention_layer,
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation
            )
            attn_enc_layers.append(layer)
            
        # 编码器
        self.encoder = Encoder(
            attn_layers=attn_enc_layers,
            norm_layer=nn.LayerNorm(d_model)
        )
        
        # 输出层
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x):
        # x: [B, T, enc_in]
        
        # 编码输入
        enc_out = self.enc_embedding(x)  # [B, T, d_model]
        
        # 编码器
        enc_out, attns = self.encoder(enc_out)  # [B, T, d_model]
        
        # 输出投影
        out = self.projection(enc_out)  # [B, T, c_out]
        
        return out 

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
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
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
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dynamic_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(dynamic_dir, exist_ok=True)
    return dynamic_dir

# ========== 序列模式检测工具 ==========
class SequencePatternDetector:
    """时间序列模式检测工具，检测周期性、趋势性和平稳性，以及按周分析变化趋势"""
    @staticmethod
    def detect_pattern(sequence, threshold_periodic=0.7, threshold_trend=0.5, threshold_stationary=0.3):
        """
        检测序列模式并返回模式类型和关键位置
        Args:
            sequence: 时间序列数据，形状(T,)
            threshold_periodic: 周期性判断阈值
            threshold_trend: 趋势性判断阈值
            threshold_stationary: 平稳性判断阈值
        Returns:
            pattern_info: 包含模式类型、关键位置和详细特征的字典
        """
        # 如果序列中存在NaN或无穷大，使用插值替换
        if np.isnan(sequence).any() or np.isinf(sequence).any():
            # 记录原始非NaN/Inf索引和值
            valid_indices = ~(np.isnan(sequence) | np.isinf(sequence))
            valid_positions = np.where(valid_indices)[0]
            valid_values = sequence[valid_indices]
            
            # 如果有足够的有效值进行插值
            if len(valid_values) > 1:
                # 使用线性插值填充NaN和Inf
                x_valid = np.arange(len(sequence))[valid_indices]
                x_all = np.arange(len(sequence))
                # 如果序列开头或结尾有NaN，使用最近的有效值填充
                if not valid_indices[0]:
                    first_valid_idx = np.argmax(valid_indices)
                    sequence[:first_valid_idx] = sequence[first_valid_idx]
                if not valid_indices[-1]:
                    last_valid_idx = len(valid_indices) - 1 - np.argmax(valid_indices[::-1])
                    sequence[last_valid_idx+1:] = sequence[last_valid_idx]
                # 对中间的NaN值进行插值
                interp_values = np.interp(x_all, x_valid, valid_values)
                sequence = interp_values
            else:
                # 如果有效值不足，使用0填充
                sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)
            
        # 计算基本统计量
        mean_val = np.mean(sequence)
        std_val = np.std(sequence)
        min_val = np.min(sequence)
        max_val = np.max(sequence)
        median_val = np.median(sequence)
        
        # 防止除零错误
        if std_val == 0:
            std_val = 1e-8
            
        # 序列长度
        T = len(sequence)
        
        # 周期性检测 (自相关方法) - 扩展检查更长的周期
        max_corr = 0
        max_lag = 0
        # 检查更广范围的lag，包括7天周期
        max_check_lag = min(T // 2, 14)  # 至少检查到14天
        
        # 周期性相关系数的集合
        period_corrs = {}
        
        for lag in range(1, max_check_lag + 1):
            # 计算自相关系数
            corr_val = np.corrcoef(sequence[:-lag], sequence[lag:])[0, 1]
            if not np.isnan(corr_val):
                period_corrs[lag] = corr_val
                if corr_val > max_corr:
                    max_corr = corr_val
                    max_lag = lag
        
        periodicity_score = max_corr
        
        # 特别检查7天周期的相关性
        weekly_corr = period_corrs.get(7, 0) if 7 in period_corrs else 0
        
        # 趋势性检测
        x = np.arange(T)
        x_mean = np.mean(x)
        y_mean = mean_val
        
        numerator = np.sum((x - x_mean) * (sequence - y_mean))
        denominator = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((sequence - y_mean) ** 2))
        
        # 防止除零错误
        if denominator == 0:
            trend_score = 0
        else:
            trend_score = abs(numerator / denominator)
            
        # 计算趋势斜率
        if denominator != 0:
            slope = numerator / np.sum((x - x_mean) ** 2)
        else:
            slope = 0
        
        # 平稳性检测 - 使用变异系数
        cv = std_val / (abs(mean_val) + 1e-8)  # 变异系数
        stationary_score = 1 / (1 + cv)  # 变换为[0,1]范围，越高越平稳
        
        # 按周划分分析 - 假设序列长度为28天
        weekly_features = {}
        if T >= 28:
            # 将28天按7天一周划分为4周
            weeks = [sequence[i:i+7] for i in range(0, 28, 7)]
            
            # 分析每个星期几对应的数据趋势
            for day_idx in range(7):
                # 提取每周相同天的数据
                day_data = [week[day_idx] if day_idx < len(week) else np.nan for week in weeks]
                day_data = [x for x in day_data if not np.isnan(x)]
                
                if len(day_data) >= 2:
                    # 计算这一天的均值、方差、趋势
                    day_mean = np.mean(day_data)
                    day_std = np.std(day_data)
                    day_min = np.min(day_data)
                    day_max = np.max(day_data)
                    
                    # 计算趋势斜率
                    if len(day_data) > 2:
                        day_x = np.arange(len(day_data))
                        day_trend = np.polyfit(day_x, day_data, 1)[0]
                    else:
                        day_trend = day_data[-1] - day_data[0]
                    
                    # 保存特征
                    weekly_features[f"day_{day_idx+1}"] = {
                        "values": day_data,
                        "mean": float(day_mean),
                        "std": float(day_std),
                        "min": float(day_min),
                        "max": float(day_max),
                        "trend": float(day_trend),
                        "week_to_week_change": [float(day_data[i+1] - day_data[i]) for i in range(len(day_data)-1)]
                    }
        
        # 检测峰值和谷值
        # 使用平滑后的序列检测更可靠的峰值
        from scipy.signal import find_peaks
        
        # 简单移动平均平滑
        window_size = 3
        if T >= window_size:
            smoothed = np.convolve(sequence, np.ones(window_size)/window_size, mode='valid')
            # 在平滑序列上寻找峰值
            peaks, _ = find_peaks(smoothed, prominence=0.1*np.std(smoothed))
            valleys, _ = find_peaks(-smoothed, prominence=0.1*np.std(smoothed))
            
            # 将峰值位置映射回原始序列
            peaks = peaks + window_size//2
            valleys = valleys + window_size//2
            
            # 限制在有效范围内
            peaks = peaks[peaks < T]
            valleys = valleys[valleys < T]
        else:
            # 序列太短，直接使用原序列
            peaks, _ = find_peaks(sequence, prominence=0.1*np.std(sequence))
            valleys, _ = find_peaks(-sequence, prominence=0.1*np.std(sequence))
        
        # 确定主要类型，但保留所有分数
        # 这里不再简单划分为单一类型，而是计算各种特征分数
        pattern_types = []
        if periodicity_score > threshold_periodic:
            pattern_types.append("periodic")
        if trend_score > threshold_trend:
            pattern_types.append("trend")
        if stationary_score > threshold_stationary:
            pattern_types.append("stationary")
        if not pattern_types:
            pattern_types.append("mixed")
        
        # 找出主要模式类型
        main_type = pattern_types[0]
        
        # 计算振幅
        amplitude = max_val - min_val
        
        # 计算季节性指数 - 适用于7天周期的时间序列
        seasonality_index = weekly_corr
        
        # 计算波动指数 - 归一化的方差
        variability_index = std_val / (max_val - min_val + 1e-8)
        
        # 主要关键位置
        if peaks.size > 0:
            key_position = int(peaks[np.argmax(sequence[peaks])])
        else:
            key_position = int(np.argmax(sequence))
        
        # 构建详细的模式信息
        pattern_info = {
            "types": pattern_types,
            "main_type": main_type,
            "position": key_position,
            "peaks": [int(p) for p in peaks],
            "valleys": [int(v) for v in valleys],
            "stats": {
                "mean": float(mean_val),
                "std": float(std_val),
                "min": float(min_val),
                "max": float(max_val),
                "median": float(median_val),
                "amplitude": float(amplitude)
            },
            "scores": {
                "periodicity": float(periodicity_score),
                "trend": float(trend_score),
                "stationary": float(stationary_score),
                "seasonality": float(seasonality_index),
                "variability": float(variability_index)
            },
            "trend": {
                "slope": float(slope),
                "direction": "上升" if slope > 0 else "下降" if slope < 0 else "平稳"
            },
            "weekly_analysis": weekly_features,
            "period_lag": int(max_lag)
        }
        
        return pattern_info

    @staticmethod
    def generate_prompt(station_i, station_j, io_flow_i, io_flow_j, pattern_info_in_i, pattern_info_out_i, 
                       pattern_info_in_j, pattern_info_out_j, pop_density_i=None, pop_density_j=None):
        """
        生成用于大模型的提示文本，包含更丰富的时序特征描述
        Args:
            station_i: 站点i的ID
            station_j: 站点j的ID
            io_flow_i: 站点i的IO流量, 形状(T, 2)
            io_flow_j: 站点j的IO流量, 形状(T, 2)
            pattern_info_*: 四个序列的模式信息字典
            pop_density_i: 站点i的人口密度
            pop_density_j: 站点j的人口密度
        Returns:
            prompt: 格式化的提示文本
        """
        # 提取数据
        inflow_i = io_flow_i[:, 0]
        outflow_i = io_flow_i[:, 1]
        inflow_j = io_flow_j[:, 0]
        outflow_j = io_flow_j[:, 1]
        
        # 计算净流量
        net_flow_i = outflow_i - inflow_i
        net_flow_j = outflow_j - inflow_j
        
        # 格式化数据，只保留小数点后2位
        inflow_i_str = np.array2string(inflow_i, precision=2, separator=',')
        outflow_i_str = np.array2string(outflow_i, precision=2, separator=',')
        inflow_j_str = np.array2string(inflow_j, precision=2, separator=',')
        outflow_j_str = np.array2string(outflow_j, precision=2, separator=',')
        net_flow_i_str = np.array2string(net_flow_i, precision=2, separator=',')
        net_flow_j_str = np.array2string(net_flow_j, precision=2, separator=',')
        
        # 生成丰富的动态标记
        def _get_detailed_marker(pattern_info, flow_type):
            marker_parts = []
            
            # 添加主要类型标记
            main_type = pattern_info["main_type"]
            
            if main_type == "periodic":
                # 周期性信息
                period_lag = pattern_info["period_lag"]
                period_str = f"{period_lag}天" if period_lag > 0 else "不规则"
                seasonality = pattern_info["scores"]["seasonality"]
                peaks = pattern_info["peaks"]
                peak_str = ", ".join([f"{p+1}天" for p in peaks[:3]])
                if len(peaks) > 3:
                    peak_str += f"... 共{len(peaks)}个峰值"
                
                marker_parts.append(f"[周期性序列 周期:{period_str} 周期强度:{seasonality:.2f} 主要峰值日:{peak_str}]")
                
            elif main_type == "trend":
                # 趋势性信息
                slope = pattern_info["trend"]["slope"]
                direction = pattern_info["trend"]["direction"]
                trend_score = pattern_info["scores"]["trend"]
                
                marker_parts.append(f"[趋势性序列 方向:{direction} 斜率:{slope:.4f} 趋势强度:{trend_score:.2f}]")
                
            elif main_type == "stationary":
                # 平稳性信息
                mean = pattern_info["stats"]["mean"]
                std = pattern_info["stats"]["std"]
                cv = std / (abs(mean) + 1e-8)
                
                marker_parts.append(f"[平稳性序列 均值:{mean:.2f} 标准差:{std:.2f} 变异系数:{cv:.2f}]")
                
            else:
                # 混合型
                marker_parts.append("[混合型序列]")
                
            # 添加统计信息
            stats = pattern_info["stats"]
            marker_parts.append(f"[统计数据 均值:{stats['mean']:.2f} 中位数:{stats['median']:.2f} 振幅:{stats['amplitude']:.2f}]")
            
            # 添加按周分析信息
            if "weekly_analysis" in pattern_info and pattern_info["weekly_analysis"]:
                weekly_trends = []
                
                # 分析每周同一天的变化
                for day_idx, day_info in pattern_info["weekly_analysis"].items():
                    if "trend" in day_info:
                        day_num = int(day_idx.split("_")[1])
                        trend_val = day_info["trend"]
                        trend_dir = "上升" if trend_val > 0.05 else "下降" if trend_val < -0.05 else "平稳"
                        weekly_trends.append(f"周{day_num}:{trend_dir}({trend_val:.2f})")
                
                # 如果有周趋势数据，添加到标记中
                if weekly_trends:
                    marker_parts.append(f"[周间变化 {' '.join(weekly_trends)}]")
            
            # 最终合并所有标记
            return " ".join(marker_parts)
        
        # 人口密度信息
        pop_density_info = ""
        if pop_density_i is not None and pop_density_j is not None:
            # 计算人口密度差异和比例
            density_diff = abs(pop_density_i - pop_density_j)
            density_ratio = max(pop_density_i, pop_density_j) / (min(pop_density_i, pop_density_j) + 1e-8)
            density_avg = (pop_density_i + pop_density_j) / 2
            
            # 人口密度差异描述
            density_diff_desc = "显著" if density_ratio > 3 else "中等" if density_ratio > 1.5 else "相近"
            
            pop_density_info = f"""
站点{station_i}附近人口密度: {pop_density_i:.2f} 人/平方公里
站点{station_j}附近人口密度: {pop_density_j:.2f} 人/平方公里
站点对平均人口密度: {density_avg:.2f} 人/平方公里
人口密度差异: {density_diff:.2f} ({density_diff_desc})
"""
        
        # 生成提示文本
        prompt = f"""站点对({station_i},{station_j})流量模式分析:
时间范围: 6月1日至6月28日 (共28天，4个完整周期)

-------- 站点{station_i}流量特征 --------
流入序列: {inflow_i_str}
流出序列: {outflow_i_str}
净流量序列: {net_flow_i_str}

站点{station_i}流入模式特征: {_get_detailed_marker(pattern_info_in_i, "流入")}
站点{station_i}流出模式特征: {_get_detailed_marker(pattern_info_out_i, "流出")}

-------- 站点{station_j}流量特征 --------
流入序列: {inflow_j_str}
流出序列: {outflow_j_str}
净流量序列: {net_flow_j_str}

站点{station_j}流入模式特征: {_get_detailed_marker(pattern_info_in_j, "流入")}
站点{station_j}流出模式特征: {_get_detailed_marker(pattern_info_out_j, "流出")}{pop_density_info}

-------- 站点间流量关系 --------
站点{station_i}到站点{station_j}的流量代表从起点{station_i}到终点{station_j}的交通需求
站点{station_j}到站点{station_i}的流量代表从起点{station_j}到终点{station_i}的交通需求

基于上述站点流量模式特征、每周变化趋势和人口密度信息，请详细分析站点{station_i}到站点{station_j}以及站点{station_j}到站点{station_i}的28天OD流量变化特征与模式，特别关注工作日与周末的差异、每周同一天的趋势变化。
"""
        return prompt

# ========== Qwen2特征提取器 ==========
class QwenFeatureExtractor(nn.Module):
    """使用Qwen2模型提取特征的模块"""
    def __init__(self, api_key=None, feature_dim=768, device="cuda:1"):
        """
        初始化Qwen2特征提取器
        Args:
            api_key: 不再使用，保留参数兼容性
            feature_dim: 输出特征维度
            device: 设备
        """
        super().__init__()
        self.device = device
        self.feature_dim = feature_dim
        
        print("初始化Qwen2词嵌入模型...")
        # 尝试导入transformers库
        try:
            from transformers import AutoTokenizer, AutoModel
            print("成功导入transformers库")
        except ImportError:
            print("未找到transformers库，正在尝试安装...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
            from transformers import AutoTokenizer, AutoModel
            print("transformers库安装并导入成功")
            
        # 初始化模型
        self.model_path = "/private/od/Qwen2-7B-embed-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path).to(self.device)
        print(f"Qwen2词嵌入模型初始化成功: {self.model_path}")
        
        # 如果需要，初始化特征降维网络
        if feature_dim != 768:  # Qwen2的嵌入维度通常是768
            self.projection = nn.Sequential(
                nn.Linear(768, self.feature_dim),
                nn.LayerNorm(self.feature_dim)
            ).to(self.device)
        else:
            self.projection = None
    
    def extract_text_embedding(self, prompt, pattern_info):
        """
        从Qwen2模型获取文本嵌入
        Args:
            prompt: 提示文本
            pattern_info: 包含模式类型和位置的字典
        Returns:
            features: 提取的特征向量
        """
        # 对文本进行编码
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 使用模型获取嵌入
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # 使用最后一层隐藏状态的平均池化表示作为嵌入向量
        # [batch_size, seq_length, hidden_size] -> [batch_size, hidden_size]
        embedding_tensor = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        
        # 如果维度不匹配且存在投影网络，则使用投影网络调整
        if self.projection is not None:
            with torch.no_grad():
                if len(embedding_tensor.shape) == 1:
                    embedding_tensor = self.projection(embedding_tensor.unsqueeze(0)).squeeze(0)
                else:
                    embedding_tensor = self.projection(embedding_tensor)
            
        return embedding_tensor
    
    def forward(self, station_i, station_j, io_flow_i, io_flow_j, station_data=None):
        """
        前向传播，处理站点对数据并提取特征
        Args:
            station_i, station_j: 站点ID
            io_flow_i, io_flow_j: 站点IO流量数据, 形状(T, 2)
            station_data: 站点人口密度数据列表
        Returns:
            token_features: 提取的令牌特征
            pattern_info: 模式信息
        """
        # 分解IO流量数据
        inflow_i = io_flow_i[:, 0].cpu().numpy()
        outflow_i = io_flow_i[:, 1].cpu().numpy()
        inflow_j = io_flow_j[:, 0].cpu().numpy()
        outflow_j = io_flow_j[:, 1].cpu().numpy()
        
        # 检测各序列的模式
        pattern_info_in_i = SequencePatternDetector.detect_pattern(inflow_i)
        pattern_info_out_i = SequencePatternDetector.detect_pattern(outflow_i)
        pattern_info_in_j = SequencePatternDetector.detect_pattern(inflow_j)
        pattern_info_out_j = SequencePatternDetector.detect_pattern(outflow_j)
        
        # 获取站点人口密度信息
        pop_density_i = None
        pop_density_j = None
        if station_data is not None and len(station_data) > 0:
            # 确保站点索引不超过可用的站点数据
            if station_i < len(station_data) and station_j < len(station_data):
                pop_density_i = station_data[station_i].get('grid_population_density', 0.0)
                pop_density_j = station_data[station_j].get('grid_population_density', 0.0)
        
        # 生成提示文本
        prompt = SequencePatternDetector.generate_prompt(
            station_i, station_j, 
            io_flow_i.cpu().numpy(), io_flow_j.cpu().numpy(),
            pattern_info_in_i, pattern_info_out_i, 
            pattern_info_in_j, pattern_info_out_j,
            pop_density_i, pop_density_j
        )
        
        # 选择最主要模式作为主导模式
        # 根据各模式分数选择
        scores = [
            pattern_info_in_i["scores"]["periodicity"],
            pattern_info_out_i["scores"]["periodicity"],
            pattern_info_in_j["scores"]["periodicity"],
            pattern_info_out_j["scores"]["periodicity"]
        ]
        
        # 选择周期性分数最高的模式
        max_score_idx = np.argmax(scores)
        if max_score_idx == 0:
            dominant_pattern = pattern_info_in_i
        elif max_score_idx == 1:
            dominant_pattern = pattern_info_out_i
        elif max_score_idx == 2:
            dominant_pattern = pattern_info_in_j
        else:
            dominant_pattern = pattern_info_out_j
        
        # 提取特征向量
        token_features = self.extract_text_embedding(prompt, dominant_pattern)
        
        pattern_info = {
            "in_i": pattern_info_in_i,
            "out_i": pattern_info_out_i,
            "in_j": pattern_info_in_j,
            "out_j": pattern_info_out_j,
            "dominant": dominant_pattern
        }
        
        return token_features, pattern_info

# ========== 时空特征注意力增强模块 ==========
class CrossModalityAlignment(nn.Module):
    """
    跨模态对齐模块，基于相似度检索增强时序特征
    """
    def __init__(self, seq_dim=128, token_dim=768, hidden_dim=64):
        """
        初始化跨模态对齐模块
        Args:
            seq_dim: 时序特征维度
            token_dim: 令牌特征维度
            hidden_dim: 隐藏层维度，用于降维
        """
        super().__init__()
        self.seq_dim = seq_dim
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        
        # 查询映射 - 将时序特征映射到隐藏空间
        self.psi_q = nn.Linear(seq_dim, hidden_dim)
        
        # 键映射 - 将令牌特征映射到隐藏空间
        self.psi_k = nn.Linear(token_dim, hidden_dim)
        
        # 值映射 - 将令牌特征映射到隐藏空间
        self.psi_v = nn.Linear(token_dim, seq_dim)
        
        # 输出层 - 用于调整输出特征
        self.omega_c = nn.Sequential(
            nn.Linear(seq_dim, seq_dim),
            nn.LayerNorm(seq_dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, H_T, L_N):
        """
        前向传播，执行相似度检索和特征增强
        Args:
            H_T: 时间序列分支输出的弱特征 [batch_size, seq_dim]
            L_N: 提示分支输出的强鲁棒特征 [batch_size, token_dim]
        Returns:
            H_C: 增强后的时序特征 [batch_size, seq_dim]
            M_T: 相似度矩阵 [batch_size, 1]
        """
        batch_size = H_T.shape[0]
        
        # 确保输入维度正确
        if len(H_T.shape) == 3:  # 如果输入是[batch_size, seq_len, seq_dim]
            # 取平均值压缩时间维度
            H_T = torch.mean(H_T, dim=1)  # [batch_size, seq_dim]
            
        if len(L_N.shape) == 1:  # 如果输入是[token_dim]
            # 扩展为批次
            L_N = L_N.unsqueeze(0).expand(batch_size, -1)  # [batch_size, token_dim]
        
        # 处理token_dim不匹配的情况
        if L_N.shape[-1] != self.token_dim:

            # 创建一个线性投影层
            projection = nn.Linear(L_N.shape[-1], self.token_dim).to(L_N.device)
            L_N = projection(L_N)
            
        # 映射到查询和键空间
        Q = self.psi_q(H_T)  # [batch_size, hidden_dim]
        K = self.psi_k(L_N)  # [batch_size, hidden_dim]
        
        # 计算相似度矩阵
        # Q: [batch_size, hidden_dim], K: [batch_size, hidden_dim]
        similarity = torch.bmm(Q.unsqueeze(1), K.unsqueeze(2)).squeeze(1).squeeze(1)
        M_T = F.softmax(similarity, dim=-1)  # [batch_size, 1]
        
        # 映射到值空间
        V = self.psi_v(L_N)  # [batch_size, seq_dim]
        
        # 加权聚合
        # M_T: [batch_size, 1], V: [batch_size, seq_dim]
        retrieved_features = V * M_T.unsqueeze(1)  # [batch_size, seq_dim]
        
        # 应用输出变换
        retrieved_features = self.omega_c(retrieved_features)
        
        # 特征融合：原始特征 + 检索到的特征
        H_C = H_T + retrieved_features
        
        return H_C, M_T



# ========== 位置编码 ==========
class PositionalEncoding(nn.Module):
    """Transformer的位置编码"""
    def __init__(self, d_model, max_len=50):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: 输入序列 [batch_size, seq_len, d_model]
        Returns:
            x: 添加位置编码后的序列 [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1)]

# ========== 时间特征编码器 ==========
class TimeFeatureEncoder(nn.Module):
    """时间特征编码器，提取日期的周期性特征"""
    def __init__(self, num_features=7, hidden_dim=128):
        """
        初始化时间特征编码器
        Args:
            num_features: 时间特征数量（如一周7天）
            hidden_dim: 隐藏层维度
        """
        super(TimeFeatureEncoder, self).__init__()
        self.num_features = num_features
        self.embedding = nn.Embedding(num_features, hidden_dim)
        
    def forward(self, time_idx):
        """
        前向传播
        Args:
            time_idx: 时间索引 [batch_size, time_steps]，值范围0-6表示一周内的天
        Returns:
            time_features: 时间特征 [batch_size, time_steps, hidden_dim]
        """
        # 确保输入是整数类型
        time_idx = time_idx.long()
        
        # 嵌入时间特征
        time_features = self.embedding(time_idx)
        
        return time_features

# ========== 时间注意力模块 ==========
class TemporalAttention(nn.Module):
    """时间注意力模块，捕捉时间序列内的依赖关系"""
    def __init__(self, hidden_dim):
        """
        Initialize temporal attention module
        Args:
            hidden_dim: Hidden layer dimension
        """
        super(TemporalAttention, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        # 在forward中计算scale以确保它在正确的设备上
        
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input features [batch_size, time_steps, hidden_dim]
        Returns:
            out: Attention output [batch_size, time_steps, hidden_dim]
        """
        # Calculate query, key, and value
        q = self.query(x)  # [batch_size, time_steps, hidden_dim]
        k = self.key(x)    # [batch_size, time_steps, hidden_dim]
        v = self.value(x)  # [batch_size, time_steps, hidden_dim]
        
        # Calculate attention scores
        attn = torch.matmul(q, k.transpose(-2, -1))  # [batch_size, time_steps, time_steps]
        
        # 在正确的设备上计算scale值
        scale = torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32, device=x.device))
        attn = attn / scale
        
        # Apply softmax to get attention weights
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention weights
        out = torch.matmul(attn, v)  # [batch_size, time_steps, hidden_dim]
        
        return out 

# ========== 时序注意力模块 ==========
class TemporalAttention(nn.Module):
    """时序注意力机制"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 注意力计算层
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        # 输出层
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入特征 [batch_size, time_steps, hidden_dim]
        Returns:
            output: 注意力输出 [batch_size, time_steps, hidden_dim]
        """
        batch_size, time_steps, _ = x.shape
        
        # 计算注意力
        q = self.query(x)  # [batch_size, time_steps, hidden_dim]
        k = self.key(x)    # [batch_size, time_steps, hidden_dim]
        v = self.value(x)  # [batch_size, time_steps, hidden_dim]
        
        # 计算注意力分数
        scores = torch.bmm(q, k.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        # 应用注意力
        context = torch.bmm(attn_weights, v)
        
        # 残差连接和层归一化
        output = self.out_proj(context)
        output = self.dropout(output)
        output = self.layer_norm(output + x)
        
        return output 

# ========== 分层时空联合注意力机制 ==========
class HierarchicalSpatioTemporalAttention(nn.Module):
    """分层时空联合注意力机制，分别建模时间和空间维度的依赖关系，然后融合"""
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 时间维度自注意力
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1
        )
        
        # 空间条件注意力
        self.spatial_query = nn.Linear(hidden_dim, hidden_dim)
        self.spatial_key = nn.Linear(hidden_dim, hidden_dim)
        self.spatial_value = nn.Linear(hidden_dim, hidden_dim)
        
        # 时间位置编码
        self.temporal_pe = PositionalEncoding(hidden_dim, max_len=50)
        
        # 空间处理层
        self.spatial_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 特征融合层 - 用于融合时间和空间注意力结果
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, 2),
            nn.Softmax(dim=-1)
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(0.1)
        )
        
        # 归一化层
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, spatial_features=None):
        """
        前向传播
        Args:
            x: 输入特征 [batch_size, time_steps, hidden_dim]
            spatial_features: 空间特征 [batch_size, hidden_dim]
        Returns:
            output: 注意力输出 [batch_size, time_steps, hidden_dim]
        """
        batch_size, time_steps, _ = x.shape
        
        # 添加时间位置编码
        x = self.temporal_pe(x)
        x_orig = x  # 保存原始输入用于残差连接
        
        # 1. 时间维度注意力
        # 转换形状以适应多头注意力
        x_t = x.transpose(0, 1)  # [time_steps, batch_size, hidden_dim]
        temporal_out, _ = self.temporal_attn(x_t, x_t, x_t)
        temporal_out = temporal_out.transpose(0, 1)  # [batch_size, time_steps, hidden_dim]
        
        # 2. 空间条件注意力
        if spatial_features is not None:
            # 处理空间特征
            spatial_encoding = self.spatial_proj(spatial_features)  # [batch_size, hidden_dim]
            
            # 计算空间条件的查询、键、值
            q = self.spatial_query(x)  # [batch_size, time_steps, hidden_dim]
            
            # 扩展空间编码到时间维度
            spatial_expanded = spatial_encoding.unsqueeze(1).expand(-1, time_steps, -1)
            k = self.spatial_key(spatial_expanded)  # [batch_size, time_steps, hidden_dim]
            v = self.spatial_value(spatial_expanded)  # [batch_size, time_steps, hidden_dim]
            
            # 计算空间注意力
            attn_weights = torch.bmm(q, k.transpose(1, 2)) / (self.hidden_dim ** 0.5)
            attn_weights = F.softmax(attn_weights, dim=-1)
            spatial_out = torch.bmm(attn_weights, v)  # [batch_size, time_steps, hidden_dim]
        else:
            # 如果没有提供空间特征，使用相同的时间特征
            spatial_out = temporal_out
        
        # 3. 时空特征融合
        combined = torch.cat([temporal_out, spatial_out], dim=-1)  # [batch_size, time_steps, hidden_dim*2]
        fusion_weights = self.fusion_gate(combined)  # [batch_size, time_steps, 2]
        
        # 加权融合时间和空间特征
        fused_out = fusion_weights[:,:,0:1] * temporal_out + fusion_weights[:,:,1:2] * spatial_out
        
        # 残差连接和层归一化
        out1 = self.norm1(x_orig + self.dropout(fused_out))
        
        # 前馈网络
        ffn_out = self.ffn(out1)
        
        # 再次残差连接和层归一化
        output = self.norm2(out1 + self.dropout(ffn_out))
        
        return output


# 保留原始实现以兼容旧代码
class SpatioTemporalAttention(HierarchicalSpatioTemporalAttention):
    """时空联合注意力机制，使用分层实现"""
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__(hidden_dim, num_heads) 

# ========== 空间特征编码器 ==========
class SpatialEncoder(nn.Module):
    """空间特征编码器"""
    def __init__(self, input_dim=6, hidden_dim=128):
        """
        初始化编码器
        Args:
            input_dim: 输入特征维度，默认为6（两个站点的进出站流量、距离和人口密度）
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        # 保存隐藏层维度作为类属性
        self.hidden_dim = hidden_dim
        
        # 特征提取网络
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim),
        )
        
        # 使用Informer进行序列特征提取（替代标准Transformer）
        self.position_encoding = PositionalEncoding(hidden_dim, max_len=50)
        # 使用Informer替代标准Transformer以更好地处理长期时序依赖关系
        self.encoder = Informer(
            enc_in=hidden_dim,  # 输入特征维度
            d_model=hidden_dim,  # 模型维度
            c_out=hidden_dim,   # 输出维度
            factor=5,           # 注意力稀疏因子
            n_heads=8,          # 注意力头数量
            e_layers=2,         # 编码器层数
            d_ff=hidden_dim*4,  # 前馈网络维度
            dropout=0.1,        # 丢弃率
            activation='gelu'   # 激活函数
        )
        
        # 时间特征编码器
        self.time_feature_encoder = TimeFeatureEncoder(num_features=7, hidden_dim=hidden_dim)
        
        # 时间注意力模块
        self.temporal_attention = TemporalAttention(hidden_dim)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )
    
    def forward(self, x, time_idx=None):
        """
        前向传播
        Args:
            x: 输入特征 [batch_size, time_steps, input_dim]
            time_idx: 时间索引 [batch_size, time_steps], 可选，表示一周内的天
        Returns:
            encoded: 编码后的特征 [batch_size, time_steps, hidden_dim]
        """
        # 检查输入是否包含NaN值
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        try:
            batch_size, time_steps, _ = x.shape
            
            # 如果未提供时间索引，创建默认的时间索引（0-6循环）
            if time_idx is None:
                time_idx = torch.arange(time_steps, device=x.device) % 7
                time_idx = time_idx.unsqueeze(0).expand(batch_size, -1)
            
            # 特征提取
            x_reshaped = x.view(batch_size * time_steps, -1)
            features = self.feature_extractor(x_reshaped)
            
            # 检查特征提取后是否包含NaN值
            if torch.isnan(features).any():
                print("警告：特征提取后包含NaN值，尝试修复...")
                features = torch.nan_to_num(features, nan=0.0)
            
            features = features.view(batch_size, time_steps, -1)
            
            # Transformer序列处理
            try:
                # 添加位置编码
                features = self.position_encoding(features)
                
                # 应用Informer编码器
                informer_out = self.encoder(features)
                
                # 检查Informer输出是否包含NaN值
                if torch.isnan(informer_out).any():
                    print("警告：Informer输出包含NaN值，尝试修复...")
                    informer_out = torch.nan_to_num(informer_out, nan=0.0)
                
                # 应用时间注意力
                attn_out = self.temporal_attention(informer_out)
                
                # 检查注意力输出是否包含NaN值
                if torch.isnan(attn_out).any():
                    print("警告：注意力输出包含NaN值，尝试修复...")
                    attn_out = torch.nan_to_num(attn_out, nan=0.0)
                
                # 添加时间特征（如果提供）
                if time_idx is not None:
                    time_features = self.time_feature_encoder(time_idx)
                    # 将时间特征与注意力输出结合
                    combined = attn_out + time_features
                else:
                    combined = attn_out
                
                # 输出层
                encoded = self.output_layer(combined)
                
                # 最终检查
                if torch.isnan(encoded).any():
                    print("警告：编码器最终输出包含NaN值，尝试修复...")
                    encoded = torch.nan_to_num(encoded, nan=0.0)
                
                return encoded
            except RuntimeError as e:
                print(f"Transformer运行时错误：{str(e)}")
                # 如果Transformer失败，跳过它并直接使用特征
                print("跳过Transformer，直接使用提取的特征...")
                # 创建具有预期输出形状的张量
                encoded = torch.zeros(batch_size, time_steps, self.hidden_dim, device=x.device)
                return encoded
        except Exception as e:
            print(f"SpatialEncoder前向传播错误：{str(e)}")
            # 返回零张量作为安全回退
            return torch.zeros(x.shape[0], x.shape[1], self.hidden_dim, device=x.device)

# ========== 动态时空特征融合 ==========
class DynamicSpatioTemporalFusion(nn.Module):
    """动态时空特征融合模块，根据空间特征调整时间特征的重要性"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 空间到时间的映射 - 为每个空间位置生成时间注意力权重
        self.space_to_time_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()  # 输出归一化的注意力权重
        )
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1)
        )
    
    def forward(self, temporal_features, spatial_features):
        """
        前向传播
        Args:
            temporal_features: 时间特征 [batch_size, time_steps, hidden_dim]
            spatial_features: 空间特征 [batch_size, hidden_dim]
        Returns:
            fused_features: 融合后的特征 [batch_size, time_steps, hidden_dim]
        """
        batch_size, time_steps, _ = temporal_features.shape
        
        # 为每个样本生成时间注意力权重
        time_weights = self.space_to_time_attention(spatial_features)  # [batch_size, hidden_dim]
        
        # 扩展到时间维度
        time_weights = time_weights.unsqueeze(1).expand(-1, time_steps, -1)  # [batch_size, time_steps, hidden_dim]
        
        # 应用动态时间注意力
        weighted_temporal = temporal_features * time_weights
        
        # 扩展空间特征到时间维度
        expanded_spatial = spatial_features.unsqueeze(1).expand(-1, time_steps, -1)  # [batch_size, time_steps, hidden_dim]
        
        # 融合时间和空间特征
        fused = torch.cat([weighted_temporal, expanded_spatial], dim=-1)  # [batch_size, time_steps, hidden_dim*2]
        fused_features = self.fusion(fused)  # [batch_size, time_steps, hidden_dim]
        
        return fused_features


# ========== GAN生成器 ==========
class ODFlowGANGenerator(nn.Module):
    """OD流量GAN生成器 """
    def __init__(self, feature_dim=6, hidden_dim=128, token_dim=768, time_steps=28, output_dim=2):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.token_dim = token_dim
        self.time_steps = time_steps
        self.output_dim = output_dim
        
        # 数据归一化层
        self.feature_norm = nn.LayerNorm(feature_dim)
        
        # 特征编码器
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim),  # 添加层归一化提高稳定性
        )
        
        # 令牌处理器
        self.token_processor = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )
        
        # 跨模态对齐模块 - 用于时序特征与令牌特征的交互
        self.cross_modal_alignment = CrossModalityAlignment(
            seq_dim=hidden_dim,
            token_dim=token_dim,
            hidden_dim=64
        )
        
        # 添加时空联合注意力机制
        self.spatiotemporal_attention = SpatioTemporalAttention(
            hidden_dim=hidden_dim,
            num_heads=8
        )
        
        # 添加动态时空特征融合模块
        self.dynamic_fusion = DynamicSpatioTemporalFusion(hidden_dim)
        
        # 空间编码器 - 用于提取空间特征
        self.spatial_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim)
        )
        
        # 最终融合层 - 应用相似度对齐后
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),  # 特征 + 噪声 = hidden_dim*2
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(0.1)
        )
        
        # 噪声投影层，用于将噪声调整到合适的维度
        self.noise_projection = nn.Linear(hidden_dim, hidden_dim * 2)
        
        # 使用Informer替代标准Transformer
        self.position_encoding = PositionalEncoding(hidden_dim * 2, max_len=50)
        # 使用Informer替代标准Transformer以更好地处理长期时序依赖关系
        self.encoder = Informer(
            enc_in=hidden_dim * 2,  # 输入维度: 融合特征
            d_model=hidden_dim * 2, # 模型维度
            c_out=hidden_dim * 2,   # 输出维度
            factor=5,                # 注意力稀疏因子
            n_heads=8,               # 注意力头数量
            e_layers=2,              # 编码器层数
            d_ff=(hidden_dim * 2)*4, # 前馈网络维度
            dropout=0.1,             # 丢弃率
            activation='gelu'        # 激活函数
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim),  # 添加层归一化
            nn.Dropout(0.1),           # 添加Dropout
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()               # 使用Sigmoid直接在模型中确保输出范围为[0,1]
        )
    
    def forward(self, features, noise, token_features=None):
        """
        前向传播 - 使用时空联合注意力机制增强特征表示
        Args:
            features: 输入特征 [batch_size, time_steps, feature_dim]
            noise: 噪声 [batch_size, time_steps, hidden_dim]
            token_features: 令牌特征 [batch_size, token_dim]
        Returns:
            od_flows: 生成的OD流量 [batch_size, time_steps, output_dim]
        """
        batch_size, time_steps, _ = features.shape
        
        # 归一化输入特征
        features = self.feature_norm(features)
        
        # 分离时间和空间特征
        temporal_features = features[:, :, 0:4]  # 站点i和j的流入流出 [batch, time_steps, 4]
        spatial_features_raw = features[:, :, 4:6]  # 距离和人口密度 [batch, time_steps, 2]
        
        # 取空间特征的平均值，得到时间无关的空间表示
        spatial_features_mean = torch.mean(spatial_features_raw, dim=1)  # [batch, 2]
        
        # 编码时间特征
        encoded_temporal = self.feature_encoder(features)  # 时序特征 [batch_size, time_steps, hidden_dim]
        
        # 编码空间特征 - 使用更丰富的空间信息
        spatial_features = self.spatial_encoder(
            torch.cat([
                features[:, 0, 0:2],  # 站点i的第一个时间步流入流出
                features[:, 0, 2:4],  # 站点j的第一个时间步流入流出
                spatial_features_mean  # 平均距离和人口密度
            ], dim=1)
        )  # [batch_size, hidden_dim]
        
        # 时序特征与令牌特征交互
        if token_features is not None:
            # 创建令牌特征的副本
            token_features_copy = token_features.clone()
            
            # 检查并适配token_features的维度
            if len(token_features_copy.shape) == 2 and token_features_copy.shape[1] != self.token_dim:

                # 创建一个临时投影层
                temp_projection = nn.Linear(token_features_copy.shape[1], self.token_dim).to(token_features_copy.device)
                token_features_copy = temp_projection(token_features_copy)
                
            # 应用跨模态对齐 - 每个时间步独立处理
            enhanced_features = []
            
            for t in range(time_steps):
                # 当前时间步的时序特征
                time_step_feat = encoded_temporal[:, t, :]  # [batch_size, hidden_dim]
                
                try:
                    # 使用相似度检索增强特征（令牌特征增强时序特征）
                    enhanced_feat, _ = self.cross_modal_alignment(time_step_feat, token_features_copy)
                    enhanced_features.append(enhanced_feat)
                except Exception as e:
                    print(f"跨模态对齐出错: {str(e)}")
                    # 失败时使用原始特征
                    enhanced_features.append(time_step_feat)
            
            # 将增强的特征堆叠回时间维度
            enhanced_temporal = torch.stack(enhanced_features, dim=1)  # [batch_size, time_steps, hidden_dim]
            
            # 分两个阶段应用时空融合
            try:
                # 1. 应用分层时空联合注意力
                st_enhanced = self.spatiotemporal_attention(enhanced_temporal, spatial_features)

                
                # 2. 应用动态时空特征融合
                dynamic_enhanced = self.dynamic_fusion(st_enhanced, spatial_features)

                
                # 最终增强特征
                final_enhanced = dynamic_enhanced
            except Exception as e:
                print(f"时空融合出错: {str(e)}")
                # 失败时使用原始增强特征
                final_enhanced = enhanced_temporal
            
            # 与噪声融合
            x = torch.cat([final_enhanced, noise], dim=-1)
            x = self.fusion_layer(x)
        else:
            # 如果没有令牌特征，仍然应用时空融合
            try:
                # 1. 应用分层时空联合注意力
                st_enhanced = self.spatiotemporal_attention(encoded_temporal, spatial_features)
                
                # 2. 应用动态时空特征融合
                dynamic_enhanced = self.dynamic_fusion(st_enhanced, spatial_features)
                
                # 与噪声融合
                x = torch.cat([dynamic_enhanced, noise], dim=-1)
                x = self.fusion_layer(x)
            except Exception as e:
                print(f"时空融合出错 (无令牌特征): {str(e)}")
                # 失败时直接使用原始特征与噪声融合
                x = torch.cat([encoded_temporal, noise], dim=-1)
                x = self.fusion_layer(x)
        
        # 应用位置编码和Informer编码器
        x = self.position_encoding(x)
        informer_out = self.encoder(x)
        
        # 输出层
        out = self.output_layer(informer_out)
        
        return out

# ========== GAN判别器 ==========
class ODFlowGANDiscriminator(nn.Module):
    """OD流量GAN判别器"""
    def __init__(self, feature_dim=6, hidden_dim=128, token_dim=768, time_steps=28, input_dim=2):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.token_dim = token_dim
        self.time_steps = time_steps
        self.input_dim = input_dim
        
        # 数据归一化层
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.flow_norm = nn.LayerNorm(input_dim)
        
        # 特征编码器 - 确保输入维度为feature_dim + input_dim（特征+OD流量）
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim + input_dim, hidden_dim),  # 输入特征 + OD流量
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim),  # 添加层归一化
        )
        
        # 空间编码器 - 用于提取空间特征
        self.spatial_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim)
        )
        
        # 令牌处理器
        self.token_processor = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )
        
        # 跨模态对齐模块 - 用于时序特征与令牌特征的交互
        self.cross_modal_alignment = CrossModalityAlignment(
            seq_dim=hidden_dim,
            token_dim=token_dim,
            hidden_dim=64
        )
        
        # 添加分层时空联合注意力机制
        self.spatiotemporal_attention = SpatioTemporalAttention(
            hidden_dim=hidden_dim,
            num_heads=8
        )
        
        # 添加动态时空特征融合模块
        self.dynamic_fusion = DynamicSpatioTemporalFusion(hidden_dim)
        
        # 最终融合层 - 应用相似度对齐后
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # 使用增强后的特征
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )
        
        # 添加时序注意力机制 - 从TemporalAttention类实例化
        self.temporal_attention = TemporalAttention(
            hidden_dim=hidden_dim
        )
        
        # 使用Informer替代标准Transformer
        self.position_encoding = PositionalEncoding(hidden_dim, max_len=50)
        # 使用Informer替代标准Transformer以更好地处理长期时序依赖关系
        self.encoder = Informer(
            enc_in=hidden_dim,      # 输入维度
            d_model=hidden_dim,     # 模型维度
            c_out=hidden_dim,       # 输出维度
            factor=5,               # 注意力稀疏因子
            n_heads=8,              # 注意力头数量
            e_layers=2,             # 编码器层数
            d_ff=hidden_dim*4,      # 前馈网络维度
            dropout=0.1,            # 丢弃率
            activation='gelu'       # 激活函数
        )
        
        # 输出层 - 不使用sigmoid，因为后面会用BCE with logits
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # Transformer输出维度
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim),  # 添加层归一化
            nn.Dropout(0.1),           # 添加Dropout
            nn.Linear(hidden_dim, 1)   # 输出logits
        )
    
    def forward(self, features, od_flows, token_features=None):
        """
        前向传播
        Args:
            features: 输入特征 [batch_size, time_steps, feature_dim]
            od_flows: OD流量 [batch_size, time_steps, input_dim]
            token_features: 令牌特征 [batch_size, token_dim]
        Returns:
            scores: 真实性分数 [batch_size, 1]
        """
        batch_size, time_steps, _ = features.shape
        
        # 归一化输入
        features = self.feature_norm(features)
        od_flows = self.flow_norm(od_flows)
        
        # 分离时间和空间特征
        temporal_features = features[:, :, 0:4]  # 站点i和j的流入流出 [batch, time_steps, 4]
        spatial_features_raw = features[:, :, 4:6]  # 距离和人口密度 [batch, time_steps, 2]
        
        # 取空间特征的平均值，得到时间无关的空间表示
        spatial_features_mean = torch.mean(spatial_features_raw, dim=1)  # [batch, 2]
        
        # 编码空间特征 - 使用更丰富的空间信息
        spatial_features = self.spatial_encoder(
            torch.cat([
                features[:, 0, 0:2],  # 站点i的第一个时间步流入流出
                features[:, 0, 2:4],  # 站点j的第一个时间步流入流出
                spatial_features_mean  # 平均距离和人口密度
            ], dim=1)
        )  # [batch_size, hidden_dim]
        
        # 连接特征与OD流量
        combined_input = torch.cat([features, od_flows], dim=-1)
        
        # 编码输入特征
        encoded_features = self.feature_encoder(combined_input)
        
        # 第一阶段：应用跨模态对齐
        if token_features is not None:
            # 创建令牌特征的副本
            token_features_copy = token_features.clone()
            
            # 检查并适配token_features的维度
            if len(token_features_copy.shape) == 2 and token_features_copy.shape[1] != self.token_dim:

                # 创建一个临时投影层
                temp_projection = nn.Linear(token_features_copy.shape[1], self.token_dim).to(token_features_copy.device)
                token_features_copy = temp_projection(token_features_copy)
            
            # 应用跨模态对齐 - 每个时间步独立处理
            enhanced_features = []
            
            for t in range(time_steps):
                # 当前时间步的特征
                time_step_feat = encoded_features[:, t, :]
                
                try:
                    # 使用相似度检索增强特征
                    enhanced_feat, _ = self.cross_modal_alignment(time_step_feat, token_features_copy)
                    enhanced_features.append(enhanced_feat)
                except Exception as e:
                    print(f"判别器跨模态对齐出错: {str(e)}")
                    # 失败时使用原始特征
                    enhanced_features.append(time_step_feat)
            
            # 将增强的特征堆叠回时间维度
            enhanced_temporal = torch.stack(enhanced_features, dim=1)
            
            # 分两个阶段应用时空融合
            try:
                # 1. 应用分层时空联合注意力
                st_enhanced = self.spatiotemporal_attention(enhanced_temporal, spatial_features)
                
                # 2. 应用动态时空特征融合
                dynamic_enhanced = self.dynamic_fusion(st_enhanced, spatial_features)

                
                # 最终增强特征
                final_features = dynamic_enhanced
            except Exception as e:
                print(f"判别器时空融合出错: {str(e)}")
                # 失败时使用原始增强特征
                final_features = enhanced_temporal
            
            # 应用特征融合层
            encoded = self.fusion_layer(final_features)
        else:
            # 如果没有令牌特征，应用时空融合
            try:
                # 1. 应用分层时空联合注意力
                st_enhanced = self.spatiotemporal_attention(encoded_features, spatial_features)
                
                # 2. 应用动态时空特征融合
                dynamic_enhanced = self.dynamic_fusion(st_enhanced, spatial_features)
                
                # 应用特征融合层
                encoded = self.fusion_layer(dynamic_enhanced)
            except Exception as e:
                print(f"判别器时空融合出错 (无令牌特征): {str(e)}")
                # 失败时直接应用特征融合层
                encoded = self.fusion_layer(encoded_features)
        
        # 应用位置编码
        encoded = self.position_encoding(encoded)
        
        # 使用Informer编码器获取全局依赖
        encoded = self.encoder(encoded)
        
        try:
            # 应用时序注意力（如果可用）
            attended = self.temporal_attention(encoded)
        except (AttributeError, Exception) as e:
            print(f"时序注意力不可用，使用原始特征: {str(e)}")
            attended = encoded
        
        # 全局平均池化 - 从时间维度汇总特征
        global_feat = torch.mean(attended, dim=1)
        
        # 应用输出层获取判别分数
        scores = self.output_layer(global_feat)
        
        return scores

# ========== 判别器森林 ==========
class ODFlowGANDiscriminatorForest(nn.Module):
    """OD流量GAN判别器森林，包含K个独立的判别器"""
    def __init__(self, feature_dim=6, hidden_dim=128, token_dim=768, time_steps=28, input_dim=2, num_discriminators=3):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.token_dim = token_dim
        self.time_steps = time_steps
        self.input_dim = input_dim
        self.num_discriminators = num_discriminators
        
        # 创建多个独立的判别器
        self.discriminators = nn.ModuleList([
            ODFlowGANDiscriminator(
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                token_dim=token_dim,
                time_steps=time_steps,
                input_dim=input_dim
            ) for _ in range(num_discriminators)
        ])
        
    def forward(self, features, od_flows, token_features=None, return_all=False):
        """
        前向传播
        Args:
            features: 输入特征 [batch_size, time_steps, feature_dim]
            od_flows: OD流量 [batch_size, time_steps, input_dim]
            token_features: 令牌特征 [batch_size, token_dim]
            return_all: 是否返回所有判别器的输出
        Returns:
            avg_scores: 平均分数 [batch_size, 1] 如果return_all=False
            all_scores: 所有判别器的分数列表 如果return_all=True
        """
        all_scores = []
        
        for discriminator in self.discriminators:
            # 每个判别器独立评估输入
            scores = discriminator(features, od_flows, token_features)
            all_scores.append(scores)
        
        if return_all:
            return all_scores
        else:
            # 计算所有判别器输出的平均值
            avg_scores = torch.mean(torch.stack(all_scores), dim=0)
            return avg_scores

# ========== OD流量GAN生成器（Forest-GAN） ==========
class ODFlowGenerator_GAN(nn.Module):
    def __init__(self, hidden_dim=128, token_dim=768, time_steps=28, num_discriminators=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_steps = time_steps
        self.token_dim = token_dim
        self.num_discriminators = num_discriminators
        
        # Create generator
        self.generator = ODFlowGANGenerator(
            feature_dim=6,  # 更新为6个特征：站点a流入流出、站点b流入流出、距离特征和人口密度特征
            hidden_dim=hidden_dim,
            token_dim=token_dim,
            time_steps=time_steps,
            output_dim=2
        )
        
        # 创建判别器森林
        self.discriminator = ODFlowGANDiscriminatorForest(
            feature_dim=6,  # 更新为6个特征：站点a流入流出、站点b流入流出、距离特征和人口密度特征
            hidden_dim=hidden_dim,
            token_dim=token_dim,
            time_steps=time_steps,
            input_dim=2,
            num_discriminators=num_discriminators
        )
        
        # 特征融合层 - 用于将OD流量融合到特征中
        self.feature_fusion = nn.Linear(6 + 2, hidden_dim)
    
    def _bootstrap_batch(self, features, od_flows, token_features=None):
        """
        对输入数据进行自举采样
        Args:
            features: 输入特征 [batch_size, time_steps, feature_dim]
            od_flows: OD流量 [batch_size, time_steps, input_dim]
            token_features: 令牌特征 [batch_size, token_dim]
        Returns:
            bootstrap_data: 自举采样后的数据，包含features、od_flows和token_features
        """
        batch_size = features.shape[0]
        
        # 有放回采样的索引
        indices = torch.randint(0, batch_size, (batch_size,), device=features.device)
        
        # 对数据进行自举采样
        sampled_features = features[indices]
        sampled_od_flows = od_flows[indices]
        
        # 如果提供了token特征，也进行自举采样
        sampled_token_features = None
        if token_features is not None:
            sampled_token_features = token_features[indices]
            
        return sampled_features, sampled_od_flows, sampled_token_features
    
    def forward(self, features, target_od=None, token_features=None, mode='train'):
        batch_size, time_steps, _ = features.shape
        
        # Check if input contains NaN values
        if torch.isnan(features).any():
            features = torch.nan_to_num(features, nan=0.0)
        
        if target_od is not None and torch.isnan(target_od).any():
            target_od = torch.nan_to_num(target_od, nan=0.0)
        
        # Generate random noise
        noise = torch.randn(batch_size, time_steps, self.hidden_dim, device=features.device)
        
        # Generate OD flow
        generated_od = self.generator(features, noise, token_features)
        
        if mode == 'train' and target_od is not None:
            # 为每个判别器创建自举数据集
            bootstrap_data_sets = [
                self._bootstrap_batch(features, target_od, token_features)
                for _ in range(self.num_discriminators)
            ]
            
            # 获取判别器森林的所有输出
            real_scores_list = []
            fake_scores_list = []
            
            for i, (bs_features, bs_target_od, bs_token_features) in enumerate(bootstrap_data_sets):
                # 确保计算图分离
                bs_features_d = bs_features.detach().clone() if bs_features.requires_grad else bs_features
                bs_target_od_d = bs_target_od.detach().clone() if bs_target_od.requires_grad else bs_target_od
                bs_token_features_d = bs_token_features.detach().clone() if bs_token_features is not None and bs_token_features.requires_grad else bs_token_features
                
                # 获取第i个判别器的输出
                real_score = self.discriminator.discriminators[i](bs_features_d, bs_target_od_d, bs_token_features_d)
                
                # 对于生成的样本
                generated_od_d = generated_od.detach().clone()  # 必须分离生成器的输出
                fake_score = self.discriminator.discriminators[i](bs_features_d, generated_od_d, bs_token_features_d)
                
                real_scores_list.append(real_score)
                fake_scores_list.append(fake_score)
            
            # 计算每个判别器的损失
            d_losses = []
            for real_score, fake_score in zip(real_scores_list, fake_scores_list):
                d_loss_real = F.binary_cross_entropy_with_logits(
                    real_score, torch.ones_like(real_score)
                )
                d_loss_fake = F.binary_cross_entropy_with_logits(
                    fake_score, torch.zeros_like(fake_score)
                )
                d_losses.append(d_loss_real + d_loss_fake)
            
            # 平均所有判别器的损失
            d_loss = torch.mean(torch.stack(d_losses))
            
            # 为生成器创建新的计算图
            # 生成器需要与所有判别器对抗
            fake_scores_g_list = []
            for i in range(self.num_discriminators):
                fake_score_g = self.discriminator.discriminators[i](features, generated_od, token_features)
                fake_scores_g_list.append(fake_score_g)
            
            # 计算生成器对抗损失
            g_loss_adv_list = [
                F.binary_cross_entropy_with_logits(fake_score_g, torch.ones_like(fake_score_g))
                for fake_score_g in fake_scores_g_list
            ]
            g_loss_adv = torch.mean(torch.stack(g_loss_adv_list))
            
            # Add L1 loss to make generated results closer to real data
            g_loss_l1 = F.l1_loss(generated_od, target_od)
            
            # Add MSE loss
            g_loss_mse = F.mse_loss(generated_od, target_od)
            
            # 添加PCC损失(皮尔逊相关系数)
            def pearson_correlation_loss(pred, target):
                # 将张量展平
                pred_flat = pred.reshape(pred.size(0), -1)
                target_flat = target.reshape(target.size(0), -1)
                
                # 计算每个样本的均值
                pred_mean = torch.mean(pred_flat, dim=1, keepdim=True)
                target_mean = torch.mean(target_flat, dim=1, keepdim=True)
                
                # 计算协方差
                pred_centered = pred_flat - pred_mean
                target_centered = target_flat - target_mean
                
                # 计算皮尔逊相关系数
                covariance = torch.sum(pred_centered * target_centered, dim=1)
                pred_std = torch.sqrt(torch.sum(pred_centered ** 2, dim=1))
                target_std = torch.sqrt(torch.sum(target_centered ** 2, dim=1))
                
                # 避免除零
                epsilon = 1e-8
                correlation = covariance / (pred_std * target_std + epsilon)
                
                # 返回损失 (1 - 相关系数的均值)，确保最小化
                return 1.0 - torch.mean(correlation)
            
            # 计算PCC损失
            g_loss_pcc = pearson_correlation_loss(generated_od, target_od)
            
            # 移除时序平滑性损失
            g_loss_smoothness = torch.tensor(0.0, device=features.device)
            
            # 使用模型自带的特征融合层，而不是每次创建临时层
            features_with_od = torch.cat([features, generated_od], dim=-1)
            hidden_features = self.feature_fusion(features_with_od)
            
            # 删除流量守恒约束，避免数据泄露问题
            flow_conservation_loss = torch.tensor(0.0, device=features.device)
            
            # 删除时空一致性损失，不再需要
            spatiotemporal_consistency_loss = torch.tensor(0.0, device=features.device)
            
            # 损失缩放因子 - 用于控制损失值的数量级
            loss_scale = 0.01
            
            # Combined loss with adjusted weights - 添加PCC损失
            g_loss = g_loss_adv + 1.0 * g_loss_l1 + 0.5 * g_loss_mse + 0.3 * g_loss_pcc
            
            # 应用缩放因子
            g_loss = g_loss * loss_scale
            
            return {
                'od_flows': generated_od,
                'g_loss': g_loss,
                'd_loss': d_loss,
                'g_loss_adv': g_loss_adv,
                'g_loss_l1': g_loss_l1,
                'g_loss_mse': g_loss_mse,
                'g_loss_pcc': g_loss_pcc,
                'g_loss_smoothness': g_loss_smoothness,
                'spatiotemporal_consistency_loss': spatiotemporal_consistency_loss,
                'flow_conservation_loss': flow_conservation_loss
            }
        else:
            # Inference mode, return only generated OD flow
            return {
                'od_flows': generated_od
            }
    
    def generate(self, features, token_features=None):
        """Generate OD flow"""
        outputs = self.forward(features, token_features=token_features, mode='sample')
        return outputs['od_flows'] 

# ========== 数据集类 ==========
class ODFlowDataset(Dataset):
    """OD流量数据集"""
    def __init__(self, io_flow_path, graph_path, od_matrix_path, test_ratio=0.2, val_ratio=0.1, seed=42):
        """
        初始化数据集
        Args:
            io_flow_path: IO流量数据路径 (站点i流入, 站点i流出, 站点j流入, 站点j流出)
            graph_path: 站点间邻接矩阵路径
            od_matrix_path: OD矩阵路径
            test_ratio: 测试集比例
            val_ratio: 验证集比例
            seed: 随机种子
        """
        super().__init__()
        self.io_flow = np.load(io_flow_path) # 原始: (时间步, 站点数, 特征)
        self.graph = np.load(graph_path) # (站点数, 站点数)
        self.od_matrix = np.load(od_matrix_path) # 原始: (时间步, 站点数, 站点数)
        
        # 转换维度顺序：从 (时间步, 站点数, 特征) 到 (站点数, 时间步, 特征)
        if self.io_flow.shape[0] == 28:  # 如果第一个维度是时间步
            self.io_flow = np.transpose(self.io_flow, (1, 0, 2))
        
        # 转换维度顺序：从 (时间步, 站点数, 站点数) 到 (站点数, 站点数, 时间步)  
        if self.od_matrix.shape[0] == 28:  # 如果第一个维度是时间步
            self.od_matrix = np.transpose(self.od_matrix, (1, 2, 0))
        
        # 设置归一化的分位数参数
        self.quantile_lower = 0.01
        self.quantile_upper = 0.99
        
        # 设置基本属性
        self.num_nodes = self.io_flow.shape[0]
        self.time_steps = self.io_flow.shape[1]
        
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
        
        print(f"数据维度: IO流量{self.io_flow.shape}, 图{self.graph.shape}, OD矩阵{self.od_matrix.shape}")
        
        # 数据一致性验证 - 确保所有数据的节点维度匹配
        assert self.graph.shape[0] == self.graph.shape[1], f"图数据必须是方阵: {self.graph.shape}"
        assert self.io_flow.shape[0] == self.graph.shape[0], f"IO流量节点数与图节点数不匹配: {self.io_flow.shape[0]} vs {self.graph.shape[0]}"
        assert self.od_matrix.shape[0] == self.graph.shape[0], f"OD矩阵节点数与图节点数不匹配: {self.od_matrix.shape[0]} vs {self.graph.shape[0]}"
        assert self.od_matrix.shape[1] == self.graph.shape[0], f"OD矩阵节点数与图节点数不匹配: {self.od_matrix.shape[1]} vs {self.graph.shape[0]}"
        
        print(f"✅ 数据一致性验证通过: {self.num_nodes}个节点, {self.time_steps}个时间步")
        
        # 站点对列表
        self.od_pairs = []
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                self.od_pairs.append((i, j))
        
        # 随机划分数据集
        self.current_indices = []
        all_indices = list(range(len(self.od_pairs)))
        random.seed(seed)
        random.shuffle(all_indices)
        
        test_size = int(len(all_indices) * test_ratio)
        val_size = int(len(all_indices) * val_ratio)
        
        self.test_indices = all_indices[:test_size]
        self.val_indices = all_indices[test_size:test_size + val_size]
        self.train_indices = all_indices[test_size + val_size:]
        
        self.set_mode('train')
    
    def set_mode(self, mode):
        """
        设置数据集模式
        Args:
            mode: 'train', 'val', 'test'
        """
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
        # 获取站点对索引
        site_pair_idx = self.current_indices[idx]
        site_i, site_j = self.od_pairs[site_pair_idx]
        
        # 获取双向OD流量 - 注意维度顺序已转换为(站点数, 站点数, 时间步)
        od_i_to_j = self.od_matrix[site_i, site_j, :]  # 从i到j的流量 (时间步,)
        od_j_to_i = self.od_matrix[site_j, site_i, :]  # 从j到i的流量 (时间步,)
        
        # 将两个方向的OD流量组合 [时间步, 2]
        od_flows = np.stack([od_i_to_j, od_j_to_i], axis=1)
        
        # 获取IO流量 - 注意维度顺序已转换为(站点数, 时间步, 特征)
        io_flow_i = self.io_flow[site_i, :, :]  # (时间步, 2)
        io_flow_j = self.io_flow[site_j, :, :]  # (时间步, 2)
        
        # 对IO流量数据进行归一化处理
        # 获取当前批次IO流量的统计信息进行归一化
        io_flow_combined = np.concatenate([io_flow_i, io_flow_j], axis=1)  # (时间步, 4)
        io_min = np.min(io_flow_combined)
        io_max = np.max(io_flow_combined)
        # 避免除零错误
        io_range = io_max - io_min
        if io_range == 0:
            io_range = 1.0
        
        # 归一化IO流量
        io_flow_i_normalized = (io_flow_i - io_min) / io_range
        io_flow_j_normalized = (io_flow_j - io_min) / io_range
        
        # 归一化OD流量 - 使用当前OD流量的分位数而不是全局统计
        od_min = np.min(od_flows)
        od_max = np.max(od_flows)
        # 避免除零错误
        od_range = od_max - od_min
        if od_range == 0:
            od_range = 1.0
        
        od_flows_normalized = (od_flows - od_min) / od_range
        
        # 获取站点对距离并归一化
        distance = self.graph[site_i, site_j]
        # 对距离进行归一化处理
        distance_normalized = distance / np.max(self.graph)  # 归一化到[0,1]范围
        
        # 获取站点人口密度并归一化
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
        
        # 构建人口密度特征向量 [时间步, 1]
        pop_density_feature = np.ones((io_flow_i.shape[0], 1)) * pop_density_normalized
        
        # 构建特征向量 [时间步, 6]，包含站点i的流入流出，站点j的流入流出，站点对距离特征，以及人口密度特征
        distance_feature = np.ones((io_flow_i.shape[0], 1)) * distance_normalized
        features = np.concatenate([io_flow_i_normalized, io_flow_j_normalized, distance_feature, pop_density_feature], axis=1)
        
        # 返回特征和归一化后的OD流量
        return torch.FloatTensor(features), torch.FloatTensor(od_flows_normalized)

# ========== 训练函数 ==========
def train_llm_gan(args, precomputed_token_features=None):
    """
    使用大模型特征训练GAN
    Args:
        args: 参数对象
    Returns:
        best_model_path: 最佳模型路径
    """
    # 设置随机种子
    set_seed(args.seed)
    
    # 启用异常检测，帮助定位梯度问题
    torch.autograd.set_detect_anomaly(True)
    
    # 设置设备（使用CUDA 2）
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据集
    dataset = ODFlowDataset(
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
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    dataset.set_mode('val')
    val_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    dataset.set_mode('train')
    
    # 创建测试集数据加载器
    dataset.set_mode('test')
    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    dataset.set_mode('train')
    
    # 🔧 修改：使用传入的预计算token特征
    if precomputed_token_features is None:
        raise ValueError("必须提供预计算的token特征")
    
    print(f"训练函数接收到预计算token特征，包含 {len(precomputed_token_features)} 个站点对")
    
    
    # 不再需要QwenFeatureExtractor，注释掉
    # qwen_extractor = QwenFeatureExtractor(
    #     feature_dim=args.token_dim,
    #     device=device
    # )
    
    # 创建Forest-GAN模型
    model = ODFlowGenerator_GAN(
        hidden_dim=args.hidden_dim,
        token_dim=args.token_dim,
        time_steps=28,
        num_discriminators=args.num_discriminators
    ).to(device)
    
    # 优化器 - Forest-GAN架构
    g_optimizer = torch.optim.AdamW(model.generator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    d_optimizer = torch.optim.AdamW(model.discriminator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 学习率调度器
    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(g_optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(d_optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    # 训练准备
    best_val_loss = float('inf')
    best_model_path = os.path.join(args.output_dir, 'best_forest_gan_model.pth')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 日志文件
    log_file = os.path.join(args.output_dir, 'training_log.txt')
    with open(log_file, 'w') as f:
        f.write(f"Training start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Parameters: {args}\n\n")
    
    # 早停设置
    early_stop_counter = 0
    early_stop_patience = args.patience
        

    
    # 开始训练
    print(f"Starting to train Forest-GAN model with {args.num_discriminators} discriminators...")
    try:
        for epoch in range(args.epochs):
            model.train()
            train_g_losses, train_d_losses = [], []
            train_token_losses, train_flow_losses = [], []
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Training)")
            for batch in pbar:
                # 清除缓存和梯度，防止内存泄漏
                torch.cuda.empty_cache()
                
                # 从DataLoader获取数据 (返回的是features和od_flows)
                features, od_flows = batch
                batch_size = features.shape[0]
                
                # 打印批次数据的形状
                print(f"\n批次数据形状 - batch_size: {batch_size}")
                print(f"features形状: {features.shape}")
                print(f"od_flows形状: {od_flows.shape}")
                
                # 将数据移至设备
                features = features.to(device)
                od_flows = od_flows.to(device)
                
                # 检查并确保od_flows的形状是[batch_size, time_steps, 2]
                if len(od_flows.shape) == 2:  # [batch_size, time_steps]
                    # 如果只有一个方向的流量，将其扩展为两个方向
                    print(f"od_flows形状不正确，扩展维度...")
                    od_flows = od_flows.unsqueeze(-1)  # [batch_size, time_steps, 1]
                    od_flows = od_flows.repeat(1, 1, 2)  # 复制到两个方向 [batch_size, time_steps, 2]
                    print(f"扩展后od_flows形状: {od_flows.shape}")
                
                # 确保形状正确
                time_steps = features.size(1)
                if od_flows.size(1) != time_steps or od_flows.size(2) != 2:
                    print(f"警告: OD流量形状不正确，当前为 {od_flows.shape}，应该是 [{batch_size}, {time_steps}, 2]")
                    # 如果维度不匹配，尝试调整
                    if od_flows.size(1) == 2 and od_flows.size(2) == time_steps:
                        # 可能维度顺序错误，进行转置
                        od_flows = od_flows.transpose(1, 2)
                    elif od_flows.dim() == 2 and od_flows.size(1) == time_steps:
                        # 缺少最后一个维度
                        od_flows = od_flows.unsqueeze(-1)
                        od_flows = od_flows.repeat(1, 1, 2)  # 复制到两个方向
                    print(f"调整后的OD流量形状: {od_flows.shape}")
                
                print(f"构建的特征向量形状: {features.shape}")
                print(f"特征数据类型: {features.dtype}")
                
                # 检查NaN
                if torch.isnan(features).any():
                    print("警告: 特征包含NaN值，已替换为0")
                    features = torch.nan_to_num(features, nan=0.0)
                if torch.isnan(od_flows).any():
                    print("警告: OD流量包含NaN值，已替换为0")
                    od_flows = torch.nan_to_num(od_flows, nan=0.0)
                
                # 获取当前批次的站点对
                batch_start_idx = pbar.n * args.batch_size
                batch_indices = torch.arange(batch_start_idx, batch_start_idx + batch_size)
                # 确保索引不会超出范围
                batch_indices = batch_indices[batch_indices < len(dataset.current_indices)]
                site_pairs = [dataset.od_pairs[dataset.current_indices[idx.item()]] for idx in batch_indices]
                
                print(f"站点对数量: {len(site_pairs)}")
                
                # 🔧 修改：使用预计算的令牌特征，而不是重新计算
                token_features_batch = []
                for i, (site_i, site_j) in enumerate(site_pairs):
                    # 构建站点对的键（与预计算时保持一致）
                    pair_key = f"{site_i}_{site_j}"
                    reverse_pair_key = f"{site_j}_{site_i}"
                    
                    # 尝试获取预计算的token特征
                    if pair_key in precomputed_token_features:
                        token_feature = precomputed_token_features[pair_key].to(device)
                    elif reverse_pair_key in precomputed_token_features:
                        token_feature = precomputed_token_features[reverse_pair_key].to(device)
                    else:
                        # 如果找不到预计算特征，使用零向量
                        print(f"警告：未找到站点对 ({site_i}, {site_j}) 的预计算token特征，使用零向量")
                        token_feature = torch.zeros(args.token_dim, device=device)
                    
                    token_features_batch.append(token_feature)
                
                # 将令牌特征组合为批次
                if token_features_batch:
                    token_features = torch.stack(token_features_batch).to(device)
                    print(f"令牌特征形状: {token_features.shape}")
                else:
                    # 如果批次为空，创建零张量
                    token_features = torch.zeros(batch_size, args.token_dim, device=device)
                    print(f"创建空令牌特征，形状: {token_features.shape}")
                
                # 打印特征形状以便调试
                print(f"令牌特征形状: {token_features.shape}")
                
                # 优化判别器
                d_optimizer.zero_grad()
                try:

                    # 创建输入的副本，确保不会影响生成器的计算图
                    features_d = features.detach().clone()
                    od_flows_d = od_flows.detach().clone()
                    token_features_d = token_features.detach().clone()
                    
                    # 使用detach的输入执行前向传播，只计算判别器的损失
                    with torch.set_grad_enabled(True):
                        # 我们只计算判别器的损失，使用forward方法的返回值
                        # 令牌特征会在内部增强时序特征
                        
                        # 传递特征给模型
                        out = model(features_d, od_flows_d, token_features_d, mode='train')

                        
                        # 获取判别器的损失
                        d_loss = out['d_loss']
                        
                        # 单独进行判别器的反向传播
                        d_loss.backward()
                        
                        # 梯度裁剪 - Forest-GAN判别器森林
                        torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), max_norm=1.0)
                        
                        # 优化判别器参数
                        d_optimizer.step()
                        
                        # 记录判别器损失
                        train_d_losses.append(d_loss.item())
                except Exception as e:
                    print(f"判别器前向传播或反向传播出错: {str(e)}")
                    traceback.print_exc()
                    continue
                
                # 优化生成器 - 完全独立的前向传播和反向传播
                g_optimizer.zero_grad()
                try:

                    # 确保使用新的计算图
                    with torch.set_grad_enabled(True):
                        # 令牌特征用于增强时序特征
                        
                        # 传递特征给模型
                        out_g = model(features, od_flows, token_features, mode='train')
                        g_loss = out_g['g_loss']
                        
                        # 生成器反向传播
                        g_loss.backward()
                        
                        # 梯度裁剪 - Forest-GAN生成器
                        torch.nn.utils.clip_grad_norm_(model.generator.parameters(), max_norm=1.0)
                        
                        # 优化生成器参数
                        g_optimizer.step()
                        
                        # 记录损失
                        train_g_losses.append(g_loss.item())
                        train_token_losses.append(out_g['spatiotemporal_consistency_loss'].item())
                        train_flow_losses.append(out_g['flow_conservation_loss'].item())
                        
                        # 显示进度
                        pbar.set_postfix({
                            'g_loss': f"{g_loss.item():.4f}", 
                            'd_loss': f"{d_loss.item():.4f}",
                            'st_loss': f"{out_g['spatiotemporal_consistency_loss'].item():.4f}",
                            'flow_loss': f"{out_g['flow_conservation_loss'].item():.4f}"
                        })
                except Exception as e:
                    print(f"生成器反向传播出错: {str(e)}")
                    traceback.print_exc()
                    continue
            
            # 更新学习率
            scheduler_g.step()
            scheduler_d.step()
            
            # 计算平均损失
            avg_g_loss = np.mean(train_g_losses) if train_g_losses else float('inf')
            avg_d_loss = np.mean(train_d_losses) if train_d_losses else float('inf')
            avg_token_loss = np.mean(train_token_losses) if train_token_losses else float('inf')
            avg_flow_loss = np.mean(train_flow_losses) if train_flow_losses else float('inf')
            
            # 验证
            model.eval()
            val_losses = []
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Validation)"):
                    # 清除缓存，防止内存泄漏
                    torch.cuda.empty_cache()
                    
                    # 获取特征和OD流量
                    features, od_flows = batch
                    batch_size = features.shape[0]
                    
                    # 将数据移至设备
                    features = features.to(device)
                    od_flows = od_flows.to(device)
                    
                    # 检查并确保od_flows的形状是[batch_size, time_steps, 2]
                    if len(od_flows.shape) == 2:  # [batch_size, time_steps]
                        # 如果只有一个方向的流量，将其扩展为两个方向
                        print(f"od_flows形状不正确，扩展维度...")
                        od_flows = od_flows.unsqueeze(-1)  # [batch_size, time_steps, 1]
                        od_flows = od_flows.repeat(1, 1, 2)  # 复制到两个方向 [batch_size, time_steps, 2]
                        print(f"扩展后od_flows形状: {od_flows.shape}")
                    
                    # 确保形状正确
                    time_steps = features.size(1)
                    if od_flows.size(1) != time_steps or od_flows.size(2) != 2:
                        print(f"警告: OD流量形状不正确，当前为 {od_flows.shape}，应该是 [{batch_size}, {time_steps}, 2]")
                        # 如果维度不匹配，尝试调整
                        if od_flows.size(1) == 2 and od_flows.size(2) == time_steps:
                            # 可能维度顺序错误，进行转置
                            od_flows = od_flows.transpose(1, 2)
                        elif od_flows.dim() == 2 and od_flows.size(1) == time_steps:
                            # 缺少最后一个维度
                            od_flows = od_flows.unsqueeze(-1)
                            od_flows = od_flows.repeat(1, 1, 2)  # 复制到两个方向
                        print(f"调整后的OD流量形状: {od_flows.shape}")
                    
                    # 检查NaN
                    if torch.isnan(features).any():
                        print("警告: 特征包含NaN值，已替换为0")
                        features = torch.nan_to_num(features, nan=0.0)
                    if torch.isnan(od_flows).any():
                        print("警告: OD流量包含NaN值，已替换为0")
                        od_flows = torch.nan_to_num(od_flows, nan=0.0)
                    
                    try:
                        # 获取当前批次的站点对
                        batch_start_idx = len(all_preds) * args.batch_size
                        batch_indices = batch_start_idx + torch.arange(batch_size)
                        batch_indices = batch_indices[batch_indices < len(dataset.current_indices)]
                        site_pairs = [dataset.od_pairs[dataset.current_indices[idx.item()]] for idx in batch_indices]
                        
                        # 🔧 修改：使用预计算的令牌特征（验证阶段）
                        token_features_batch = []
                        for i, (site_i, site_j) in enumerate(site_pairs):
                            # 构建站点对的键（与预计算时保持一致）
                            pair_key = f"{site_i}_{site_j}"
                            reverse_pair_key = f"{site_j}_{site_i}"
                            
                            # 尝试获取预计算的token特征
                            if pair_key in precomputed_token_features:
                                token_feature = precomputed_token_features[pair_key].to(device)
                            elif reverse_pair_key in precomputed_token_features:
                                token_feature = precomputed_token_features[reverse_pair_key].to(device)
                            else:
                                # 如果找不到预计算特征，使用零向量
                                print(f"警告：验证时未找到站点对 ({site_i}, {site_j}) 的预计算token特征，使用零向量")
                                token_feature = torch.zeros(args.token_dim, device=device)
                            
                            token_features_batch.append(token_feature)
                        
                        # 将令牌特征组合为批次
                        if token_features_batch:
                            token_features = torch.stack(token_features_batch).to(device)
                        else:
                            # 如果批次为空，创建零张量
                            token_features = torch.zeros(batch_size, args.token_dim, device=device)
                        
                        # 不再预先融合特征，直接传递
                        print(f"验证阶段 - 令牌特征形状: {token_features.shape}")
                        
                        # 生成OD流量 - 使用推理模式
                        pred_od_flows = model.generate(features, token_features)
                        
                        # 计算损失
                        loss = F.mse_loss(pred_od_flows, od_flows)
                        val_losses.append(loss.item())
                        all_preds.append(pred_od_flows.cpu().numpy())
                        all_targets.append(od_flows.cpu().numpy())
                    except Exception as e:
                        print(f"Error occurred during validation: {str(e)}")
                        traceback.print_exc()
                        continue
            
            # 计算验证指标
            avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
            
            # 计算RMSE、MAE、PCC
            if all_preds and all_targets:
                all_preds = np.concatenate(all_preds, axis=0)
                all_targets = np.concatenate(all_targets, axis=0)
                
                # 检查NaN
                valid_indices = ~np.isnan(all_preds) & ~np.isnan(all_targets)
                all_preds = all_preds[valid_indices]
                all_targets = all_targets[valid_indices]
                
                if len(all_preds) > 0:
                    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
                    mae = np.mean(np.abs(all_preds - all_targets))
                    
                    # 计算PCC - 优化计算以提高准确性
                    all_preds_flat = all_preds.reshape(-1)
                    all_targets_flat = all_targets.reshape(-1)
                    
                    # 更严格的数据清理
                    valid_indices = ~(np.isnan(all_preds_flat) | np.isnan(all_targets_flat) | np.isinf(all_preds_flat) | np.isinf(all_targets_flat))
                    
                    if np.sum(valid_indices) > 10:  # 确保有足够的有效数据点
                        pred_valid = all_preds_flat[valid_indices]
                        target_valid = all_targets_flat[valid_indices]
                        
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
                else:
                    rmse = float('nan')
                    mae = float('nan')
                    pcc = 0.0
            else:
                rmse = float('nan')
                mae = float('nan')
                pcc = 0.0

            # 测试集评估
            test_losses = []
            test_all_preds = []
            test_all_targets = []
            
            dataset.set_mode('test')
            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Testing)"):
                    # 清除缓存，防止内存泄漏
                    torch.cuda.empty_cache()
                    
                    # 获取特征和OD流量
                    features, od_flows = batch
                    batch_size = features.shape[0]
                    
                    # 将数据移至设备
                    features = features.to(device)
                    od_flows = od_flows.to(device)
                    
                    # 检查并确保od_flows的形状是[batch_size, time_steps, 2]
                    if len(od_flows.shape) == 2:  # [batch_size, time_steps]
                        od_flows = od_flows.unsqueeze(-1)  # [batch_size, time_steps, 1]
                        od_flows = od_flows.repeat(1, 1, 2)  # 复制到两个方向 [batch_size, time_steps, 2]
                    
                    # 确保形状正确
                    time_steps = features.size(1)
                    if od_flows.size(1) != time_steps or od_flows.size(2) != 2:
                        if od_flows.size(1) == 2 and od_flows.size(2) == time_steps:
                            od_flows = od_flows.transpose(1, 2)
                        elif od_flows.dim() == 2 and od_flows.size(1) == time_steps:
                            od_flows = od_flows.unsqueeze(-1)
                            od_flows = od_flows.repeat(1, 1, 2)  # 复制到两个方向
                    
                    # 检查NaN
                    if torch.isnan(features).any():
                        features = torch.nan_to_num(features, nan=0.0)
                    if torch.isnan(od_flows).any():
                        od_flows = torch.nan_to_num(od_flows, nan=0.0)
                    
                    try:
                        # 获取当前批次的站点对
                        batch_start_idx = len(test_all_preds) * args.batch_size
                        batch_indices = batch_start_idx + torch.arange(batch_size)
                        batch_indices = batch_indices[batch_indices < len(dataset.current_indices)]
                        site_pairs = [dataset.od_pairs[dataset.current_indices[idx.item()]] for idx in batch_indices]
                        
                        # 🔧 修改：使用预计算的令牌特征（测试阶段）
                        token_features_batch = []
                        for i, (site_i, site_j) in enumerate(site_pairs):
                            # 构建站点对的键（与预计算时保持一致）
                            pair_key = f"{site_i}_{site_j}"
                            reverse_pair_key = f"{site_j}_{site_i}"
                            
                            # 尝试获取预计算的token特征
                            if pair_key in precomputed_token_features:
                                token_feature = precomputed_token_features[pair_key].to(device)
                            elif reverse_pair_key in precomputed_token_features:
                                token_feature = precomputed_token_features[reverse_pair_key].to(device)
                            else:
                                # 如果找不到预计算特征，使用零向量
                                print(f"警告：测试时未找到站点对 ({site_i}, {site_j}) 的预计算token特征，使用零向量")
                                token_feature = torch.zeros(args.token_dim, device=device)
                            
                            token_features_batch.append(token_feature)
                        
                        # 将令牌特征组合为批次
                        if token_features_batch:
                            token_features = torch.stack(token_features_batch).to(device)
                        else:
                            # 如果批次为空，创建零张量
                            token_features = torch.zeros(batch_size, args.token_dim, device=device)
                        
                        # 生成OD流量 - 使用推理模式
                        pred_od_flows = model.generate(features, token_features)
                        
                        # 计算损失
                        loss = F.mse_loss(pred_od_flows, od_flows)
                        test_losses.append(loss.item())
                        test_all_preds.append(pred_od_flows.cpu().numpy())
                        test_all_targets.append(od_flows.cpu().numpy())
                    except Exception as e:
                        print(f"Error occurred during testing: {str(e)}")
                        continue
            
            dataset.set_mode('train')  # 恢复训练模式
            
            # 计算测试指标
            test_avg_val_loss = np.mean(test_losses) if test_losses else float('inf')
            
            # 计算测试集RMSE、MAE、PCC
            if test_all_preds and test_all_targets:
                test_all_preds = np.concatenate(test_all_preds, axis=0)
                test_all_targets = np.concatenate(test_all_targets, axis=0)
                
                # 检查NaN
                test_valid_indices = ~np.isnan(test_all_preds) & ~np.isnan(test_all_targets)
                test_all_preds = test_all_preds[test_valid_indices]
                test_all_targets = test_all_targets[test_valid_indices]
                
                if len(test_all_preds) > 0:
                    test_rmse = np.sqrt(np.mean((test_all_preds - test_all_targets) ** 2))
                    test_mae = np.mean(np.abs(test_all_preds - test_all_targets))
                    
                    # 计算测试集PCC - 优化计算以提高准确性
                    test_all_preds_flat = test_all_preds.reshape(-1)
                    test_all_targets_flat = test_all_targets.reshape(-1)
                    
                    # 更严格的数据清理
                    test_valid_indices = ~(np.isnan(test_all_preds_flat) | np.isnan(test_all_targets_flat) | np.isinf(test_all_preds_flat) | np.isinf(test_all_targets_flat))
                    
                    if np.sum(test_valid_indices) > 10:  # 确保有足够的有效数据点
                        test_pred_valid = test_all_preds_flat[test_valid_indices]
                        test_target_valid = test_all_targets_flat[test_valid_indices]
                        
                        # 检查方差是否为0（避免除零错误）
                        if np.var(test_pred_valid) > 1e-10 and np.var(test_target_valid) > 1e-10:
                            try:
                                test_correlation_matrix = np.corrcoef(test_pred_valid, test_target_valid)
                                test_pcc = test_correlation_matrix[0, 1]
                                
                                # 确保测试PCC在合理范围内
                                if np.isnan(test_pcc) or np.isinf(test_pcc):
                                    test_pcc = 0.0
                                else:
                                    test_pcc = np.clip(test_pcc, -1.0, 1.0)  # 限制在[-1, 1]范围内
                            except Exception as e:
                                print(f"⚠️ 测试PCC计算异常: {e}")
                                test_pcc = 0.0
                        else:
                            # 如果方差为0，说明预测值或目标值是常数
                            test_pcc = 0.0
                    else:
                        test_pcc = 0.0
                else:
                    test_rmse = float('nan')
                    test_mae = float('nan')
                    test_pcc = 0.0
            else:
                test_rmse = float('nan')
                test_mae = float('nan')
                test_pcc = 0.0
            
            # 打印训练信息
            print(f"Epoch {epoch+1}/{args.epochs}")
            print(f"   Training - G_Loss: {avg_g_loss:.6f}, D_Loss: {avg_d_loss:.6f}, ST_Loss: {avg_token_loss:.6f}, Flow_Loss: {avg_flow_loss:.6f}")
            print(f"   Validation - RMSE: {rmse:.6f}, MAE: {mae:.6f}, PCC: {pcc:.6f}")
            print(f"   Test - RMSE: {test_rmse:.6f}, MAE: {test_mae:.6f}, PCC: {test_pcc:.6f}")
            
            # 记录到日志
            with open(log_file, 'a') as f:
                f.write(f"Epoch {epoch+1}/{args.epochs}\n")
                f.write(f"   Training - G_Loss: {avg_g_loss:.6f}, D_Loss: {avg_d_loss:.6f}, ST_Loss: {avg_token_loss:.6f}, Flow_Loss: {avg_flow_loss:.6f}\n")
                f.write(f"   Validation - RMSE: {rmse:.6f}, MAE: {mae:.6f}, PCC: {pcc:.6f}\n")
                f.write(f"   Test - RMSE: {test_rmse:.6f}, MAE: {test_mae:.6f}, PCC: {test_pcc:.6f}\n")
            
            # 检查早停 - 使用组合指标，给予PCC更高的权重
            # 只有当有有效指标时才保存模型
            if not np.isnan(rmse) and not np.isnan(mae) and pcc != 0.0:
                # 修改组合分数计算方式，更强调PCC（序列趋势相关性）
                combined_score = rmse - pcc * 1.2  # 值越低越好（较低RMSE和较高PCC）
                
                if combined_score < best_val_loss:
                    best_val_loss = combined_score
                    early_stop_counter = 0
                    
                    # 保存最佳Forest-GAN模型
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'g_optimizer_state_dict': g_optimizer.state_dict(),
                        'd_optimizer_state_dict': d_optimizer.state_dict(),
                        'rmse': rmse,
                        'mae': mae,
                        'pcc': pcc,
                        'epoch': epoch,
                        'token_dim': args.token_dim,
                        'hidden_dim': args.hidden_dim,
                        'num_discriminators': args.num_discriminators,
                        'model_type': 'Forest-GAN'
                    }, best_model_path)
                    print(f"   New best model saved (RMSE: {rmse:.6f}, PCC: {pcc:.6f}, Combined score: {combined_score:.6f})")
                    with open(log_file, 'a') as f:
                        f.write(f"   New best model saved (RMSE: {rmse:.6f}, PCC: {pcc:.6f}, Combined score: {combined_score:.6f})\n")
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= early_stop_patience:
                        print(f"Early stopping: {early_stop_patience} epochs of validation performance did not improve")
                        break
            else:
                early_stop_counter += 1
                if early_stop_counter >= early_stop_patience:
                    print(f"Early stopping: Invalid metrics for {early_stop_patience} epochs")
                    break
            
            # 不再每5个epoch保存检查点，只保存最佳模型和最后一轮
            
            print(f"   Best result so far: RMSE: {best_val_loss:.6f}")
        
        # 保存最后一轮Forest-GAN模型
        final_model_path = os.path.join(args.output_dir, 'final_forest_gan_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'g_optimizer_state_dict': g_optimizer.state_dict(),
            'd_optimizer_state_dict': d_optimizer.state_dict(),
            'rmse': rmse if not np.isnan(rmse) else -1,
            'mae': mae if not np.isnan(mae) else -1,
            'pcc': pcc,
            'epoch': epoch,
            'token_dim': args.token_dim,
            'hidden_dim': args.hidden_dim,
            'num_discriminators': args.num_discriminators,
            'model_type': 'Forest-GAN'
        }, final_model_path)
        print(f"Final model saved to {final_model_path}")
        
        # 训练完成
        print(f"Training completed, best combined score: {best_val_loss:.6f}")
        with open(log_file, 'a') as f:
            f.write(f"\nTraining completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Best combined score: {best_val_loss:.6f}\n")
            f.write(f"Final model saved to {final_model_path}\n")
    
    except KeyboardInterrupt:
        print("Training interrupted by user")
        with open(log_file, 'a') as f:
            f.write(f"\nTraining interrupted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    except Exception as e:
        print(f"Error occurred during training: {str(e)}")
        traceback.print_exc()
        with open(log_file, 'a') as f:
            f.write(f"\nError occurred during training: {str(e)}\n")
            f.write(traceback.format_exc())
    
    return best_model_path

def evaluate_llm_gan_model(test_loader, precomputed_token_features, model, device, args, output_dir=None):
    """
    评估模型性能
    Args:
        test_loader: 测试数据加载器
        precomputed_token_features: 预计算的token特征字典
        model: GAN模型
        device: 设备
        args: 参数对象
        output_dir: 输出目录
    Returns:
        metrics: 评估指标
    """
    model.eval()
    
    # 评估指标
    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0
    
    # 用于PCC计算
    all_generated = []
    all_targets = []
    
    # 保存前5个样本
    saved_samples = []
    
    # 获取数据集以供获取原始站点对
    dataset = test_loader.dataset
    
    
    with torch.no_grad():
        for batch_idx, (features, od_flows_normalized) in enumerate(tqdm(test_loader, desc="Evaluating model")):
            # 获取批次大小
            batch_size = features.shape[0]
            
            # 获取当前批次的站点对
            batch_start_idx = batch_idx * args.batch_size
            batch_indices = batch_start_idx + torch.arange(batch_size)
            batch_indices = batch_indices[batch_indices < len(dataset.current_indices)]
            site_pairs = [dataset.od_pairs[dataset.current_indices[idx.item()]] for idx in batch_indices]
            
            # 如果需要保存样本
            current_batch_size = min(features.shape[0], 5 - len(saved_samples))
            
            # 移动数据到设备
            features = features.to(device)
            od_flows_normalized = od_flows_normalized.to(device)
            
            # 检查NaN值
            if torch.isnan(features).any():
                features = torch.nan_to_num(features, nan=0.0)
            if torch.isnan(od_flows_normalized).any():
                od_flows_normalized = torch.nan_to_num(od_flows_normalized, nan=0.0)
            
            # 🔧 修改：使用预计算的令牌特征
            token_features_batch = []
            for i, (site_i, site_j) in enumerate(site_pairs):
                # 构建站点对的键（与预计算时保持一致）
                pair_key = f"{site_i}_{site_j}"
                reverse_pair_key = f"{site_j}_{site_i}"
                
                # 尝试获取预计算的token特征
                if pair_key in precomputed_token_features:
                    token_feature = precomputed_token_features[pair_key].to(device)
                elif reverse_pair_key in precomputed_token_features:
                    token_feature = precomputed_token_features[reverse_pair_key].to(device)
                else:
                    # 如果找不到预计算特征，使用零向量
                    print(f"警告：评估时未找到站点对 ({site_i}, {site_j}) 的预计算token特征，使用零向量")
                    token_feature = torch.zeros(args.token_dim, device=device)
                
                token_features_batch.append(token_feature)
            
            # 将令牌特征组合为批次
            if token_features_batch:
                token_features = torch.stack(token_features_batch).to(device)
            else:
                # 如果批次为空，创建零张量
                token_features = torch.zeros(batch_size, args.token_dim, device=device)
            
            print(f"评估阶段 - 令牌特征形状: {token_features.shape}")
            
            # 生成样本，直接传递token_features
            generated_normalized = model.generate(features, token_features)
            
            # 计算MSE和MAE
            mse = F.mse_loss(generated_normalized, od_flows_normalized, reduction='sum').item()
            mae = F.l1_loss(generated_normalized, od_flows_normalized, reduction='sum').item()
            
            # 存储用于PCC计算
            all_generated.append(generated_normalized.cpu().numpy())
            all_targets.append(od_flows_normalized.cpu().numpy())
            
            # 保存前5个样本的数据（反归一化后）
            if len(saved_samples) < 5:
                # 获取反归一化参数 - 确保dataset有quantile_lower和quantile_upper属性
                if hasattr(dataset, 'quantile_lower') and hasattr(dataset, 'quantile_upper'):
                    q_lower = np.percentile(dataset.od_matrix.flatten(), dataset.quantile_lower * 100)
                    q_upper = np.percentile(dataset.od_matrix.flatten(), dataset.quantile_upper * 100)
                else:
                    # 默认使用1%和99%的分位数
                    q_lower = np.percentile(dataset.od_matrix.flatten(), 1)
                    q_upper = np.percentile(dataset.od_matrix.flatten(), 99)
                
                # 反归一化
                generated = generated_normalized.cpu().numpy() * (q_upper - q_lower) + q_lower
                real_od = od_flows_normalized.cpu().numpy() * (q_upper - q_lower) + q_lower
                
                # 保存样本数据
                for i in range(current_batch_size):
                    if len(saved_samples) < 5 and i < len(site_pairs):
                        saved_samples.append({
                            'station_pair': site_pairs[i],
                            'predicted': generated[i],
                            'real': real_od[i]
                        })
            
            # 累计指标
            total_mse += mse
            total_mae += mae
            total_samples += od_flows_normalized.shape[0] * od_flows_normalized.shape[1] * od_flows_normalized.shape[2]  # 样本 * 时间步 * 特征
    
    # 计算平均指标
    avg_mse = total_mse / total_samples
    avg_mae = total_mae / total_samples
    rmse = np.sqrt(avg_mse)
    
    # 将3D数组变为序列评估格式
    all_generated_array = np.concatenate(all_generated, axis=0)  # [batch*N, time_steps, 2]
    all_targets_array = np.concatenate(all_targets, axis=0)      # [batch*N, time_steps, 2]
    
    batch_size, time_steps, dims = all_generated_array.shape
    
    # 计算每个时间序列维度的PCC
    pcc_by_dim = []
    dtw_distances = []
    
    for dim in range(dims):
        dim_pccs = []
        dim_dtws = []
        
        for i in range(batch_size):
            # 获取每个样本的时间序列
            pred_seq = all_generated_array[i, :, dim]
            true_seq = all_targets_array[i, :, dim]
            
            # 过滤NaN值
            valid_indices = ~np.isnan(pred_seq) & ~np.isnan(true_seq)
            if np.sum(valid_indices) > 0:
                # 计算PCC
                if np.std(pred_seq[valid_indices]) > 1e-8 and np.std(true_seq[valid_indices]) > 1e-8:
                    pcc_val = np.corrcoef(pred_seq[valid_indices], true_seq[valid_indices])[0, 1]
                    dim_pccs.append(pcc_val)
                
                # 计算简化的DTW距离（动态时间规整距离）- 评估时序模式的相似度
                # 简单实现，实际使用可以用fastdtw库
                from scipy.spatial.distance import euclidean
                
                def simple_dtw(s1, s2):
                    # 创建成本矩阵
                    n, m = len(s1), len(s2)
                    dtw_matrix = np.zeros((n+1, m+1))
                    
                    # 初始化第一行和第一列为无穷大
                    dtw_matrix[0, :] = np.inf
                    dtw_matrix[:, 0] = np.inf
                    dtw_matrix[0, 0] = 0
                    
                    # 填充DTW矩阵
                    for i in range(1, n+1):
                        for j in range(1, m+1):
                            cost = abs(s1[i-1] - s2[j-1])
                            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], 
                                                         dtw_matrix[i, j-1], 
                                                         dtw_matrix[i-1, j-1])
                    
                    # 返回归一化的DTW距离
                    return dtw_matrix[n, m] / (n + m)
                
                dtw_dist = simple_dtw(pred_seq[valid_indices], true_seq[valid_indices])
                dim_dtws.append(dtw_dist)
        
        # 计算该维度的平均PCC和DTW
        if dim_pccs:
            avg_dim_pcc = np.mean(dim_pccs)
            pcc_by_dim.append(avg_dim_pcc)
        
        if dim_dtws:
            avg_dim_dtw = np.mean(dim_dtws)
            dtw_distances.append(avg_dim_dtw)
    
    # 计算所有维度的平均PCC
    if pcc_by_dim:
        pcc = np.mean(pcc_by_dim)
    else:
        pcc = 0.0
        
    # 计算所有维度的平均DTW距离
    if dtw_distances:
        dtw_dist = np.mean(dtw_distances)
    else:
        dtw_dist = float('inf')
        
    # 也计算全局扁平PCC（用于兼容之前的代码）
    all_generated_flat = all_generated_array.reshape(-1)
    all_targets_flat = all_targets_array.reshape(-1)
    
    # 过滤NaN值
    valid_indices = ~np.isnan(all_generated_flat) & ~np.isnan(all_targets_flat)
    if np.sum(valid_indices) > 0:
        flat_pcc = np.corrcoef(all_generated_flat[valid_indices], all_targets_flat[valid_indices])[0, 1]
    else:
        flat_pcc = 0.0
    
    # 打印评估结果
    print(f"评估结果 (归一化后):")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {avg_mae:.6f}")
    print(f"  序列PCC: {pcc:.6f}")
    print(f"  扁平PCC: {flat_pcc:.6f}")
    print(f"  DTW距离: {dtw_dist:.6f}")
    
    # 保存结果
    if output_dir:
        # 保存评估结果
        results_file = os.path.join(output_dir, "evaluation_results.txt")
        with open(results_file, "w") as f:
            f.write(f"评估结果 (归一化后):\n")
            f.write(f"  RMSE: {rmse:.6f}\n")
            f.write(f"  MAE: {avg_mae:.6f}\n")
            f.write(f"  序列PCC: {pcc:.6f}\n")
            f.write(f"  扁平PCC: {flat_pcc:.6f}\n")
            f.write(f"  DTW距离: {dtw_dist:.6f}\n")
            
            # 记录每个维度的PCC
            f.write(f"\n每个维度的PCC值:\n")
            for dim, dim_pcc in enumerate(pcc_by_dim):
                f.write(f"  维度 {dim}: {dim_pcc:.6f}\n")
                
            # 记录每个维度的DTW距离
            f.write(f"\n每个维度的DTW距离:\n")
            for dim, dim_dtw in enumerate(dtw_distances):
                f.write(f"  维度 {dim}: {dim_dtw:.6f}\n")
        
        # 保存5个样本的预测值和实际值
        samples_file = os.path.join(output_dir, "sample_predictions.txt")
        with open(samples_file, "w") as f:
            f.write("站点对预测值与真实值对比 (反归一化后的实际流量):\n\n")
            
            for idx, sample in enumerate(saved_samples):
                station_i, station_j = sample['station_pair']
                f.write(f"样本 {idx+1} - 站点对: ({station_i}, {station_j})\n")
                
                # i到j方向
                f.write(f"站点 {station_i} 到站点 {station_j} 的流量:\n")
                f.write("时间步\t预测值\t真实值\n")
                for t in range(len(sample['predicted'])):
                    f.write(f"{t+1}\t{sample['predicted'][t, 0]:.6f}\t{sample['real'][t, 0]:.6f}\n")
                
                # j到i方向
                f.write(f"\n站点 {station_j} 到站点 {station_i} 的流量:\n")
                f.write("时间步\t预测值\t真实值\n")
                for t in range(len(sample['predicted'])):
                    f.write(f"{t+1}\t{sample['predicted'][t, 1]:.6f}\t{sample['real'][t, 1]:.6f}\n")
                
                f.write("\n" + "="*50 + "\n\n")
        
        print(f"Results saved to {results_file}")
        print(f"Sample predictions saved to {samples_file}")
    
    return {
        'RMSE': rmse,
        'MAE': avg_mae,
        'PCC': pcc
    } 

# ========== 主函数 ==========
def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="OD流量预测 - Forest-GAN方案")
    
    # 数据参数 - 更新为52节点数据结构路径
    parser.add_argument("--io_flow_path", type=str, default="/private/od/data_NYTaxi/io_flow_daily.npy", help="IO流量数据路径")
    parser.add_argument("--graph_path", type=str, default="/private/od/data_NYTaxi/graph.npy", help="图结构数据路径")
    parser.add_argument("--od_matrix_path", type=str, default="/private/od/data_NYTaxi/od_matrix_daily.npy", help="OD矩阵数据路径")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, default="/private/od/Qwen2-7B", help="Qwen模型路径")
    parser.add_argument("--hidden_dim", type=int, default=128, help="隐藏层维度")
    parser.add_argument("--token_dim", type=int, default=768, help="令牌特征维度")
    parser.add_argument("--num_discriminators", type=int, default=3, help="判别器森林中判别器的数量")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=5e-5, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="权重衰减")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="测试集比例")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--seed", type=int, default=82, help="随机种子")
    parser.add_argument("--num_workers", type=int, default=0, help="数据加载线程数")
    parser.add_argument("--patience", type=int, default=10, help="早停耐心值")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="/private/od/paper_ny/ForestGAN", help="输出目录")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="运行模式：训练或测试")
    parser.add_argument("--load_model", type=str, default=None, help="加载模型路径（用于测试模式）")
    
    args = parser.parse_args()
    
    # 设置随机种子以确保可重复性
    set_seed(args.seed)
    
    # 创建输出目录
    output_dir = create_dynamic_output_dir(args.output_dir)
    args.output_dir = output_dir
    
    # 初始化日志记录器
    global logger, file_logger
    log_path = os.path.join(output_dir, "forest_gan.log")
    logger = logging.getLogger(__name__)
    file_logger = FileLogger(log_path)
    logger.info("=== OD流量Forest-GAN模型训练开始 ===")
    
    # 设置设备
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    print(f"使用设备: {device}")
    
    # 加载数据集
    print("加载数据集...")
    logger.info("加载数据集...")
    dataset = ODFlowDataset(
        io_flow_path=args.io_flow_path,
        graph_path=args.graph_path,
        od_matrix_path=args.od_matrix_path,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
    
    # 创建数据加载器
    if args.mode == "train":
        # 训练模式
        dataset.set_mode('train')
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        
        dataset.set_mode('val')
        val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
        # 🔧 修复：在训练模式下加载预计算的token特征
        precomputed_token_path = "/private/od/data_NYTaxi/token_features/precomputed_token_features_m18.pt"
        print(f"训练阶段加载预计算的token特征: {precomputed_token_path}")
        
        if not os.path.exists(precomputed_token_path):
            raise FileNotFoundError(f"预计算token特征文件不存在: {precomputed_token_path}")
        
        precomputed_token_features = torch.load(precomputed_token_path, map_location=device)
        print(f"训练阶段成功加载预计算token特征，包含 {len(precomputed_token_features)} 个站点对的特征")
        
        # 🔧 动态检测并更新token_dim参数
        if precomputed_token_features:
            sample_key = list(precomputed_token_features.keys())[0]
            actual_token_dim = precomputed_token_features[sample_key].shape[0]
            if args.token_dim != actual_token_dim:
                print(f"⚠️ 检测到token_dim不匹配: 参数设置={args.token_dim}, 实际特征维度={actual_token_dim}")
                print(f"✅ 自动更新token_dim: {args.token_dim} -> {actual_token_dim}")
                args.token_dim = actual_token_dim
        
        # 训练模型
        print("训练Forest-GAN模型...")
        logger.info("训练Forest-GAN模型...")
        best_model_path = train_llm_gan(args, precomputed_token_features)
        
        # 加载训练好的最佳模型
        print(f"加载最佳模型: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)
        
        # 创建模型
        model = ODFlowGenerator_GAN(
            hidden_dim=checkpoint.get('hidden_dim', args.hidden_dim),
            token_dim=checkpoint.get('token_dim', args.token_dim),
            time_steps=28,
            num_discriminators=checkpoint.get('num_discriminators', args.num_discriminators)
        ).to(device)
        
        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 打印模型信息
        print(f"Forest-GAN模型加载成功，Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"判别器数量: {checkpoint.get('num_discriminators', args.num_discriminators)}")
        print(f"RMSE: {checkpoint.get('rmse', 'N/A')}, MAE: {checkpoint.get('mae', 'N/A')}, PCC: {checkpoint.get('pcc', 'N/A')}")
    else:
        # 测试模式
        if args.load_model is None:
            print("错误：测试模式需要提供 --load_model 参数")
            return
        
        # 加载模型
        print(f"加载模型: {args.load_model}")
        checkpoint = torch.load(args.load_model, map_location=device)
        
        # 创建模型
        model = ODFlowGenerator_GAN(
            hidden_dim=checkpoint.get('hidden_dim', args.hidden_dim),
            token_dim=checkpoint.get('token_dim', args.token_dim),
            time_steps=28,
            num_discriminators=checkpoint.get('num_discriminators', args.num_discriminators)
        ).to(device)
        
        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 打印模型信息
        print(f"Forest-GAN模型加载成功，Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"判别器数量: {checkpoint.get('num_discriminators', args.num_discriminators)}")
        print(f"RMSE: {checkpoint.get('rmse', 'N/A')}, MAE: {checkpoint.get('mae', 'N/A')}, PCC: {checkpoint.get('pcc', 'N/A')}")
    
    # 测试模型
    dataset.set_mode('test')
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # 🔧 修改：加载预计算的token特征用于评估
    precomputed_token_path = "/private/od/data_NYTaxi/token_features/precomputed_token_features_m18.pt"
    if os.path.exists(precomputed_token_path):
        precomputed_token_features_eval = torch.load(precomputed_token_path, map_location=device)
        print(f"评估阶段加载预计算token特征，包含 {len(precomputed_token_features_eval)} 个站点对")
        
        # 🔧 动态检测并更新token_dim参数
        if precomputed_token_features_eval:
            sample_key = list(precomputed_token_features_eval.keys())[0]
            actual_token_dim = precomputed_token_features_eval[sample_key].shape[0]
            if args.token_dim != actual_token_dim:
                print(f"⚠️ 检测到token_dim不匹配: 参数设置={args.token_dim}, 实际特征维度={actual_token_dim}")
                print(f"✅ 自动更新token_dim: {args.token_dim} -> {actual_token_dim}")
                args.token_dim = actual_token_dim
    else:
        print(f"警告：预计算token特征文件不存在: {precomputed_token_path}")
        precomputed_token_features_eval = {}
    
    # 评估模型
    print("评估模型...")
    logger.info("评估模型...")
    metrics = evaluate_llm_gan_model(
        test_loader, 
        precomputed_token_features_eval, 
        model, 
        device, 
        args,
        output_dir=output_dir
    )
    
    # 创建小批次测试数据加载器，专门用于样本预测和可视化
    small_test_loader = DataLoader(dataset, batch_size=5, shuffle=False, num_workers=0)
    
    # 保存前5个样本的预测值和真实值
    print("保存前5个样本的预测值和真实值...")
    logger.info("保存前5个样本的预测值和真实值...")
    evaluate_llm_gan_model(
        small_test_loader, 
        precomputed_token_features_eval, 
        model, 
        device, 
        args,
        output_dir=output_dir
    )
    
    print("完成!")
    logger.info("=== OD流量Forest-GAN模型训练完成 ===")
    file_logger.close()

if __name__ == "__main__":
    main()