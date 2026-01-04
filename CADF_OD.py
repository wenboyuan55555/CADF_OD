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
        # 使用北京时区获取本地时间
        try:
            import datetime
            import pytz
            beijing_tz = pytz.timezone('Asia/Shanghai')
            now = datetime.datetime.now(beijing_tz)
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        except:
            # 备选方案：手动添加8小时偏移
            import datetime
            utc_now = datetime.datetime.utcnow()
            local_now = utc_now + datetime.timedelta(hours=8)
            timestamp = local_now.strftime("%Y-%m-%d %H:%M:%S")
        
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
    import pytz
    
    # 使用北京时区（UTC+8）来获取本地时间
    try:
        beijing_tz = pytz.timezone('Asia/Shanghai')
        now = datetime.datetime.now(beijing_tz)
        timestamp = now.strftime("%Y%m%d_%H%M%S")
    except ImportError:
        # 如果pytz不可用，手动添加8小时
        import datetime
        utc_now = datetime.datetime.utcnow()
        local_now = utc_now + datetime.timedelta(hours=8)
        timestamp = local_now.strftime("%Y%m%d_%H%M%S")
    except Exception as e:
        # 备选方案：使用系统本地时间
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"时区设置失败，使用系统时间: {e}")
    
    dynamic_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(dynamic_dir, exist_ok=True)
    return dynamic_dir

# ========== 序列模式检测工具 ==========
class SequencePatternDetector:
    """时间序列模式检测工具，检测周期性、趋势性和平稳性，以及按周分析变化趋势"""
    @staticmethod
    def detect_pattern(sequence, threshold_periodic=0.7, threshold_trend=0.5, threshold_areaary=0.3):
        """
        检测序列模式并返回模式类型和关键位置
        Args:
            sequence: 时间序列数据，形状(T,)
            threshold_periodic: 周期性判断阈值
            threshold_trend: 趋势性判断阈值
            threshold_areaary: 平稳性判断阈值
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
        areaary_score = 1 / (1 + cv)  # 变换为[0,1]范围，越高越平稳
        
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
        if areaary_score > threshold_areaary:
            pattern_types.append("areaary")
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
                "areaary": float(areaary_score),
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
    def generate_prompt(area_i, area_j, io_flow_i, io_flow_j, pattern_info_in_i, pattern_info_out_i, 
                       pattern_info_in_j, pattern_info_out_j, pop_density_i=None, pop_density_j=None):
        """
        生成用于大模型的提示文本，包含更丰富的时序特征描述
        Args:
            area_i: 站点i的ID
            area_j: 站点j的ID
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
                period_str = f"{period_lag} days" if period_lag > 0 else "irregular"
                seasonality = pattern_info["scores"]["seasonality"]
                peaks = pattern_info["peaks"]
                peak_str = ", ".join([f"day {p+1}" for p in peaks[:3]])
                if len(peaks) > 3:
                    peak_str += f"... total {len(peaks)} peaks"
                
                marker_parts.append(f"[Periodic sequence Period:{period_str} Seasonality:{seasonality:.2f} Major peaks:{peak_str}]")
                
            elif main_type == "trend":
                # 趋势性信息
                slope = pattern_info["trend"]["slope"]
                direction = pattern_info["trend"]["direction"]
                # 将方向翻译为英文，避免提示词中出现中文
                dir_map = {"上升": "increasing", "下降": "decreasing", "平稳": "stable"}
                direction_en = dir_map.get(direction, str(direction))
                trend_score = pattern_info["scores"]["trend"]
                
                marker_parts.append(f"[Trend sequence Direction:{direction_en} Slope:{slope:.4f} Trend strength:{trend_score:.2f}]")
                
            elif main_type == "areaary":
                # 平稳性信息
                mean = pattern_info["stats"]["mean"]
                std = pattern_info["stats"]["std"]
                cv = std / (abs(mean) + 1e-8)
                
                marker_parts.append(f"[Areaary sequence Mean:{mean:.2f} Std:{std:.2f} CV:{cv:.2f}]")
                
            else:
                # 混合型
                marker_parts.append("[Mixed sequence]")
                
            # 添加统计信息
            stats = pattern_info["stats"]
            marker_parts.append(f"[Statistics Mean:{stats['mean']:.2f} Median:{stats['median']:.2f} Amplitude:{stats['amplitude']:.2f}]")
            
            # 添加按周分析信息
            if "weekly_analysis" in pattern_info and pattern_info["weekly_analysis"]:
                weekly_trends = []
                
                # 分析每周同一天的变化
                for day_idx, day_info in pattern_info["weekly_analysis"].items():
                    if "trend" in day_info:
                        day_num = int(day_idx.split("_")[1])
                        trend_val = day_info["trend"]
                        trend_dir = "increasing" if trend_val > 0.05 else "decreasing" if trend_val < -0.05 else "stable"
                        weekly_trends.append(f"Week{day_num}:{trend_dir}({trend_val:.2f})")
                
                # 如果有周趋势数据，添加到标记中
                if weekly_trends:
                    marker_parts.append(f"[Weekly changes {' '.join(weekly_trends)}]")
            
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
            density_diff_desc = "significant" if density_ratio > 3 else "moderate" if density_ratio > 1.5 else "similar"
            
            pop_density_info = f"""
Area {area_i} nearby population density: {pop_density_i:.2f} people/sq.km
Area {area_j} nearby population density: {pop_density_j:.2f} people/sq.km
Average population density of the area pair: {density_avg:.2f} people/sq.km
Population density difference: {density_diff:.2f} ({density_diff_desc})
"""
        
        # 距离邻接矩阵信息（来自 /private/od/data_NYTaxi/graph.npy）
        try:
            graph_matrix = np.load("/private/od/data_NYTaxi/graph.npy")
            if area_i < graph_matrix.shape[0] and area_j < graph_matrix.shape[1]:
                distance_ij_val = float(graph_matrix[area_i, area_j])
                distance_ij_str = f"{distance_ij_val:.2f}"
                distance_row_i_str = np.array2string(graph_matrix[area_i], precision=2, separator=',')
                distance_row_j_str = np.array2string(graph_matrix[area_j], precision=2, separator=',')
            else:
                distance_ij_str = "N/A"
                distance_row_i_str = "N/A"
                distance_row_j_str = "N/A"
        except Exception:
            distance_ij_str = "N/A"
            distance_row_i_str = "N/A"
            distance_row_j_str = "N/A"
        
        # 生成提示文本
        prompt = f"""Area pair ({area_i},{area_j}) flow pattern analysis:
Time range: June 1 to June 28 (28 days, 4 full weeks)

-------- Area {area_i} Flow Characteristics --------
Inflow sequence: {inflow_i_str}
Outflow sequence: {outflow_i_str}
Net flow sequence: {net_flow_i_str}

Area {area_i} inflow pattern: {_get_detailed_marker(pattern_info_in_i, "inflow")}
Area {area_i} outflow pattern: {_get_detailed_marker(pattern_info_out_i, "outflow")}

-------- Area {area_j} Flow Characteristics --------
Inflow sequence: {inflow_j_str}
Outflow sequence: {outflow_j_str}
Net flow sequence: {net_flow_j_str}

Area {area_j} inflow pattern: {_get_detailed_marker(pattern_info_in_j, "inflow")}
Area {area_j} outflow pattern: {_get_detailed_marker(pattern_info_out_j, "outflow")}""" + (f"\n\n-------- Population Density --------\n{pop_density_info}" if pop_density_info else "") + f"""

-------- Distance Adjacency Matrix (/private/od/data_NYTaxi/graph.npy) --------
Note: smaller values mean closer, larger values mean farther; 0 indicates no direct connection; shape (52, 52)
Distance between stations D({area_i},{area_j}): {distance_ij_str}
Distance vector from station {area_i} to all stations (length 52): {distance_row_i_str}
Distance vector from station {area_j} to all stations (length 52): {distance_row_j_str}

-------- Inter-Area Flow Relationship --------
Flow from area {area_i} to area {area_j} represents traffic demand from origin {area_i} to destination {area_j}
Flow from area {area_j} to area {area_i} represents traffic demand from origin {area_j} to destination {area_i}

Based on the above flow patterns, weekly trend changes, population density information, and the distance adjacency matrix, please analyze in detail the 28-day OD flow characteristics and patterns from area {area_i} to area {area_j} and from area {area_j} to area {area_i}, with special attention to differences between weekdays and weekends, and trend changes on the same weekday across weeks, as well as the impact of spatial distance on flows.
"""
        return prompt

    @staticmethod
    def generate_prompt_nyc(region_i, region_j, io_flow_i, io_flow_j, pattern_info_in_i, pattern_info_out_i, 
                           pattern_info_in_j, pattern_info_out_j, pop_density_i=None, pop_density_j=None):
        """
        生成用于纽约出租车网格OD数据的提示文本
        Args:
            region_i: 区域i的ID
            region_j: 区域j的ID
            io_flow_i: 区域i的IO流量, 形状(T, 2)
            io_flow_j: 区域j的IO流量, 形状(T, 2)
            pattern_info_*: 四个序列的模式信息字典
            pop_density_i: 区域i的人口密度
            pop_density_j: 区域j的人口密度
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
                period_str = f"{period_lag} days" if period_lag > 0 else "irregular"
                seasonality = pattern_info["scores"]["seasonality"]
                peaks = pattern_info["peaks"]
                peak_str = ", ".join([f"day {p+1}" for p in peaks[:3]])
                if len(peaks) > 3:
                    peak_str += f"... total {len(peaks)} peaks"
                
                marker_parts.append(f"[Periodic sequence Period:{period_str} Seasonality:{seasonality:.2f} Major peaks:{peak_str}]")
                
            elif main_type == "trend":
                # 趋势性信息
                slope = pattern_info["trend"]["slope"]
                direction = pattern_info["trend"]["direction"]
                # 将方向翻译为英文，避免提示词中出现中文
                dir_map = {"上升": "increasing", "下降": "decreasing", "平稳": "stable"}
                direction_en = dir_map.get(direction, str(direction))
                trend_score = pattern_info["scores"]["trend"]
                
                marker_parts.append(f"[Trend sequence Direction:{direction_en} Slope:{slope:.4f} Trend strength:{trend_score:.2f}]")
                
            elif main_type == "areaary":
                # 平稳性信息
                mean = pattern_info["stats"]["mean"]
                std = pattern_info["stats"]["std"]
                cv = std / (abs(mean) + 1e-8)
                
                marker_parts.append(f"[Areaary sequence Mean:{mean:.2f} Std:{std:.2f} CV:{cv:.2f}]")
                
            else:
                # 混合型
                marker_parts.append("[Mixed sequence]")
                
            # 添加统计信息
            stats = pattern_info["stats"]
            marker_parts.append(f"[Statistics Mean:{stats['mean']:.2f} Median:{stats['median']:.2f} Amplitude:{stats['amplitude']:.2f}]")
            
            # 添加按周分析信息
            if "weekly_analysis" in pattern_info and pattern_info["weekly_analysis"]:
                weekly_trends = []
                
                # 分析每周同一天的变化
                for day_idx, day_info in pattern_info["weekly_analysis"].items():
                    if "trend" in day_info:
                        day_num = int(day_idx.split("_")[1])
                        trend_val = day_info["trend"]
                        trend_dir = "increasing" if trend_val > 0.05 else "decreasing" if trend_val < -0.05 else "stable"
                        weekly_trends.append(f"Week{day_num}:{trend_dir}({trend_val:.2f})")
                
                # 如果有周趋势数据，添加到标记中
                if weekly_trends:
                    marker_parts.append(f"[Weekly changes {' '.join(weekly_trends)}]")
            
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
            density_diff_desc = "significant" if density_ratio > 3 else "moderate" if density_ratio > 1.5 else "similar"
            
            pop_density_info = f"""
Region {region_i} population density: {pop_density_i:.2f} people/sq.km
Region {region_j} population density: {pop_density_j:.2f} people/sq.km
Average population density of the region pair: {density_avg:.2f} people/sq.km
Population density difference: {density_diff:.2f} ({density_diff_desc})
"""
        
        # 距离邻接矩阵信息（来自 /private/od/data_NYTaxi/graph.npy）
        try:
            graph_matrix = np.load("/private/od/data_NYTaxi/graph.npy")
            if region_i < graph_matrix.shape[0] and region_j < graph_matrix.shape[1]:
                distance_ij_val = float(graph_matrix[region_i, region_j])
                distance_ij_str = f"{distance_ij_val:.2f}"
                distance_row_i_str = np.array2string(graph_matrix[region_i], precision=2, separator=',')
                distance_row_j_str = np.array2string(graph_matrix[region_j], precision=2, separator=',')
            else:
                distance_ij_str = "N/A"
                distance_row_i_str = "N/A"
                distance_row_j_str = "N/A"
        except Exception:
            distance_ij_str = "N/A"
            distance_row_i_str = "N/A"
            distance_row_j_str = "N/A"
        
        # 生成纽约出租车专用提示文本
        prompt = f"""NYC taxi grid region pair ({region_i},{region_j}) flow pattern analysis:
Time range: 28 consecutive days (4 full weeks)
Data source: NYC taxi trip records aggregated by grid regions

-------- Region {region_i} Flow Characteristics --------
Taxi pickup flow (inflow) sequence: {inflow_i_str}
Taxi dropoff flow (outflow) sequence: {outflow_i_str}
Net taxi flow sequence: {net_flow_i_str}

Region {region_i} pickup flow pattern: {_get_detailed_marker(pattern_info_in_i, "pickup")}
Region {region_i} dropoff flow pattern: {_get_detailed_marker(pattern_info_out_i, "dropoff")}

-------- Region {region_j} Flow Characteristics --------
Taxi pickup flow (inflow) sequence: {inflow_j_str}
Taxi dropoff flow (outflow) sequence: {outflow_j_str}
Net taxi flow sequence: {net_flow_j_str}

Region {region_j} pickup flow pattern: {_get_detailed_marker(pattern_info_in_j, "pickup")}
Region {region_j} dropoff flow pattern: {_get_detailed_marker(pattern_info_out_j, "dropoff")}""" + (f"\n\n-------- Population Density --------\n{pop_density_info}" if pop_density_info else "") + f"""

-------- Distance Adjacency Matrix (/private/od/data_NYTaxi/graph.npy) --------
Note: spatial distance between NYC grid regions; smaller values mean closer, larger values mean farther; shape (52, 52)
Distance between regions D({region_i},{region_j}): {distance_ij_str}
Distance vector from region {region_i} to all regions (length 52): {distance_row_i_str}
Distance vector from region {region_j} to all regions (length 52): {distance_row_j_str}

-------- Inter-Region Taxi Flow Relationship --------
Flow from region {region_i} to region {region_j} represents taxi trip demand from origin region {region_i} to destination region {region_j}
Flow from region {region_j} to region {region_i} represents taxi trip demand from origin region {region_j} to destination region {region_i}

Based on the above taxi flow patterns, weekly trend changes, population density information, and the spatial distance matrix, please analyze in detail the 28-day taxi OD flow characteristics and patterns from region {region_i} to region {region_j} and from region {region_j} to region {region_i}, with special attention to differences between weekdays and weekends, and trend changes on the same weekday across weeks, as well as the impact of spatial distance on taxi demand flows in NYC.
"""
        return prompt

# ========== Qwen2特征提取器 ==========
class QwenFeatureExtractor(nn.Module):
    """使用Qwen2模型提取特征的模块"""
    def __init__(self, api_key=None, feature_dim=768, device="cuda:0"):
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
    
    def forward(self, area_i, area_j, io_flow_i, io_flow_j, area_data=None, prompt_type="beijing"):
        """
        前向传播，处理站点对数据并提取特征
        Args:
            area_i, area_j: 站点ID
            io_flow_i, io_flow_j: 站点IO流量数据, 形状(T, 2)
            area_data: 站点人口密度数据列表
            prompt_type: 提示词类型，"beijing"为北京地铁数据，"nyc"为纽约出租车数据
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
        if area_data is not None and len(area_data) > 0:
            # 确保站点索引不超过可用的站点数据
            if area_i < len(area_data) and area_j < len(area_data):
                pop_density_i = area_data[area_i].get('grid_population_density', 0.0)
                pop_density_j = area_data[area_j].get('grid_population_density', 0.0)
        
        # 生成提示文本，根据prompt_type选择不同的模板
        if prompt_type == "nyc":
            prompt = SequencePatternDetector.generate_prompt_nyc(
                area_i, area_j, 
                io_flow_i.cpu().numpy(), io_flow_j.cpu().numpy(),
                pattern_info_in_i, pattern_info_out_i, 
                pattern_info_in_j, pattern_info_out_j,
                pop_density_i, pop_density_j
            )
        else:  # 默认使用北京地铁数据的提示词
            prompt = SequencePatternDetector.generate_prompt(
                area_i, area_j, 
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
            H_T: 时间序列分支输出的弱特征 [batch_size, seq_dim] 或 [batch_size, time_steps, seq_dim]
            L_N: 提示分支输出的强鲁棒特征 [batch_size, token_dim]
        Returns:
            H_C: 增强后的时序特征，与输入维度相同
            M_T: 相似度矩阵 [batch_size, 1]
        """
        # 保存原始形状和维度信息
        original_shape = H_T.shape
        original_dim = len(original_shape)
        
        # 对于3D输入，先记录时间步维度，然后压缩为2D进行处理
        if original_dim == 3:  # [batch_size, time_steps, seq_dim]
            batch_size, time_steps, seq_dim = original_shape
            # 将时间步维度与批次维度合并
            H_T_flat = H_T.reshape(-1, seq_dim)  # [batch_size*time_steps, seq_dim]
            # 对每个时间步都使用相同的令牌特征
            L_N_expanded = L_N.unsqueeze(1).expand(-1, time_steps, -1)  # [batch_size, time_steps, token_dim]
            L_N_flat = L_N_expanded.reshape(-1, L_N.shape[-1])  # [batch_size*time_steps, token_dim]
            
            # print(f"CrossModalityAlignment: 处理3D输入, 原始形状: {original_shape}")
            # 在2D上处理
            H_C_flat, M_T_flat = self._forward_2d(H_T_flat, L_N_flat)
            
            # 恢复原始形状
            H_C = H_C_flat.reshape(batch_size, time_steps, seq_dim)
            M_T = M_T_flat.reshape(batch_size, time_steps)
            
            return H_C, M_T
        else:  # 2D输入 [batch_size, seq_dim]
            return self._forward_2d(H_T, L_N)
    
    def _forward_2d(self, H_T, L_N):
        """
        对2D输入的前向传播实现
        Args:
            H_T: 时间序列分支输出的弱特征 [batch_size, seq_dim]
            L_N: 提示分支输出的强鲁棒特征 [batch_size, token_dim]
        Returns:
            H_C: 增强后的时序特征 [batch_size, seq_dim]
            M_T: 相似度矩阵 [batch_size, 1]
        """
        batch_size = H_T.shape[0]
        
        if len(L_N.shape) == 1:  # 如果输入是[token_dim]
            # 扩展为批次
            L_N = L_N.unsqueeze(0).expand(batch_size, -1)  # [batch_size, token_dim]
        
        # 处理token_dim不匹配的情况
        if L_N.shape[-1] != self.token_dim:
            print(f"跨模态对齐模块调整令牌特征维度: 从 {L_N.shape[-1]} 到 {self.token_dim}")
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

# ========== 时空联合注意力机制 ==========
class SpatioTemporalAttention(nn.Module):
    """时空联合注意力机制，同时建模时间和空间维度上的依赖关系"""
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 多头自注意力
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1
        )
        
        # 时间位置编码
        self.temporal_pe = PositionalEncoding(hidden_dim, max_len=50)
        
        # 空间位置编码处理
        self.spatial_proj = nn.Linear(hidden_dim, hidden_dim)
        
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
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, spatial_encoding=None):
        """
        前向传播
        Args:
            x: 输入特征 [batch_size, time_steps, hidden_dim] 或 [batch_size, hidden_dim]
            spatial_encoding: 空间编码 [batch_size, hidden_dim]
        Returns:
            output: 注意力输出 [batch_size, time_steps, hidden_dim] 或 [batch_size, hidden_dim]
        """
        # 检查x的维度，如果是2维，扩展为3维
        is_2d_input = len(x.shape) == 2
        if is_2d_input:
            batch_size, hidden_dim = x.shape
            # 扩展为 [batch_size, 1, hidden_dim]
            x = x.unsqueeze(1)
            print(f"SpatioTemporalAttention: 输入维度为2D，已扩展为3D: {x.shape}")
        
        batch_size, time_steps, _ = x.shape
        
        # 添加时间位置编码
        x = self.temporal_pe(x)
        
        # 添加空间编码（如果提供）
        if spatial_encoding is not None:
            # 空间编码 [batch_size, hidden_dim]
            # 处理空间编码
            spatial_encoding = self.spatial_proj(spatial_encoding)
            # 扩展到时间维度
            spatial_encoding = spatial_encoding.unsqueeze(1).expand(-1, time_steps, -1)
            x = x + spatial_encoding
        
        # 转换形状以适应多头注意力
        x_orig = x
        x = x.transpose(0, 1)  # [time_steps, batch_size, hidden_dim]
        
        # 应用时空联合注意力
        attn_out, _ = self.mha(x, x, x)
        
        # 恢复形状
        attn_out = attn_out.transpose(0, 1)  # [batch_size, time_steps, hidden_dim]
        
        # 残差连接和层归一化
        out1 = self.norm1(x_orig + self.dropout(attn_out))
        
        # 前馈网络
        ffn_out = self.ffn(out1)
        
        # 再次残差连接和层归一化
        output = self.norm2(out1 + self.dropout(ffn_out))
        
        # 如果输入是2D，将输出也转换回2D
        if is_2d_input:
            output = output.squeeze(1)
        
        return output

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
        
        # 编码输入特征
        encoded_features = self.feature_encoder(features)  # 时序特征 [batch_size, time_steps, hidden_dim]
        
        # 提取空间特征 - 使用所有时间步的平均值
        spatial_features = self.spatial_encoder(torch.mean(features, dim=1))  # [batch_size, hidden_dim]
        
        # 时序特征与令牌特征交互
        if token_features is not None:
            # 检查并适配token_features的维度
            if len(token_features.shape) == 2 and token_features.shape[1] != self.token_dim:
                # print(f"生成器调整令牌特征维度: 从 {token_features.shape[1]} 到 {self.token_dim}")
                # 创建一个临时投影层
                temp_projection = nn.Linear(token_features.shape[1], self.token_dim).to(token_features.device)
                token_features = temp_projection(token_features)
                
            # 应用跨模态对齐 - 使用修改后的支持3D输入的版本
            enhanced_features, _ = self.cross_modal_alignment(encoded_features, token_features)
            
            # 应用时空联合注意力机制
            try:
                # 空间特征作为条件输入，增强时间序列特征
                st_enhanced = self.spatiotemporal_attention(enhanced_features, spatial_features)
                print("生成器时空联合注意力机制应用成功")
            except Exception as e:
                print(f"生成器时空联合注意力出错: {str(e)}")
                # 失败时使用原始增强特征
                st_enhanced = enhanced_features
            
            # 与噪声融合
            x = torch.cat([st_enhanced, noise], dim=-1)
            x = self.fusion_layer(x)
        else:
            # 如果没有令牌特征，仍然应用时空联合注意力
            try:
                # 应用时空联合注意力 - 只使用空间编码
                st_enhanced = self.spatiotemporal_attention(encoded_features, spatial_features)
                
                # 与噪声融合
                x = torch.cat([st_enhanced, noise], dim=-1)
                x = self.fusion_layer(x)
            except Exception as e:
                print(f"生成器时空联合注意力出错 (无令牌特征): {str(e)}")
                # 失败时直接使用原始特征与噪声融合
                x = torch.cat([encoded_features, noise], dim=-1)
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
        
        # 添加时空联合注意力机制
        self.spatiotemporal_attention = SpatioTemporalAttention(
            hidden_dim=hidden_dim,
            num_heads=8
        )
        
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

        # 数据归一化
        features = self.feature_norm(features)
        od_flows = self.flow_norm(od_flows)
        
        # 合并特征和OD流量
        combined_input = torch.cat([features, od_flows], dim=-1)
        
        # 编码特征
        encoded_features = self.feature_encoder(combined_input)
        
        # 处理令牌特征（如果有）
        if token_features is not None:
            # 检查并适配token_features的维度
            if len(token_features.shape) == 2 and token_features.shape[1] != self.token_dim:
                # print(f"判别器调整令牌特征维度: 从 {token_features.shape[1]} 到 {self.token_dim}")
                # 创建一个临时投影层
                temp_projection = nn.Linear(token_features.shape[1], self.token_dim).to(token_features.device)
                token_features = temp_projection(token_features)
            
            # 使用跨模态对齐模块增强时序特征
            # 注意: CrossModalityAlignment已修改为支持3D输入，不需要遍历时间步
            enhanced_features, _ = self.cross_modal_alignment(encoded_features, token_features)
            encoded_features = enhanced_features
        
        # 使用空间编码器处理features
        # 原来使用的是第一个时间步的features，现在改为使用所有时间步并取平均值
        spatial_features = self.spatial_encoder(torch.mean(features, dim=1))  # [batch_size, hidden_dim]
        
        # 应用时空联合注意力机制
        attended_features = self.spatiotemporal_attention(encoded_features, spatial_features)
        
        # 融合处理
        fused_features = self.fusion_layer(attended_features)
        
        # 添加位置编码并应用Informer编码器
        fused_features = self.position_encoding(fused_features)
        encoded_output = self.encoder(fused_features)
        
        # 应用时间注意力机制
        temporal_weighted = self.temporal_attention(encoded_output)
        
        # 取时间维度上的平均作为最终特征表示
        pooled_features = torch.mean(temporal_weighted, dim=1)
        
        # 输出层
        output = self.output_layer(pooled_features)
        
        return output

# ========== 判别器森林(Forest-GAN) ==========
"""
Forest-GAN核心原理:

1. 问题背景
   - 传统GAN中，单一判别器容易过拟合，导致生成器训练不稳定和崩溃
   - 高容量判别器虽然能提高生成质量，但容易过度拟合真实数据分布

2. 解决方案
   - 借鉴随机森林(Random Forest)的思想，使用多个独立判别器组成"判别森林"
   - 每个判别器在自举(Bootstrap)采样的数据集上独立训练，增加鲁棒性
   - 生成器同时对抗多个判别器，学习更一致和稳定的特征

3. 实现原理
   - 自举采样: 对原始数据随机重采样，使每个判别器看到略有不同的数据分布
   - 独立训练: 每个判别器有独立参数，避免判别器集体过拟合相同模式
   - 平均梯度: 生成器训练时聚合所有判别器的梯度信息，提高稳定性

4. 优势
   - 提高生成数据的多样性，避免模式崩溃
   - 减少训练不稳定性，更容易收敛到平衡点
   - 增强对噪声和异常样本的鲁棒性
"""
class ODFlowGANDiscriminatorForest(nn.Module):
    """Forest-GAN判别器森林 - 采用Stacking机制融合多个基础判别器的结果"""
    def __init__(self, feature_dim=6, hidden_dim=128, token_dim=768, time_steps=28, input_dim=2, n_discriminators=3):
        """
        初始化基于Stacking机制的判别器森林 - 简化版本，专注于时序和空间特征
        Args:
            feature_dim: 特征维度
            hidden_dim: 隐藏层维度
            token_dim: 令牌特征维度
            time_steps: 时间步数
            input_dim: 输入维度
            n_discriminators: 基础判别器数量（默认3个）
        """
        super().__init__()
        self.n_discriminators = n_discriminators
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.token_dim = token_dim
        self.time_steps = time_steps
        
        # 创建专注于时序和空间特征的3个判别器（第一层）
        self.base_discriminators = nn.ModuleList([
            # 时间敏感判别器 - 专注于时序模式和时间依赖关系
            ODFlowGANDiscriminatorTemporal(
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                token_dim=token_dim,
                time_steps=time_steps,
                input_dim=input_dim
            ),
            # 空间敏感判别器 - 专注于空间关系和站点间交互
            ODFlowGANDiscriminatorSpatial(
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                token_dim=token_dim,
                time_steps=time_steps,
                input_dim=input_dim
            ),
            # 基础判别器 - 作为通用判别器，平衡时空特征
            ODFlowGANDiscriminator(
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                token_dim=token_dim,
                time_steps=time_steps,
                input_dim=input_dim
            )
        ])
        
        # 如果需要更多判别器，添加基础判别器的变体
        if n_discriminators > 3:
            for _ in range(n_discriminators - 3):
                self.base_discriminators.append(
                    ODFlowGANDiscriminator(
                        feature_dim=feature_dim,
                        hidden_dim=hidden_dim,
                        token_dim=token_dim,
                        time_steps=time_steps,
                        input_dim=input_dim
                    )
                )
        
        # 元判别器（第二层）- 用于融合基础判别器的结果
        # 元判别器输入包括: 基础判别器输出 + 原始特征的关键信息 + token特征
        self.token_output_dim = token_dim//8  # 压缩后的token维度
        meta_input_dim = n_discriminators + feature_dim + self.token_output_dim
        
        self.meta_discriminator = nn.Sequential(
            nn.Linear(meta_input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim//2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//2, 1)
        )
        
        # 特征压缩层 - 将原始特征压缩为低维表示，提供给元判别器
        self.feature_compressor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(feature_dim)
        )
        
        # Token特征压缩层 - 在初始化时不创建，而是在forward时根据实际维度创建
        self.token_compressor = None
        self.token_output_dim = token_dim//8  # 压缩后的输出维度
        
        # 注意力权重生成器 - 生成每个基础判别器的动态权重
        self.attention_generator = nn.Sequential(
            nn.Linear(n_discriminators, n_discriminators*2),
            nn.LeakyReLU(0.2),
            nn.Linear(n_discriminators*2, n_discriminators),
            nn.Softmax(dim=1)  # 确保权重和为1
        )

    def bootstrap_batch(self, features, od_flows, token_features=None):
        """
        对批次数据进行自举采样 - 增强版，支持不同采样策略
        Args:
            features: 输入特征 [batch_size, time_steps, feature_dim]
            od_flows: OD流量 [batch_size, time_steps, input_dim]
            token_features: 令牌特征 [batch_size, token_dim]
        Returns:
            bootstrap_features: 自举采样后的特征
            bootstrap_od_flows: 自举采样后的OD流量
            bootstrap_token_features: 自举采样后的令牌特征
        """
        batch_size = features.size(0)
        
        # 基础自举采样策略：有放回地抽样，生成索引
        indices = torch.randint(0, batch_size, (batch_size,), device=features.device)
        
        # 使用索引抽取样本
        bootstrap_features = features[indices]
        bootstrap_od_flows = od_flows[indices]
        
        # 如果有令牌特征，也进行采样
        if token_features is not None:
            bootstrap_token_features = token_features[indices]
            return bootstrap_features, bootstrap_od_flows, bootstrap_token_features
        
        return bootstrap_features, bootstrap_od_flows, None
    
    def forward(self, features, od_flows, token_features=None, use_bootstrap=True):
        """
        前向传播 - 采用Stacking机制的判别器森林
        Args:
            features: 输入特征 [batch_size, time_steps, feature_dim]
            od_flows: OD流量 [batch_size, time_steps, input_dim]
            token_features: 令牌特征 [batch_size, token_dim]
            use_bootstrap: 是否使用自举采样
        Returns:
            all_base_scores: 各基础判别器的分数列表
            meta_score: 元判别器的最终分数 [batch_size, 1]
            weighted_avg_score: 加权平均分数 [batch_size, 1]
        """
        batch_size = features.shape[0]
        all_base_scores = []
        
        # 第一层：基础判别器层
        for i, discriminator in enumerate(self.base_discriminators):
            if use_bootstrap:
                # 为每个判别器创建不同的自举样本
                bootstrap_features, bootstrap_od_flows, bootstrap_token_features = self.bootstrap_batch(
                    features, od_flows, token_features
                )
                # 使用自举样本获取分数
                score = discriminator(bootstrap_features, bootstrap_od_flows, bootstrap_token_features)
            else:
                # 在推理时不使用自举采样
                score = discriminator(features, od_flows, token_features)
                
            all_base_scores.append(score)
        
        # 将所有基础判别器的分数拼接成一个张量
        # [n_discriminators, batch_size, 1] -> [batch_size, n_discriminators]
        base_scores_tensor = torch.cat([score.view(batch_size, 1) for score in all_base_scores], dim=1)
        
        # 生成基础判别器的注意力权重
        attention_weights = self.attention_generator(base_scores_tensor)
        
        # 计算加权平均分数作为传统方法的输出
        weighted_avg_score = torch.sum(base_scores_tensor * attention_weights, dim=1, keepdim=True)
        
        # 计算加权后的基础分数，用于元判别器输入
        weighted_base_scores_tensor = base_scores_tensor * attention_weights

        # 第二层：元判别器层
        # 1. 压缩原始特征
        # 对时间维度取平均，得到每个样本的特征概要
        avg_features = torch.mean(features, dim=1)  # [batch_size, feature_dim]
        compressed_features = self.feature_compressor(avg_features)  # [batch_size, feature_dim]
        
        # 2. 压缩token特征（如果有）
        if token_features is not None:
            # 检查token特征的维度，并根据需要动态创建压缩器
            actual_token_dim = token_features.shape[1]
            if self.token_compressor is None or self.token_compressor[0].in_features != actual_token_dim:
                print(f"动态创建token压缩器: 输入维度 {actual_token_dim} -> 输出维度 {self.token_output_dim}")
                self.token_compressor = nn.Sequential(
                    nn.Linear(actual_token_dim, self.token_output_dim).to(token_features.device),
                    nn.LeakyReLU(0.2),
                    nn.LayerNorm(self.token_output_dim)
                ).to(token_features.device)
            
            compressed_tokens = self.token_compressor(token_features)  # [batch_size, token_output_dim]
        else:
            # 如果没有token特征，创建全零张量
            compressed_tokens = torch.zeros(batch_size, self.token_output_dim, device=features.device)
        
        # 3. 构建元判别器的输入
        try:
            meta_input = torch.cat([
                weighted_base_scores_tensor, # 使用加权后的基础分数作为元判别器输入
                compressed_features,      # 压缩后的特征 [batch_size, feature_dim]
                compressed_tokens         # 压缩后的token特征 [batch_size, self.token_output_dim]
            ], dim=1)
        except RuntimeError as e:
            print(f"构建元判别器输入时出错: {e}")
            print(f"weighted_base_scores_tensor形状: {weighted_base_scores_tensor.shape}")
            print(f"compressed_features形状: {compressed_features.shape}")
            print(f"compressed_tokens形状: {compressed_tokens.shape}")
            print(f"期望的meta_input维度: {self.n_discriminators + self.feature_dim + self.token_output_dim}")
            # 尝试重建meta_discriminator以适应实际维度
            actual_meta_input_dim = weighted_base_scores_tensor.shape[1] + compressed_features.shape[1] + compressed_tokens.shape[1]
            print(f"重建元判别器: 输入维度从 {self.n_discriminators + self.feature_dim + self.token_output_dim} 调整为 {actual_meta_input_dim}")
            self.meta_discriminator = nn.Sequential(
                nn.Linear(actual_meta_input_dim, self.hidden_dim),
                nn.LeakyReLU(0.2),
                nn.LayerNorm(self.hidden_dim),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim, self.hidden_dim//2),
                nn.LeakyReLU(0.2),
                nn.LayerNorm(self.hidden_dim//2),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim//2, 1)
            ).to(base_scores_tensor.device)
            meta_input = torch.cat([weighted_base_scores_tensor, compressed_features, compressed_tokens], dim=1)
        
        # 4. 元判别器前向传播
        meta_score = self.meta_discriminator(meta_input)
        
        return all_base_scores, meta_score, weighted_avg_score

# 专门针对时序模式的判别器
class ODFlowGANDiscriminatorTemporal(ODFlowGANDiscriminator):
    """时间敏感判别器 - 增强时间特征提取能力"""
    def __init__(self, feature_dim=6, hidden_dim=128, token_dim=768, time_steps=28, input_dim=2):
        super().__init__(feature_dim, hidden_dim, token_dim, time_steps, input_dim)
        
        # 增强型时间注意力机制
        self.enhanced_temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim*2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim*2, hidden_dim)
        )
        
        # 时间卷积网络 - 捕获多尺度时间模式
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.LayerNorm([hidden_dim, time_steps]),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),
            nn.LayerNorm([hidden_dim, time_steps])
        )
    
    def forward(self, features, od_flows, token_features=None):
        # 使用基类的部分处理流程
        batch_size, time_steps, _ = features.shape
        
        # 数据归一化
        features = self.feature_norm(features)
        od_flows = self.flow_norm(od_flows)
        
        # 合并特征和OD流量
        combined_input = torch.cat([features, od_flows], dim=-1)
        
        # 编码特征
        encoded_features = self.feature_encoder(combined_input)
        
        # 处理令牌特征（如果有）
        if token_features is not None:
            # 检查并适配token_features的维度
            if len(token_features.shape) == 2 and token_features.shape[1] != self.token_dim:
                # 创建一个临时投影层
                temp_projection = nn.Linear(token_features.shape[1], self.token_dim).to(token_features.device)
                token_features = temp_projection(token_features)
            
            # 使用跨模态对齐模块增强时序特征
            enhanced_features, _ = self.cross_modal_alignment(encoded_features, token_features)
            encoded_features = enhanced_features
        
        # 应用时间卷积网络增强时间特征
        # [batch_size, time_steps, hidden_dim] -> [batch_size, hidden_dim, time_steps]
        temporal_features = encoded_features.transpose(1, 2)
        temporal_features = self.temporal_conv(temporal_features)
        # [batch_size, hidden_dim, time_steps] -> [batch_size, time_steps, hidden_dim]
        temporal_features = temporal_features.transpose(1, 2)
        
        # 融合增强型时间注意力
        temporal_attention = self.temporal_attention(temporal_features)
        enhanced_temporal = self.enhanced_temporal_attention(temporal_attention)
        
        # 添加位置编码并应用Informer编码器
        encoded_output = self.position_encoding(enhanced_temporal)
        encoded_output = self.encoder(encoded_output)
        
        # 取时间维度上的加权平均作为最终特征表示
        temporal_weights = F.softmax(torch.sum(encoded_output, dim=2), dim=1).unsqueeze(2)
        weighted_features = encoded_output * temporal_weights
        pooled_features = torch.sum(weighted_features, dim=1)
        
        # 输出层
        output = self.output_layer(pooled_features)
        
        return output


# 专门针对空间模式的判别器
class ODFlowGANDiscriminatorSpatial(ODFlowGANDiscriminator):
    """空间敏感判别器 - 增强空间特征提取能力"""
    def __init__(self, feature_dim=6, hidden_dim=128, token_dim=768, time_steps=28, input_dim=2):
        super().__init__(feature_dim, hidden_dim, token_dim, time_steps, input_dim)
        
        # 增强型空间编码器
        self.enhanced_spatial_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim*2),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim*2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim*2, hidden_dim)
        )
        
        # 自注意力机制处理空间关系
        self.spatial_self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1
        )
        
        # 空间投影层
        self.spatial_projection = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, features, od_flows, token_features=None):
        # 使用基类的部分处理流程
        batch_size, time_steps, _ = features.shape
        
        # 数据归一化
        features = self.feature_norm(features)
        od_flows = self.flow_norm(od_flows)
        
        # 空间特征增强
        spatial_features = self.enhanced_spatial_encoder(features)
        
        # 合并特征和OD流量
        combined_input = torch.cat([features, od_flows], dim=-1)
        
        # 编码特征
        encoded_features = self.feature_encoder(combined_input)
        
        # 处理令牌特征（如果有）
        if token_features is not None:
            # 检查并适配token_features的维度
            if len(token_features.shape) == 2 and token_features.shape[1] != self.token_dim:
                # 创建一个临时投影层
                temp_projection = nn.Linear(token_features.shape[1], self.token_dim).to(token_features.device)
                token_features = temp_projection(token_features)
            
            # 使用跨模态对齐模块增强时序特征
            enhanced_features, _ = self.cross_modal_alignment(encoded_features, token_features)
            encoded_features = enhanced_features
        
        # 应用空间自注意力
        # [batch_size, time_steps, hidden_dim] -> [time_steps, batch_size, hidden_dim]
        spatial_features = spatial_features.transpose(0, 1)
        attn_output, _ = self.spatial_self_attention(
            spatial_features, spatial_features, spatial_features
        )
        # [time_steps, batch_size, hidden_dim] -> [batch_size, time_steps, hidden_dim]
        attn_output = attn_output.transpose(0, 1)
        
        # 融合空间和时间特征
        fused_features = torch.cat([encoded_features, attn_output], dim=2)
        fused_features = self.spatial_projection(fused_features)
        
        # 应用位置编码和Informer编码器
        encoded_output = self.position_encoding(fused_features)
        encoded_output = self.encoder(encoded_output)
        
        # 时间注意力并池化
        temporal_weighted = self.temporal_attention(encoded_output)
        pooled_features = torch.mean(temporal_weighted, dim=1)
        
        # 输出层
        output = self.output_layer(pooled_features)
        
        return output


# 更深层的判别器，专注于复杂时序模式
class ODFlowGANDiscriminatorDeep(ODFlowGANDiscriminator):
    """深度增强判别器 - 使用更深的网络结构捕获复杂模式"""
    def __init__(self, feature_dim=6, hidden_dim=128, token_dim=768, time_steps=28, input_dim=2, e_layers=3):
        super().__init__(feature_dim, hidden_dim, token_dim, time_steps, input_dim)
        
        # 替换为更深层的Informer
        self.encoder = Informer(
            enc_in=hidden_dim,      # 输入维度
            d_model=hidden_dim,     # 模型维度
            c_out=hidden_dim,       # 输出维度
            factor=5,               # 注意力稀疏因子
            n_heads=8,              # 注意力头数量
            e_layers=e_layers,      # 更多的编码器层
            d_ff=hidden_dim*4,      # 前馈网络维度
            dropout=0.1,            # 丢弃率
            activation='gelu'       # 激活函数
        )
        
        # 残差连接层
        self.residual_layer = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim)
        )
        
        # 添加跳跃连接
        self.skip_connection = True
    
    def forward(self, features, od_flows, token_features=None):
        batch_size, time_steps, _ = features.shape
        
        # 数据归一化
        features = self.feature_norm(features)
        od_flows = self.flow_norm(od_flows)
        
        # 合并特征和OD流量
        combined_input = torch.cat([features, od_flows], dim=-1)
        
        # 编码特征
        encoded_features = self.feature_encoder(combined_input)
        encoded_features_skip = encoded_features  # 保存用于跳跃连接
        
        # 处理令牌特征（如果有）
        if token_features is not None:
            # 检查并适配token_features的维度
            if len(token_features.shape) == 2 and token_features.shape[1] != self.token_dim:
                # 创建一个临时投影层
                temp_projection = nn.Linear(token_features.shape[1], self.token_dim).to(token_features.device)
                token_features = temp_projection(token_features)
            
            # 使用跨模态对齐模块增强时序特征
            enhanced_features, _ = self.cross_modal_alignment(encoded_features, token_features)
            encoded_features = enhanced_features
        
        # 使用空间编码器处理features
        spatial_features = self.spatial_encoder(torch.mean(features, dim=1))
        
        # 应用时空联合注意力机制
        attended_features = self.spatiotemporal_attention(encoded_features, spatial_features)
        
        # 融合处理
        fused_features = self.fusion_layer(attended_features)
        
        # 添加位置编码并应用深层Informer编码器
        fused_features = self.position_encoding(fused_features)
        encoded_output = self.encoder(fused_features)
        
        # 添加跳跃连接
        if self.skip_connection:
            # 确保维度匹配
            if encoded_output.shape == encoded_features_skip.shape:
                # 使用残差连接
                residual_input = torch.cat([encoded_output, encoded_features_skip], dim=2)
                encoded_output = self.residual_layer(residual_input)
        
        # 应用时间注意力机制
        temporal_weighted = self.temporal_attention(encoded_output)
        
        # 取时间维度上的平均作为最终特征表示
        pooled_features = torch.mean(temporal_weighted, dim=1)
        
        # 输出层
        output = self.output_layer(pooled_features)
        
        return output


# 修改ODFlowGenerator_GAN类以适配新的判别器森林结构
class ODFlowGenerator_GAN(nn.Module):
    def __init__(self, hidden_dim=128, token_dim=768, time_steps=28, n_discriminators=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_steps = time_steps
        self.token_dim = token_dim
        self.n_discriminators = n_discriminators
        
        # 创建生成器
        self.generator = ODFlowGANGenerator(
            feature_dim=6,  # 更新为6个特征：站点a流入流出、站点b流入流出、距离特征和人口密度特征
            hidden_dim=hidden_dim,
            token_dim=token_dim,
            time_steps=time_steps,
            output_dim=2
        )
        
        # 使用基于Stacking机制的判别器森林 - 简化为3个专注于时序和空间特征的判别器
        self.discriminator_forest = ODFlowGANDiscriminatorForest(
            feature_dim=6,  # 更新为6个特征：站点a流入流出、站点b流入流出、距离特征和人口密度特征
            hidden_dim=hidden_dim,
            token_dim=token_dim,
            time_steps=time_steps,
            input_dim=2,
            n_discriminators=n_discriminators
        )
        
        # 特征融合层 - 用于将OD流量融合到特征中
        self.feature_fusion = nn.Linear(6 + 2, hidden_dim)
    
    def forward(self, features, target_od=None, token_features=None, valid_mask=None, mode='train'):
        batch_size, time_steps, _ = features.shape
        
        # 检查输入是否包含NaN值
        if torch.isnan(features).any():
            features = torch.nan_to_num(features, nan=0.0)
        
        if target_od is not None and torch.isnan(target_od).any():
            target_od = torch.nan_to_num(target_od, nan=0.0)
        
        # 如果没有提供掩码，创建全1掩码（所有位置都有效）
        if valid_mask is None:
            valid_mask = torch.ones(batch_size, time_steps, target_od.shape[-1] if target_od is not None else 2, 
                                   device=features.device, dtype=features.dtype)
        
        # 生成随机噪声
        noise = torch.randn(batch_size, time_steps, self.hidden_dim, device=features.device)
        
        # 生成OD流量
        generated_od = self.generator(features, noise, token_features)
        
        if mode == 'train' and target_od is not None:
            # 为判别器创建输入的副本，确保计算图分离
            features_d = features.detach().clone() if features.requires_grad else features
            target_od_d = target_od.detach().clone() if target_od.requires_grad else target_od
            token_features_d = token_features.detach().clone() if token_features is not None and token_features.requires_grad else token_features
            generated_od_d = generated_od.detach().clone()  # 必须分离生成器的输出
            
            # 使用判别器森林对真实数据进行判别 (现在返回all_base_scores, meta_score, weighted_avg_score)
            _, real_meta_score, _ = self.discriminator_forest(
                features_d, target_od_d, token_features_d, use_bootstrap=True
            )
            
            # 使用判别器森林对生成数据进行判别
            _, fake_meta_score, _ = self.discriminator_forest(
                features_d, generated_od_d, token_features_d, use_bootstrap=True
            )
            
            # 计算判别器损失 - 使用元判别器的输出
            d_loss_real = F.binary_cross_entropy_with_logits(
                real_meta_score, torch.ones_like(real_meta_score)
            )
            d_loss_fake = F.binary_cross_entropy_with_logits(
                fake_meta_score, torch.zeros_like(fake_meta_score)
            )
            d_loss = d_loss_real + d_loss_fake
            
            # 为生成器创建新的计算图
            _, fake_meta_score_g, _ = self.discriminator_forest(
                features, generated_od, token_features, use_bootstrap=False
            )
            
            # 生成器对抗损失 - 欺骗元判别器
            g_loss_adv = F.binary_cross_entropy_with_logits(
                fake_meta_score_g, torch.ones_like(fake_meta_score_g)
            )
            
            # 将掩码移到正确的设备
            if valid_mask.device != features.device:
                valid_mask = valid_mask.to(features.device)
            
            # 使用掩码计算L1损失（只计算有效位置的损失）
            masked_diff = torch.abs(generated_od - target_od) * valid_mask
            valid_count = valid_mask.sum()
            g_loss_l1 = masked_diff.sum() / (valid_count + 1e-8)
            
            # 使用掩码计算MSE损失（只计算有效位置的损失）
            masked_squared_diff = ((generated_od - target_od) ** 2) * valid_mask
            g_loss_mse = masked_squared_diff.sum() / (valid_count + 1e-8)
            
            # 改进的时间序列PCC损失 - 同时考虑序列级别和全局相关性（使用掩码）
            def pearson_correlation_loss(pred, target, mask):
                batch_size, seq_len, features = pred.shape
                
                # 方法1: 计算每个序列的PCC（时间维度），只考虑有效位置
                sequence_pccs = []
                for b in range(batch_size):
                    for f in range(features):
                        pred_seq = pred[b, :, f]  # [seq_len]
                        target_seq = target[b, :, f]  # [seq_len]
                        mask_seq = mask[b, :, f]  # [seq_len]
                        
                        # 只考虑有效位置
                        valid_indices = mask_seq > 0
                        if valid_indices.sum() < 2:  # 至少需要2个有效点才能计算相关性
                            continue
                        
                        pred_seq_valid = pred_seq[valid_indices]
                        target_seq_valid = target_seq[valid_indices]
                        
                        # 计算序列相关性
                        pred_mean = torch.mean(pred_seq_valid)
                        target_mean = torch.mean(target_seq_valid)
                        
                        pred_centered = pred_seq_valid - pred_mean
                        target_centered = target_seq_valid - target_mean
                        
                        covariance = torch.sum(pred_centered * target_centered)
                        pred_std = torch.sqrt(torch.sum(pred_centered ** 2))
                        target_std = torch.sqrt(torch.sum(target_centered ** 2))
                        
                        epsilon = 1e-8
                        correlation = covariance / (pred_std * target_std + epsilon)
                        sequence_pccs.append(correlation)
                
                if len(sequence_pccs) == 0:
                    # 如果没有有效的序列，返回一个默认值
                    return torch.tensor(1.0, device=pred.device, requires_grad=True)
                
                # 方法2: 计算全局PCC（所有有效数据展平）
                pred_flat = pred.reshape(-1)
                target_flat = target.reshape(-1)
                mask_flat = mask.reshape(-1)
                
                valid_indices_global = mask_flat > 0
                if valid_indices_global.sum() < 2:
                    # 如果没有足够的有效点，只使用序列级别的PCC
                    sequence_pcc_mean = torch.mean(torch.stack(sequence_pccs))
                    return 1.0 - sequence_pcc_mean
                
                pred_flat_valid = pred_flat[valid_indices_global]
                target_flat_valid = target_flat[valid_indices_global]
                
                pred_mean_global = torch.mean(pred_flat_valid)
                target_mean_global = torch.mean(target_flat_valid)
                
                pred_centered_global = pred_flat_valid - pred_mean_global
                target_centered_global = target_flat_valid - target_mean_global
                
                covariance_global = torch.sum(pred_centered_global * target_centered_global)
                pred_std_global = torch.sqrt(torch.sum(pred_centered_global ** 2))
                target_std_global = torch.sqrt(torch.sum(target_centered_global ** 2))
                
                epsilon = 1e-8
                correlation_global = covariance_global / (pred_std_global * target_std_global + epsilon)
                
                # 组合两种相关性：序列级别和全局级别
                sequence_pcc_mean = torch.mean(torch.stack(sequence_pccs))
                
                # 加权组合：更重视序列级别的相关性
                combined_correlation = 0.8 * sequence_pcc_mean + 0.2 * correlation_global
                
                # 返回损失 (1 - 相关系数)
                return 1.0 - combined_correlation
            
            # 计算PCC损失（使用掩码）
            g_loss_pcc = pearson_correlation_loss(generated_od, target_od, valid_mask)
            
            # DTW损失 (动态时间规整) - 使用可微分的近似实现（使用掩码）
            def dtw_loss(pred, target, mask, gamma=0.1):
                """
                可微分的DTW损失近似实现，保持梯度连续性
                使用MSE替代复杂的DTW计算，避免梯度断开问题
                """
                # 改用可微分的时序对齐损失
                # 1. 计算时序MSE损失（只考虑有效位置）
                masked_squared_diff = ((pred - target) ** 2) * mask
                valid_count = mask.sum()
                temporal_mse = masked_squared_diff.sum() / (valid_count + 1e-8)
                
                # 2. 计算时序方向的差分损失（关注趋势相似性，只考虑有效位置）
                pred_diff = pred[:, 1:, :] - pred[:, :-1, :]  # 时序差分
                target_diff = target[:, 1:, :] - target[:, :-1, :]
                # 掩码也需要相应调整（去掉第一个时间步）
                mask_diff = mask[:, 1:, :] * mask[:, :-1, :]  # 两个相邻时间步都有效才计算
                masked_diff_squared = ((pred_diff - target_diff) ** 2) * mask_diff
                valid_diff_count = mask_diff.sum()
                trend_loss = masked_diff_squared.sum() / (valid_diff_count + 1e-8) if valid_diff_count > 0 else torch.tensor(0.0, device=pred.device)
                
                # 3. 结合MSE和趋势损失
                combined_loss = temporal_mse + 0.5 * trend_loss
                
                return combined_loss
            
            # 计算DTW损失 - 计算成本较高，仅在小批量上计算
            if batch_size <= 16:  # 限制批次大小以避免计算过大
                g_loss_dtw = dtw_loss(generated_od, target_od, valid_mask)
            else:
                # 随机抽样一部分进行计算
                idx = torch.randperm(batch_size)[:16]
                g_loss_dtw = dtw_loss(generated_od[idx], target_od[idx], valid_mask[idx])
            
            # 使用模型自带的特征融合层
            features_with_od = torch.cat([features, generated_od], dim=-1)
            hidden_features = self.feature_fusion(features_with_od)
            
            # 简化的损失调试信息（仅在需要时输出）
            if torch.isnan(g_loss_adv).any() or torch.isnan(g_loss_l1).any() or torch.isnan(g_loss_mse).any() or torch.isnan(g_loss_pcc).any() or torch.isnan(g_loss_dtw).any():
                print("警告：检测到NaN损失，输出详细调试信息")
                print(f"  g_loss_adv: {g_loss_adv.item():.6f} (NaN: {torch.isnan(g_loss_adv).any()})")
                print(f"  g_loss_l1: {g_loss_l1.item():.6f} (NaN: {torch.isnan(g_loss_l1).any()})")
                print(f"  g_loss_mse: {g_loss_mse.item():.6f} (NaN: {torch.isnan(g_loss_mse).any()})")
                print(f"  g_loss_pcc: {g_loss_pcc.item():.6f} (NaN: {torch.isnan(g_loss_pcc).any()})")
                print(f"  g_loss_dtw: {g_loss_dtw.item():.6f} (NaN: {torch.isnan(g_loss_dtw).any()})")
            
            # 组合损失计算 - 增强PCC和DTW损失权重以提高时间序列相关性
            # 原权重: L1=1.0, MSE=0.5, PCC=0.4, DTW=0.2
            # 新权重: 降低L1/MSE权重，提高PCC/DTW权重
            g_loss = g_loss_adv + 0.3 * g_loss_l1 + 0.2 * g_loss_mse + 1.0 * g_loss_pcc + 0.8 * g_loss_dtw
            
            # 检查组合损失是否正常
            if torch.isnan(g_loss).any() or torch.isinf(g_loss).any():
                print(f"警告：组合损失异常 - NaN: {torch.isnan(g_loss).any()}, Inf: {torch.isinf(g_loss).any()}")
                print(f"组合损失原始值: {g_loss.item():.6f}")
                # 如果异常，重置为安全值
                g_loss = torch.tensor(0.01, device=g_loss.device, requires_grad=True)
            
            # 恢复原始缩放因子
            loss_scale = 0.01  # 恢复原来的缩放因子
            g_loss = g_loss * loss_scale
            print(f"最终生成器损失: {g_loss.item():.6f}")
            
            return {
                'd_loss': d_loss,
                'g_loss': g_loss,
                'g_loss_adv': g_loss_adv,
                'g_loss_l1': g_loss_l1,
                'g_loss_mse': g_loss_mse,
                'g_loss_pcc': g_loss_pcc,
                'g_loss_dtw': g_loss_dtw
            }
        else:
            return generated_od
            
    def generate(self, features, token_features=None):
        """生成OD流量"""
        return self.forward(features, token_features=token_features, mode='eval')

# ========== 数据集类 ==========
class ODFlowDataset(Dataset):
    """OD流量数据集"""
    def __init__(self, io_flow_path, graph_path, od_matrix_path, test_ratio=0.2, val_ratio=0.1, seed=42, zero_ratio_threshold=0.7):
        """
        初始化数据集
        Args:
            io_flow_path: IO流量数据路径 (站点i流入, 站点i流出, 站点j流入, 站点j流出)
            graph_path: 站点间邻接矩阵路径
            od_matrix_path: OD矩阵路径
            test_ratio: 测试集比例
            val_ratio: 验证集比例
            seed: 随机种子
            zero_ratio_threshold: 稀疏OD对识别阈值，IO流量序列中0值比例超过此值（默认0.7即70%）的OD对将被标记为稀疏
        """
        super().__init__()
        self.io_flow = np.load(io_flow_path) # (时间步, 站点数, 2)
        self.graph = np.load(graph_path) # (站点数, 站点数)
        self.od_matrix = np.load(od_matrix_path) # (时间步, 站点数, 站点数)
        
        # 加载站点/区域人口密度数据
        # 根据数据集类型自动选择正确的文件
        population_files = [
            "/private/od/data_NYTaxi/grid_population_density_52nodes.json",  # 纽约数据（52节点）
            "/private/od/data_NYTaxi/grid_population_density_filtered.json",  # 纽约数据（过滤后，备用）
            "/private/od/data_NYTaxi/grid_population_density.json",  # 纽约数据（原始，备用）
            "/private/od/data/area_p.json"  # 北京数据
        ]
        
        self.area_data = []
        for pop_file in population_files:
            try:
                if os.path.exists(pop_file):
                    with open(pop_file, "r", encoding="utf-8") as f:
                        self.area_data = json.load(f)
                    print(f"加载人口密度数据成功: {pop_file}，共 {len(self.area_data)} 个区域/站点")
                    break
            except Exception as e:
                print(f"加载 {pop_file} 失败: {str(e)}")
                continue
        
        if not self.area_data:
            print("未找到可用的人口密度数据文件")
            self.area_data = []
        
        # 设置归一化的分位数参数
        self.quantile_lower = 0.01
        self.quantile_upper = 0.99
        
        self.num_nodes = self.io_flow.shape[1]
        self.time_steps = self.io_flow.shape[0]
        
        # 计算全局统计量用于归一化
        print("计算全局统计量...")
        self._compute_global_statistics()
        
        # 站点对列表
        self.od_pairs = []
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                self.od_pairs.append((i, j))
        
        # 识别稀疏OD对（IO流量序列存在过多0值的）
        print(f"识别稀疏OD对（阈值: {zero_ratio_threshold*100:.1f}%）...")
        self.sparse_od_mask = self._identify_sparse_od_pairs(zero_ratio_threshold=zero_ratio_threshold)
        sparse_count = np.sum(self.sparse_od_mask == 0)
        total_count = len(self.od_pairs)
        print(f"识别出 {sparse_count}/{total_count} 个稀疏OD对 ({(sparse_count/total_count*100):.2f}%)")
        
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
    
    def _compute_global_statistics(self):
        """计算全局统计量用于数据归一化"""
        # IO流量全局统计量
        self.global_io_mean = np.mean(self.io_flow)
        self.global_io_std = np.std(self.io_flow)
        self.global_io_max = np.max(self.io_flow)
        
        # OD流量全局统计量
        self.global_od_mean = np.mean(self.od_matrix)
        self.global_od_std = np.std(self.od_matrix)
        self.global_od_max = np.max(self.od_matrix)
        
        # 避免除零错误
        if self.global_io_std == 0:
            self.global_io_std = 1.0
        if self.global_od_std == 0:
            self.global_od_std = 1.0
            
        print(f"IO流量全局统计: mean={self.global_io_mean:.2f}, std={self.global_io_std:.2f}, max={self.global_io_max:.2f}")
        print(f"OD流量全局统计: mean={self.global_od_mean:.2f}, std={self.global_od_std:.2f}, max={self.global_od_max:.2f}")
    
    def _identify_sparse_od_pairs(self, zero_ratio_threshold=0.7):
        """
        识别稀疏OD对（IO流量序列存在过多0值的）
        Args:
            zero_ratio_threshold: 0值比例阈值，超过此比例的OD对被认为是稀疏的
        Returns:
            sparse_mask: numpy数组，1表示有效OD对，0表示稀疏OD对
        """
        sparse_mask = np.ones(len(self.od_pairs), dtype=np.float32)
        
        for idx, (i, j) in enumerate(self.od_pairs):
            # 获取站点i和j的IO流量
            io_flow_i = self.io_flow[:, i, :]  # (时间步, 2)
            io_flow_j = self.io_flow[:, j, :]  # (时间步, 2)
            
            # 计算IO流量的0值比例
            io_flow_combined = np.concatenate([io_flow_i.flatten(), io_flow_j.flatten()])
            zero_ratio = np.sum(io_flow_combined == 0) / len(io_flow_combined)
            
            # 如果0值比例超过阈值，标记为稀疏OD对
            if zero_ratio > zero_ratio_threshold:
                sparse_mask[idx] = 0
        
        return sparse_mask
    
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
        
        # 获取双向OD流量
        od_i_to_j = self.od_matrix[:, site_i, site_j]  # 从i到j的流量 (时间步,)
        od_j_to_i = self.od_matrix[:, site_j, site_i]  # 从j到i的流量 (时间步,)
        
        # 将两个方向的OD流量组合 [时间步, 2]
        od_flows = np.stack([od_i_to_j, od_j_to_i], axis=1)
        
        # 获取IO流量
        io_flow_i = self.io_flow[:, site_i, :]  # (时间步, 2)
        io_flow_j = self.io_flow[:, site_j, :]  # (时间步, 2)
        
        # 改进的数据归一化策略
        # 方法1: 使用全局统计量进行标准化
        io_flow_i_normalized = (io_flow_i - self.global_io_mean) / self.global_io_std
        io_flow_j_normalized = (io_flow_j - self.global_io_mean) / self.global_io_std
        
        # 方法2: 对于小值数据，使用对数变换处理
        # 对于OD流量，由于存在大量小值，使用log1p变换
        od_flows_log = np.log1p(od_flows)  # log1p(x) = log(1+x)，对小值更友好
        od_flows_normalized = od_flows_log / np.log1p(self.global_od_max)
        
        # 备选方案：使用全局统计量标准化OD流量
        # od_flows_normalized = (od_flows - self.global_od_mean) / self.global_od_std
        
        # 获取站点对距离并归一化
        distance = self.graph[site_i, site_j]
        # 对距离进行归一化处理
        distance_normalized = distance / np.max(self.graph)  # 归一化到[0,1]范围
        
        # 获取站点人口密度并归一化
        if hasattr(self, 'area_data') and len(self.area_data) > 0:
            # 确保站点索引不超过可用的站点数据
            if site_i < len(self.area_data) and site_j < len(self.area_data):
                pop_density_i = self.area_data[site_i].get('grid_population_density', 0.0)
                pop_density_j = self.area_data[site_j].get('grid_population_density', 0.0)
            else:
                # 如果站点索引超出范围，使用默认值
                pop_density_i = 0.0
                pop_density_j = 0.0
                
            # 计算人口密度特征（两站点人口密度的平均值）
            pop_density = (pop_density_i + pop_density_j) / 2
            
            # 人口密度归一化 - 使用所有站点的最大人口密度归一化
            max_pop_density = max([area.get('grid_population_density', 1.0) for area in self.area_data])
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
        
        # 获取该OD对的稀疏掩码（1表示有效，0表示稀疏）
        od_mask = self.sparse_od_mask[site_pair_idx]
        
        # 如果是稀疏OD对，将OD流量设为0
        if od_mask == 0:
            od_flows_normalized = np.zeros_like(od_flows_normalized)
        
        # 创建时间步级别的掩码 [时间步, 2]，用于损失计算
        # 1表示有效位置，0表示应该被忽略的位置
        valid_mask = np.ones((od_flows_normalized.shape[0], od_flows_normalized.shape[1]), dtype=np.float32) * od_mask
        
        # 返回特征、归一化后的OD流量和掩码
        return torch.FloatTensor(features), torch.FloatTensor(od_flows_normalized), torch.FloatTensor(valid_mask)

# ========== 训练函数 ==========
def train_llm_gan(args, precomputed_features=None):
    """
    使用大模型特征训练GAN
    Args:
        args: 参数对象
        precomputed_features: 预计算的token特征字典 {pair_key: token_feature}
    Returns:
        best_model_path: 最佳模型路径
    """
    # 设置随机种子
    set_seed(args.seed)
    
    # 禁用异常检测以避免训练中断 - 在生产环境中通常关闭以提高性能
    # torch.autograd.set_detect_anomaly(True)  # 调试时可以临时启用
    
    # 设置设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据集
    dataset = ODFlowDataset(
        io_flow_path=args.io_flow_path,
        graph_path=args.graph_path,
        od_matrix_path=args.od_matrix_path,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        zero_ratio_threshold=args.zero_ratio_threshold
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
    
    dataset.set_mode('test')
    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    dataset.set_mode('train')
    
    # 只有在不使用预计算特征时才加载Qwen2特征提取器
    qwen_extractor = None
    if precomputed_features is None:
        qwen_extractor = QwenFeatureExtractor(
            feature_dim=args.token_dim,
            device=device
        )
    else:
        print(f"使用预计算的token特征，包含 {len(precomputed_features)} 个站点对")
    
    # 创建简化的Forest-GAN模型 - 使用3个专注于时序和空间特征的判别器
    model = ODFlowGenerator_GAN(
        hidden_dim=args.hidden_dim,
        token_dim=args.token_dim,
        time_steps=28,
        n_discriminators=3  # 简化为3个判别器：时序敏感+空间敏感+基础判别器
    ).to(device)
    
    # 优化器
    g_optimizer = torch.optim.AdamW(model.generator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    d_optimizer = torch.optim.AdamW(model.discriminator_forest.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 学习率调度器
    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(g_optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(d_optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    # 训练准备
    best_val_loss = float('inf')
    best_model_path = os.path.join(args.output_dir, 'best_llm_gan_model.pth')
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
    print("Starting to train LLM-GAN model...")
    try:
        for epoch in range(args.epochs):
            model.train()
            train_g_losses, train_d_losses = [], []
            train_dtw_losses, train_pcc_losses = [], []  # 修改为DTW损失和PCC损失
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Training)")
            for batch in pbar:
                # 清除缓存和梯度，防止内存泄漏
                torch.cuda.empty_cache()
                
                # 从DataLoader获取数据 (返回的是features、od_flows和valid_mask)
                if len(batch) == 3:
                    features, od_flows, valid_mask = batch
                else:
                    # 兼容旧版本（没有掩码的情况）
                    features, od_flows = batch
                    valid_mask = None
                
                batch_size = features.shape[0]
                
                # 打印批次数据的形状
                # print(f"\n批次数据形状 - batch_size: {batch_size}")
                # print(f"features形状: {features.shape}")
                # print(f"od_flows形状: {od_flows.shape}")
                # print(f"valid_mask形状: {valid_mask.shape if valid_mask is not None else None}")
                
                # 将数据移至设备
                features = features.to(device)
                od_flows = od_flows.to(device)
                if valid_mask is not None:
                    valid_mask = valid_mask.to(device)
                
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
                    #print(f"调整后的OD流量形状: {od_flows.shape}")
                
                # print(f"构建的特征向量形状: {features.shape}")
                # print(f"特征数据类型: {features.dtype}")
                
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
                
                # 提取批次的令牌特征
                token_features_batch = []
                
                # 使用预计算的token特征或实时提取
                if precomputed_features is not None:
                    # 使用预计算的token特征
                    for i, (site_i, site_j) in enumerate(site_pairs):
                        pair_key = f"{site_i}_{site_j}"
                        if pair_key in precomputed_features:
                            # 使用预计算特征
                            token_feature = precomputed_features[pair_key].to(device)
                        else:
                            # 没有预计算特征，使用零特征
                            token_feature = torch.zeros(args.token_dim, device=device)
                            print(f"警告: 站点对 {pair_key} 没有预计算特征")
                        
                        token_features_batch.append(token_feature)
                else:
                    # 实时提取token特征
                    for i, (site_i, site_j) in enumerate(site_pairs):
                        # 提取站点IO流量
                        area_io_flow_i = dataset.io_flow[:, site_i, :].reshape(28, 2)
                        area_io_flow_j = dataset.io_flow[:, site_j, :].reshape(28, 2)
                        
                        # 转换为tensor
                        area_io_flow_i_tensor = torch.FloatTensor(area_io_flow_i).to(device)
                        area_io_flow_j_tensor = torch.FloatTensor(area_io_flow_j).to(device)
                        
                        # 提取令牌特征
                        with torch.no_grad():
                            # 将站点数据传递给提取器
                            token_feature, _ = qwen_extractor(site_i, site_j, area_io_flow_i_tensor, area_io_flow_j_tensor, 
                                                            dataset.area_data if hasattr(dataset, 'area_data') else None,
                                                            prompt_type=args.prompt_type)
                        
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
                
                # 优化判别器森林
                d_optimizer.zero_grad()
                try:
                    print("执行判别器森林前向传播...")
                    # 创建输入的副本，确保不会影响生成器的计算图
                    features_d = features.detach().clone()
                    od_flows_d = od_flows.detach().clone()
                    token_features_d = token_features.detach().clone()
                    
                    # 使用detach的输入执行前向传播，只计算判别器的损失
                    # Forest-GAN: 每个判别器在自己的自举样本上训练
                    with torch.set_grad_enabled(True):
                        # 使用判别器森林进行训练，通过自举采样增强鲁棒性
                        # 每个判别器在不同的数据分布上训练，避免过拟合
                        
                        # 传递特征给模型 - Forest-GAN: 使用自举采样分布训练多个判别器
                        out = model(features_d, od_flows_d, token_features_d, valid_mask=valid_mask, mode='train')
                        print("判别器森林前向传播完成")
                        
                        # 获取判别器森林的平均损失 - 每个判别器在不同的数据分布上训练
                        d_loss = out['d_loss']
                        
                        # 单独进行判别器的反向传播
                        d_loss.backward()
                        
                        # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(model.discriminator_forest.parameters(), max_norm=1.0)
                        
                        # 优化判别器参数
                        d_optimizer.step()
                        
                        # 记录判别器损失
                        train_d_losses.append(d_loss.item())
                except Exception as e:
                    print(f"判别器前向传播或反向传播出错: {str(e)}")
                    traceback.print_exc()
                    continue
                
                # 优化生成器 - Forest-GAN: 生成器需要对抗多个判别器
                g_optimizer.zero_grad()
                try:
                    print("执行生成器前向传播...")
                    # 确保使用新的计算图
                    with torch.set_grad_enabled(True):
                        # 令牌特征用于增强时序特征
                        
                        # 传递特征给模型 - 生成器通过平均梯度对抗多个判别器
                        out_g = model(features, od_flows, token_features, valid_mask=valid_mask, mode='train')
                        # Forest-GAN: g_loss已经包含了来自多个判别器的平均梯度
                        g_loss = out_g['g_loss']
                        
                        # 生成器反向传播
                        g_loss.backward()
                        
                        # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(model.generator.parameters(), max_norm=1.0)
                        
                        # 优化生成器参数
                        g_optimizer.step()
                        
                        # 记录损失
                        train_g_losses.append(g_loss.item())
                        # 使用g_loss_dtw替代spatiotemporal_consistency_loss
                        train_dtw_losses.append(out_g.get('g_loss_dtw', torch.tensor(0.0)).item())
                        # 使用g_loss_pcc替代flow_conservation_loss
                        train_pcc_losses.append(out_g.get('g_loss_pcc', torch.tensor(0.0)).item())
                        
                        # 显示进度 - 使用更高精度的格式显示损失
                        pbar.set_postfix({
                            'g_loss': f"{g_loss.item():.6f}", 
                            'd_loss': f"{d_loss.item():.6f}",
                            'dtw_loss': f"{out_g.get('g_loss_dtw', torch.tensor(0.0)).item():.6f}",
                            'pcc_loss': f"{out_g.get('g_loss_pcc', torch.tensor(0.0)).item():.6f}"
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
            avg_dtw_loss = np.mean(train_dtw_losses) if train_dtw_losses else float('inf')
            avg_pcc_loss = np.mean(train_pcc_losses) if train_pcc_losses else float('inf')
            
            # 使用evaluate_llm_gan_model函数评估训练集
            dataset.set_mode('train')
            # 创建一个小的训练集样本进行评估，以避免计算过大
            # 从训练集中随机选择少量样本用于快速评估
            small_train_dataset = torch.utils.data.Subset(
                dataset,
                indices=np.random.choice(len(dataset), min(5 * args.batch_size, len(dataset)), replace=False)
            )
            small_train_loader = DataLoader(
                small_train_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True
            )
            
            # 评估训练集性能
            print("\n评估训练集性能...")
            train_metrics = evaluate_llm_gan_model(
                small_train_loader, # 使用较小的训练集样本
                qwen_extractor,
                model,
                device,
                args,
                output_dir=None, # 不输出详细结果
                precomputed_features=precomputed_features  # 传递预计算的特征
            )
            
            # 验证
            model.eval()
            dataset.set_mode('val')
            print("\n评估验证集性能...")
            val_metrics = evaluate_llm_gan_model(
                val_loader,
                qwen_extractor,
                model,
                device,
                args,
                output_dir=None,
                precomputed_features=precomputed_features  # 传递预计算的特征
            )
            
            # 评估测试集
            dataset.set_mode('test')
            print("\n评估测试集性能...")
            test_metrics = evaluate_llm_gan_model(
                test_loader,
                qwen_extractor,
                model,
                device,
                args,
                output_dir=None,
                precomputed_features=precomputed_features  # 传递预计算的特征
            )
            
            # 切换回训练模式
            dataset.set_mode('train')
            
            # 打印训练信息
            print(f"Epoch {epoch+1}/{args.epochs}")
            print(f"   Training - G_Loss: {avg_g_loss:.6f}, D_Loss: {avg_d_loss:.6f}, DTW_Loss: {avg_dtw_loss:.6f}, PCC_Loss: {avg_pcc_loss:.6f}")
            print(f"   Training - RMSE: {train_metrics['RMSE']:.6f}, MAE: {train_metrics['MAE']:.6f}, PCC: {train_metrics['PCC']:.6f}")
            print(f"   Validation - RMSE: {val_metrics['RMSE']:.6f}, MAE: {val_metrics['MAE']:.6f}, PCC: {val_metrics['PCC']:.6f}")
            print(f"   Test - RMSE: {test_metrics['RMSE']:.6f}, MAE: {test_metrics['MAE']:.6f}, PCC: {test_metrics['PCC']:.6f}")
            
            # 记录到日志
            with open(log_file, 'a') as f:
                f.write(f"Epoch {epoch+1}/{args.epochs}\n")
                f.write(f"   Training - G_Loss: {avg_g_loss:.6f}, D_Loss: {avg_d_loss:.6f}, DTW_Loss: {avg_dtw_loss:.6f}, PCC_Loss: {avg_pcc_loss:.6f}\n")
                f.write(f"   Training - RMSE: {train_metrics['RMSE']:.6f}, MAE: {train_metrics['MAE']:.6f}, PCC: {train_metrics['PCC']:.6f}\n")
                f.write(f"   Validation - RMSE: {val_metrics['RMSE']:.6f}, MAE: {val_metrics['MAE']:.6f}, PCC: {val_metrics['PCC']:.6f}\n")
                f.write(f"   Test - RMSE: {test_metrics['RMSE']:.6f}, MAE: {test_metrics['MAE']:.6f}, PCC: {test_metrics['PCC']:.6f}\n")
            
            # 使用验证集的指标来指导模型训练和早停
            # 修改组合分数计算方式，更强调PCC（序列趋势相关性）
            combined_score = val_metrics['RMSE'] - val_metrics['PCC'] * 1.2  # 值越低越好（较低RMSE和较高PCC）
            
            if combined_score < best_val_loss:
                best_val_loss = combined_score
                early_stop_counter = 0
                
                # 保存最佳模型前确保token_compressor已正确初始化
                # 检查模型中是否有动态创建的token_compressor
                if model.discriminator_forest.token_compressor is None:
                    # 如果没有token_compressor，创建一个默认的
                    print("保存模型前初始化token_compressor")
                    input_dim = 3584  # 通常为Qwen特征维度
                    output_dim = args.token_dim // 8
                    model.discriminator_forest.token_output_dim = output_dim
                    model.discriminator_forest.token_compressor = nn.Sequential(
                        nn.Linear(input_dim, output_dim),
                        nn.LeakyReLU(0.2),
                        nn.LayerNorm(output_dim)
                    ).to(device)
                
                # 保存最佳模型
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'g_optimizer_state_dict': g_optimizer.state_dict(),
                    'd_optimizer_state_dict': d_optimizer.state_dict(),
                    'rmse': val_metrics['RMSE'],
                    'mae': val_metrics['MAE'],
                    'pcc': val_metrics['PCC'],
                    'epoch': epoch,
                    'token_dim': args.token_dim,
                    'hidden_dim': args.hidden_dim,
                    'token_compressor_input_dim': model.discriminator_forest.token_compressor[0].in_features,
                    'token_compressor_output_dim': model.discriminator_forest.token_output_dim
                }, best_model_path)
                print(f"   New best model saved (Val RMSE: {val_metrics['RMSE']:.6f}, Val PCC: {val_metrics['PCC']:.6f}, Combined score: {combined_score:.6f})")
                with open(log_file, 'a') as f:
                    f.write(f"   New best model saved (Val RMSE: {val_metrics['RMSE']:.6f}, Val PCC: {val_metrics['PCC']:.6f}, Combined score: {combined_score:.6f})\n")
            else:
                early_stop_counter += 1
                if early_stop_counter >= early_stop_patience:
                    print(f"Early stopping: {early_stop_patience} epochs of validation performance did not improve")
                    break
            
            # 不再每5个epoch保存检查点，只保存最佳模型和最后一轮
            
            print(f"   Best result so far: Combined score: {best_val_loss:.6f}")
        
        # 保存最后一轮模型前确保token_compressor已正确初始化
        if model.discriminator_forest.token_compressor is None:
            # 如果没有token_compressor，创建一个默认的
            print("保存最终模型前初始化token_compressor")
            input_dim = 3584  # 通常为Qwen特征维度
            output_dim = args.token_dim // 8
            model.discriminator_forest.token_output_dim = output_dim
            model.discriminator_forest.token_compressor = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LeakyReLU(0.2),
                nn.LayerNorm(output_dim)
            ).to(device)
        
        # 保存最后一轮模型
        final_model_path = os.path.join(args.output_dir, 'final_llm_gan_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'g_optimizer_state_dict': g_optimizer.state_dict(),
            'd_optimizer_state_dict': d_optimizer.state_dict(),
            'rmse': val_metrics['RMSE'],
            'mae': val_metrics['MAE'],
            'pcc': val_metrics['PCC'],
            'epoch': epoch,
            'token_dim': args.token_dim,
            'hidden_dim': args.hidden_dim,
            'token_compressor_input_dim': model.discriminator_forest.token_compressor[0].in_features,
            'token_compressor_output_dim': model.discriminator_forest.token_output_dim
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

def evaluate_llm_gan_model(test_loader, qwen_extractor, model, device, args, output_dir=None, precomputed_features=None):
    """
    评估模型性能
    Args:
        test_loader: 测试数据加载器
        qwen_extractor: Qwen2特征提取器
        model: GAN模型
        device: 设备
        args: 参数对象
        output_dir: 输出目录
        precomputed_features: 预计算的token特征字典 {pair_key: token_feature}
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
    
    # 保存所有样本的预测结果，用于找出最准确的站点对
    all_samples = []
    
    # 获取数据集以供获取原始站点对
    dataset = test_loader.dataset
    
    # 处理Subset对象，获取原始数据集和索引
    original_dataset = dataset
    indices = None
    if isinstance(dataset, torch.utils.data.Subset):
        # 如果是Subset，获取原始数据集和子集索引
        original_dataset = dataset.dataset
        indices = dataset.indices
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating model")):
            # 处理可能包含掩码的批次数据
            if len(batch) == 3:
                features, od_flows_normalized, valid_mask = batch
            else:
                features, od_flows_normalized = batch
                valid_mask = None
            
            # 获取批次大小
            batch_size = features.shape[0]
            
            # 获取当前批次的站点对
            batch_start_idx = batch_idx * args.batch_size
            
            # 根据数据集类型处理索引和站点对获取
            if indices is not None:
                # 如果是Subset，需要通过indices获取原始数据集的索引
                batch_indices = indices[batch_start_idx:batch_start_idx + batch_size]
                # 再通过原始数据集的current_indices获取站点对
                site_indices = [original_dataset.current_indices[idx] for idx in batch_indices]
                site_pairs = [original_dataset.od_pairs[idx] for idx in site_indices]
            else:
                # 如果是原始数据集，直接使用current_indices
                batch_indices = batch_start_idx + torch.arange(batch_size)
                batch_indices = batch_indices[batch_indices < len(original_dataset.current_indices)]
                site_pairs = [original_dataset.od_pairs[original_dataset.current_indices[idx.item()]] for idx in batch_indices]
            
            # 移动数据到设备
            features = features.to(device)
            od_flows_normalized = od_flows_normalized.to(device)
            if valid_mask is not None:
                valid_mask = valid_mask.to(device)
            
            # 检查NaN值
            if torch.isnan(features).any():
                features = torch.nan_to_num(features, nan=0.0)
            if torch.isnan(od_flows_normalized).any():
                od_flows_normalized = torch.nan_to_num(od_flows_normalized, nan=0.0)
            
            # 提取批次的令牌特征
            token_features_batch = []
            
            # 使用预计算的token特征或实时提取
            if precomputed_features is not None:
                # 使用预计算的token特征
                for i, (site_i, site_j) in enumerate(site_pairs):
                    pair_key = f"{site_i}_{site_j}"
                    if pair_key in precomputed_features:
                        # 使用预计算特征
                        token_feature = precomputed_features[pair_key].to(device)
                    else:
                        # 没有预计算特征，使用零特征
                        token_feature = torch.zeros(args.token_dim, device=device)
                        print(f"警告: 评估时站点对 {pair_key} 没有预计算特征")
                    
                    token_features_batch.append(token_feature)
            elif qwen_extractor is not None:
                # 实时提取token特征
                for i, (site_i, site_j) in enumerate(site_pairs):
                    # 提取站点IO流量
                    io_flow_i = original_dataset.io_flow[:, site_i, :].reshape(28, 2)
                    io_flow_j = original_dataset.io_flow[:, site_j, :].reshape(28, 2)
                    
                    # 转换为tensor
                    io_flow_i_tensor = torch.FloatTensor(io_flow_i).to(device)
                    io_flow_j_tensor = torch.FloatTensor(io_flow_j).to(device)
                    
                    # 提取特征向量
                    token_feature, _ = qwen_extractor(site_i, site_j, io_flow_i_tensor, io_flow_j_tensor,
                                                    original_dataset.area_data if hasattr(original_dataset, 'area_data') else None,
                                                    prompt_type=args.prompt_type)
                    token_features_batch.append(token_feature)
            else:
                # 既没有预计算特征也没有提取器，使用零特征
                print("警告: 评估时既没有预计算特征也没有特征提取器，使用零特征")
                for _ in range(len(site_pairs)):
                    token_features_batch.append(torch.zeros(args.token_dim, device=device))
            
            # 将令牌特征组合为批次
            if token_features_batch:
                token_features = torch.stack(token_features_batch).to(device)
            else:
                # 如果批次为空，创建零张量
                token_features = torch.zeros(batch_size, args.token_dim, device=device)
            
            print(f"评估阶段 - 令牌特征形状: {token_features.shape}")
            
            # 生成样本，直接传递token_features
            generated_normalized = model.generate(features, token_features)
            
            # 计算MSE和MAE（只考虑有效位置）
            if valid_mask is not None:
                # 使用掩码只计算有效位置的损失
                masked_squared_diff = ((generated_normalized - od_flows_normalized) ** 2) * valid_mask
                masked_abs_diff = torch.abs(generated_normalized - od_flows_normalized) * valid_mask
                valid_count = valid_mask.sum().item()
                mse = masked_squared_diff.sum().item() / (valid_count + 1e-8) * valid_count  # 保持与原来的scale一致
                mae = masked_abs_diff.sum().item()
            else:
                # 如果没有掩码，使用原来的方法
                mse = F.mse_loss(generated_normalized, od_flows_normalized, reduction='sum').item()
                mae = F.l1_loss(generated_normalized, od_flows_normalized, reduction='sum').item()
            
            # 存储用于PCC计算（只存储有效位置的数据）
            if valid_mask is not None:
                # 只存储有效位置的数据
                generated_valid = generated_normalized * valid_mask
                targets_valid = od_flows_normalized * valid_mask
                all_generated.append(generated_valid.cpu().numpy())
                all_targets.append(targets_valid.cpu().numpy())
            else:
                all_generated.append(generated_normalized.cpu().numpy())
                all_targets.append(od_flows_normalized.cpu().numpy())
            
            # 保存所有样本数据（反归一化后）
            # 获取反归一化参数
            if hasattr(original_dataset, 'quantile_lower') and hasattr(original_dataset, 'quantile_upper'):
                q_lower = np.percentile(original_dataset.od_matrix.flatten(), original_dataset.quantile_lower * 100)
                q_upper = np.percentile(original_dataset.od_matrix.flatten(), original_dataset.quantile_upper * 100)
            else:
                # 默认使用1%和99%的分位数
                q_lower = np.percentile(original_dataset.od_matrix.flatten(), 1)
                q_upper = np.percentile(original_dataset.od_matrix.flatten(), 99)
            
            # 反归一化
            generated = generated_normalized.cpu().numpy() * (q_upper - q_lower) + q_lower
            real_od = od_flows_normalized.cpu().numpy() * (q_upper - q_lower) + q_lower
            
            # 保存所有样本数据，用于后续找出最准确的站点对
            for i in range(min(batch_size, len(site_pairs))):
                # 计算预测误差（两条预测的双向流量序列逐天计算预测值与真实值的差值绝对值，并累加）
                error_sum = np.sum(np.abs(generated[i] - real_od[i]))
                
                all_samples.append({
                    'area_pair': site_pairs[i],
                    'predicted': generated[i],
                    'real': real_od[i],
                    'error': error_sum
                })
            
            # 累计指标
            total_mse += mse
            total_mae += mae
            if valid_mask is not None:
                # 只计算有效位置的样本数
                total_samples += valid_mask.sum().item()
            else:
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
        
        # 根据误差对所有站点对进行排序，选出最准确的5个站点对
        top_samples = sorted(all_samples, key=lambda x: x['error'])[:5]
        
        # 保存5个最佳样本的预测值和实际值
        best_samples_file = os.path.join(output_dir, "best_predictions.txt")
        with open(best_samples_file, "w") as f:
            f.write("最准确的5个站点对预测值与真实值对比 (误差从小到大排序):\n\n")
            
            for idx, sample in enumerate(top_samples):
                area_i, area_j = sample['area_pair']
                f.write(f"样本 {idx+1} - 站点对: ({area_i}, {area_j}), 总误差: {sample['error']:.6f}\n")
                
                # i到j方向
                f.write(f"站点 {area_i} 到站点 {area_j} 的流量:\n")
                f.write("时间步\t预测值\t真实值\n")
                for t in range(len(sample['predicted'])):
                    f.write(f"{t+1}\t{sample['predicted'][t, 0]:.6f}\t{sample['real'][t, 0]:.6f}\n")
                
                # j到i方向
                f.write(f"\n站点 {area_j} 到站点 {area_i} 的流量:\n")
                f.write("时间步\t预测值\t真实值\n")
                for t in range(len(sample['predicted'])):
                    f.write(f"{t+1}\t{sample['predicted'][t, 1]:.6f}\t{sample['real'][t, 1]:.6f}\n")
                
                f.write("\n" + "="*50 + "\n\n")
        
        # 创建可视化图表，为最准确的5个站点对分别创建两张图表（每个方向一张）
        try:
            import matplotlib.pyplot as plt
            
            # 创建一个子目录用于保存图片
            viz_dir = os.path.join(output_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            
            for idx, sample in enumerate(top_samples):
                area_i, area_j = sample['area_pair']
                
                # 第一张图：站点i到站点j方向
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                time_steps = list(range(1, 29))
                
                # 绘制从i到j的真实流量
                ax1.plot(time_steps, sample['real'][:, 0], 'b-', linewidth=2, marker='o', label='Actual Flow')
                
                # 绘制从i到j的预测流量
                ax1.plot(time_steps, sample['predicted'][:, 0], 'r--', linewidth=2, marker='x', label='Predicted Flow')
                
                # 设置图表属性
                ax1.set_title(f'Flow from Area {area_i} to Area {area_j}', fontsize=14)
                ax1.set_xlabel('Day', fontsize=12)
                ax1.set_ylabel('Flow Volume', fontsize=12)
                ax1.legend(loc='best', fontsize=10)
                ax1.grid(True, linestyle='--', alpha=0.7)
                
                # 保存图表
                fig_path1 = os.path.join(viz_dir, f'area_{area_i}_to_{area_j}.png')
                plt.tight_layout()
                plt.savefig(fig_path1, dpi=300)
                plt.close(fig1)
                
                # 第二张图：站点j到站点i方向
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                
                # 绘制从j到i的真实流量
                ax2.plot(time_steps, sample['real'][:, 1], 'b-', linewidth=2, marker='o', label='Actual Flow')
                
                # 绘制从j到i的预测流量
                ax2.plot(time_steps, sample['predicted'][:, 1], 'r--', linewidth=2, marker='x', label='Predicted Flow')
                
                # 设置图表属性
                ax2.set_title(f'Flow from Area {area_j} to Area {area_i}', fontsize=14)
                ax2.set_xlabel('Day', fontsize=12)
                ax2.set_ylabel('Flow Volume', fontsize=12)
                ax2.legend(loc='best', fontsize=10)
                ax2.grid(True, linestyle='--', alpha=0.7)
                
                # 保存图表
                fig_path2 = os.path.join(viz_dir, f'area_{area_j}_to_{area_i}.png')
                plt.tight_layout()
                plt.savefig(fig_path2, dpi=300)
                plt.close(fig2)
                
                print(f"Visualizations for area pair ({area_i}, {area_j}) saved to {viz_dir}")
            
            print(f"All visualizations (10 images) saved to {viz_dir}")
        
        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")
            traceback.print_exc()
        
        print(f"Results saved to {results_file}")
        print(f"Best sample predictions saved to {best_samples_file}")
    
    return {
        'RMSE': rmse,
        'MAE': avg_mae,
        'PCC': pcc
    }

# ========== 主函数 ==========
def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="OD流量预测 - 动态尾部令牌选择大模型方案")
    
    # 数据参数
    parser.add_argument("--io_flow_path", type=str, default="/private/od/data_NYTaxi/io_flow_daily.npy", help="IO流量数据路径")
    parser.add_argument("--graph_path", type=str, default="/private/od/data_NYTaxi/graph.npy", help="图结构数据路径")
    parser.add_argument("--od_matrix_path", type=str, default="/private/od/data_NYTaxi/od_matrix_daily.npy", help="OD矩阵数据路径")
    
    # 模型参数 - 针对纽约数据优化
    parser.add_argument("--model_path", type=str, default="/private/od/Qwen2-7B", help="Qwen模型路径")
    parser.add_argument("--hidden_dim", type=int, default=64, help="隐藏层维度（从128减小到64适应小规模数据）")
    parser.add_argument("--token_dim", type=int, default=768, help="令牌特征维度")
    
    # 训练参数 - 针对纽约数据优化
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小（从32减小到16提高稳定性）")
    parser.add_argument("--epochs", type=int, default=200, help="训练轮数（从100增加到200）")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率（从5e-5增大到1e-4）")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="权重衰减（从1e-6增大到1e-5增强正则化）")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--seed", type=int, default=82, help="随机种子")
    parser.add_argument("--zero_ratio_threshold", type=float, default=0.7, help="稀疏OD对识别阈值，IO流量序列中0值比例超过此值（默认0.7即70%）的OD对将被标记为稀疏并忽略")
    parser.add_argument("--num_workers", type=int, default=0, help="数据加载线程数")
    parser.add_argument("--patience", type=int, default=10, help="早停耐心值")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="/private/od/paper_ny/mine", help="输出目录")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test", "precompute"], help="运行模式：训练、测试或预计算特征")
    parser.add_argument("--load_model", type=str, default=None, help="加载模型路径（用于测试模式）")
    parser.add_argument("--use_precomputed_features", action="store_true", default=True, help="使用预计算的token特征")
    parser.add_argument("--precomputed_features_path", type=str, default="/private/od/data_NYTaxi/token_features/precomputed_token_features_m18.pt", help="预计算token特征路径")
    parser.add_argument("--prompt_type", type=str, default="nyc", choices=["beijing", "nyc"], help="提示词类型：beijing为北京地铁数据，nyc为纽约出租车数据")
    
    args = parser.parse_args()
    
    # 设置随机种子以确保可重复性
    set_seed(args.seed)
    
    # 创建输出目录
    output_dir = create_dynamic_output_dir(args.output_dir)
    args.output_dir = output_dir
    
    # 初始化日志记录器
    global logger, file_logger
    log_path = os.path.join(output_dir, "llm_od_gan.log")
    logger = logging.getLogger(__name__)
    file_logger = FileLogger(log_path)
    logger.info("=== OD流量LLM-GAN模型训练开始 ===")
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    print(f"使用设备: {device}")
    
    # 只预计算特征
    if args.mode == "precompute":
        precomputed_features_path = precompute_token_features(args, device)
        print(f"Token特征预计算完成，保存在: {precomputed_features_path}")
        return
    
    # 加载数据集
    print("加载数据集...")
    logger.info("加载数据集...")
    dataset = ODFlowDataset(
        io_flow_path=args.io_flow_path,
        graph_path=args.graph_path,
        od_matrix_path=args.od_matrix_path,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        zero_ratio_threshold=args.zero_ratio_threshold
    )
    
    # 如果指定使用预计算特征，则加载
    precomputed_features = None
    if args.use_precomputed_features:
        if args.precomputed_features_path and os.path.exists(args.precomputed_features_path):
            print(f"加载预计算的token特征: {args.precomputed_features_path}")
            precomputed_features = torch.load(args.precomputed_features_path)
        else:
            print("未找到预计算的token特征文件，开始预计算...")
            precomputed_features_path = precompute_token_features(args, device)
            precomputed_features = torch.load(precomputed_features_path)
            print(f"加载预计算的token特征: {precomputed_features_path}")
    
    # 创建Qwen2特征提取器（如果不使用预计算特征）
    qwen_extractor = None
    if not args.use_precomputed_features:
        qwen_extractor = QwenFeatureExtractor(
            feature_dim=args.token_dim,
            device=device
        )
    
    # 创建模型
    model = None
    
    # 创建数据加载器
    if args.mode == "train":
        # 训练模式
        dataset.set_mode('train')
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        
        dataset.set_mode('val')
        val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
        try:
            # 训练模型
            print("训练LLM-GAN模型...")
            logger.info("训练LLM-GAN模型...")
            best_model_path = train_llm_gan(args, precomputed_features=precomputed_features)
            
            # 检查模型文件是否存在
            if os.path.exists(best_model_path):
                # 加载训练好的最佳模型
                print(f"加载最佳模型: {best_model_path}")
                checkpoint = torch.load(best_model_path, map_location=device)
                
                # 检测保存模型的判别器数量
                state_dict = checkpoint['model_state_dict']
                # 通过检查state_dict中判别器的数量来确定原始模型结构
                base_disc_keys = [k for k in state_dict.keys() if k.startswith('discriminator_forest.base_discriminators.')]
                if base_disc_keys:
                    # 提取判别器索引号
                    disc_indices = set()
                    for key in base_disc_keys:
                        parts = key.split('.')
                        if len(parts) >= 3 and parts[2].isdigit():
                            disc_indices.add(int(parts[2]))
                    original_n_discriminators = len(disc_indices)
                else:
                    original_n_discriminators = 3  # 默认值
                
                print(f"检测到保存模型的判别器数量: {original_n_discriminators}")
                
                # 根据保存模型的结构创建兼容的模型
                model = ODFlowGenerator_GAN(
                    hidden_dim=checkpoint.get('hidden_dim', args.hidden_dim),
                    token_dim=checkpoint.get('token_dim', args.token_dim),
                    time_steps=28,
                    n_discriminators=original_n_discriminators  # 使用原始模型的判别器数量
                ).to(device)
                
                print(f"创建了兼容的模型结构（{original_n_discriminators}个判别器）")
                
                # 处理token_compressor的键问题并加载权重（非严格模式）
                state_dict = checkpoint['model_state_dict']
                
                # 检查是否有token_compressor相关的键
                token_compressor_keys = [k for k in state_dict.keys() if 'token_compressor' in k]
                if token_compressor_keys:
                    print(f"检测到token_compressor相关键: {len(token_compressor_keys)}个")
                    
                    # 为了避免加载错误，先确保模型结构包含token_compressor
                    # 创建一个初始的token_compressor以接收权重
                    input_dim = 0
                    output_dim = 0
                    for key in token_compressor_keys:
                        if '.0.weight' in key:  # 第一层权重
                            input_dim = state_dict[key].shape[1]
                            output_dim = state_dict[key].shape[0]
                            break
                    
                    if input_dim > 0 and output_dim > 0:
                        print(f"为token_compressor预创建结构: 输入维度={input_dim}, 输出维度={output_dim}")
                        # 在加载权重前先创建相应结构
                        model.discriminator_forest.token_output_dim = output_dim
                        model.discriminator_forest.token_compressor = nn.Sequential(
                            nn.Linear(input_dim, output_dim),
                            nn.LeakyReLU(0.2),
                            nn.LayerNorm(output_dim)
                        ).to(device)
                
                # 加载模型权重（非严格模式，避免无关键冲突）
                try:
                    model.load_state_dict(state_dict, strict=False)
                    print("模型权重加载成功（使用非严格模式）")
                except Exception as e:
                    print(f"警告：模型权重加载时出现部分错误: {str(e)}")
                
                # 打印模型信息
                print(f"模型加载成功，Epoch: {checkpoint.get('epoch', 'unknown')}")
                print(f"RMSE: {checkpoint.get('rmse', 'N/A')}, MAE: {checkpoint.get('mae', 'N/A')}, PCC: {checkpoint.get('pcc', 'N/A')}")
            else:
                print(f"警告：最佳模型文件不存在: {best_model_path}")
                logger.warning(f"最佳模型文件不存在: {best_model_path}")
                
                # 创建一个新模型用于评估
                print("创建新模型用于评估...")
                model = ODFlowGenerator_GAN(
                    hidden_dim=args.hidden_dim,
                    token_dim=args.token_dim,
                    time_steps=28,
                    n_discriminators=3  # 简化为3个判别器
                ).to(device)
        except Exception as e:
            print(f"训练过程中出现错误: {str(e)}")
            logger.error(f"训练过程中出现错误: {str(e)}")
            traceback.print_exc()
            
            # 即使训练出错，也尝试加载最佳模型进行评估
            print("训练出错，但尝试加载可能存在的最佳模型...")
            best_model_path = os.path.join(args.output_dir, 'best_llm_gan_model.pth')
            
            if os.path.exists(best_model_path):
                print(f"发现最佳模型文件，尝试加载: {best_model_path}")
                try:
                    checkpoint = torch.load(best_model_path, map_location=device)
                    
                    # 检测保存模型的判别器数量
                    state_dict = checkpoint['model_state_dict']
                    base_disc_keys = [k for k in state_dict.keys() if k.startswith('discriminator_forest.base_discriminators.')]
                    if base_disc_keys:
                        disc_indices = set()
                        for key in base_disc_keys:
                            parts = key.split('.')
                            if len(parts) >= 3 and parts[2].isdigit():
                                disc_indices.add(int(parts[2]))
                        original_n_discriminators = len(disc_indices)
                    else:
                        original_n_discriminators = 5  # 如果检测失败，假设是旧版本的5个判别器
                    
                    print(f"检测到保存模型的判别器数量: {original_n_discriminators}")
                    
                    # 使用正确的结构创建模型
                    model = ODFlowGenerator_GAN(
                        hidden_dim=checkpoint.get('hidden_dim', args.hidden_dim),
                        token_dim=checkpoint.get('token_dim', args.token_dim),
                        time_steps=28,
                        n_discriminators=original_n_discriminators
                    ).to(device)
                    
                    # 加载权重
                    model.load_state_dict(state_dict, strict=False)
                    print("✅ 成功加载最佳模型进行评估")
                except Exception as load_e:
                    print(f"加载最佳模型失败: {load_e}")
                    # 只有在无法加载最佳模型时才创建新模型
                    print("创建新模型用于评估...")
                    model = ODFlowGenerator_GAN(
                        hidden_dim=args.hidden_dim,
                        token_dim=args.token_dim,
                        time_steps=28,
                        n_discriminators=3  # 简化为3个判别器
                    ).to(device)
            else:
                print("未找到最佳模型文件，创建新模型用于评估...")
                model = ODFlowGenerator_GAN(
                    hidden_dim=args.hidden_dim,
                    token_dim=args.token_dim,
                    time_steps=28,
                    n_discriminators=3  # 简化为3个判别器
                ).to(device)
    else:
        # 测试模式
        if args.load_model is None:
            print("错误：测试模式需要提供 --load_model 参数")
            return
        
        # 检查模型文件是否存在
        if not os.path.exists(args.load_model):
            print(f"错误：指定的模型文件不存在: {args.load_model}")
            return
            
        try:
            # 加载模型
            print(f"加载模型: {args.load_model}")
            checkpoint = torch.load(args.load_model, map_location=device)
            
            # 创建简化的模型 - 测试模式
            model = ODFlowGenerator_GAN(
                hidden_dim=checkpoint.get('hidden_dim', args.hidden_dim),
                token_dim=checkpoint.get('token_dim', args.token_dim),
                time_steps=28,
                n_discriminators=3  # 简化为3个判别器
            ).to(device)
            
            # 加载模型权重前先处理token_compressor
            state_dict = checkpoint['model_state_dict']
            
            # 检查是否有token_compressor相关的键
            token_compressor_keys = [k for k in state_dict.keys() if 'token_compressor' in k]
            if token_compressor_keys:
                print(f"检测到token_compressor相关键: {len(token_compressor_keys)}个")
                
                # 为了避免加载错误，先确保模型结构包含token_compressor
                input_dim = 0
                output_dim = 0
                for key in token_compressor_keys:
                    if '.0.weight' in key:  # 第一层权重
                        input_dim = state_dict[key].shape[1]
                        output_dim = state_dict[key].shape[0]
                        break
                
                if input_dim > 0 and output_dim > 0:
                    print(f"为token_compressor预创建结构: 输入维度={input_dim}, 输出维度={output_dim}")
                    # 在加载权重前先创建相应结构
                    model.discriminator_forest.token_output_dim = output_dim
                    model.discriminator_forest.token_compressor = nn.Sequential(
                        nn.Linear(input_dim, output_dim),
                        nn.LeakyReLU(0.2),
                        nn.LayerNorm(output_dim)
                    ).to(device)
            
            # 加载模型权重
            try:
                model.load_state_dict(state_dict, strict=False)
                print("模型权重加载成功（使用非严格模式）")
            except Exception as e:
                print(f"警告：模型权重加载时出现部分错误: {str(e)}")
            
            # 打印模型信息
            print(f"模型加载成功，Epoch: {checkpoint.get('epoch', 'unknown')}")
            print(f"RMSE: {checkpoint.get('rmse', 'N/A')}, MAE: {checkpoint.get('mae', 'N/A')}, PCC: {checkpoint.get('pcc', 'N/A')}")
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            logger.error(f"加载模型时出错: {str(e)}")
            traceback.print_exc()
            return
    
    # 确保模型已创建
    if model is None:
        print("错误：无法创建或加载模型")
        return
    
    # 设置测试模式
    dataset.set_mode('test')
    
    # 创建测试数据加载器
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    try:
        # 评估模型并生成可视化
        print("评估模型并生成站点对可视化...")
        logger.info("评估模型并生成站点对可视化...")
        metrics = evaluate_llm_gan_model(
            test_loader, 
            qwen_extractor, 
            model, 
            device, 
            args,
            output_dir=output_dir,
            precomputed_features=precomputed_features  # 传递预计算的特征
        )
        
        print(f"测试集评估结果: RMSE={metrics['RMSE']:.6f}, MAE={metrics['MAE']:.6f}, PCC={metrics['PCC']:.6f}")
        logger.info(f"测试集评估结果: RMSE={metrics['RMSE']:.6f}, MAE={metrics['MAE']:.6f}, PCC={metrics['PCC']:.6f}")
    except Exception as e:
        print(f"评估过程中出错: {str(e)}")
        logger.error(f"评估过程中出错: {str(e)}")
        traceback.print_exc()
    
    print("完成!")
    logger.info("=== OD流量LLM-GAN模型训练和评估完成 ===")
    file_logger.close()

if __name__ == "__main__":
    main()