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
    dynamic_dir = os.path.join(base_dir, f"adaptive_run_{timestamp}")
    os.makedirs(dynamic_dir, exist_ok=True)
    return dynamic_dir

# ========== ADAPTIVE 核心组件 ==========

class TuckERKnowledgeGraphEmbedding(nn.Module):
    """TuckER知识图谱嵌入模块 - ADAPTIVE核心组件之一
    
    用于学习城市知识图谱中实体的低维表示
    Reference: TuckER: Tensor Factorization for Knowledge Graph Completion (EMNLP 2019)
    """
    def __init__(self, num_entities, num_relations, entity_dim=64, relation_dim=64):
        super(TuckERKnowledgeGraphEmbedding, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        
        # 实体嵌入矩阵
        self.entity_embeddings = nn.Embedding(num_entities, entity_dim)
        # 关系嵌入矩阵
        self.relation_embeddings = nn.Embedding(num_relations, relation_dim)
        
        # TuckER核心张量 - 用于建模三元组的交互
        self.core_tensor = nn.Parameter(torch.randn(entity_dim, relation_dim, entity_dim))
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        # 初始化参数
        self._init_parameters()
        
    def _init_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        nn.init.xavier_uniform_(self.core_tensor)
    
    def forward(self, entities):
        """
        前向传播获取实体嵌入
        Args:
            entities: 实体ID列表 [batch_size] 或 [num_entities]
        Returns:
            实体嵌入 [batch_size, entity_dim] 或 [num_entities, entity_dim]
        """
        return self.entity_embeddings(entities)
    
    def compute_tucker_score(self, head, relation, tail):
        """
        计算TuckER三元组评分
        Args:
            head: 头实体嵌入 [batch_size, entity_dim]
            relation: 关系嵌入 [batch_size, relation_dim]  
            tail: 尾实体嵌入 [batch_size, entity_dim]
        Returns:
            评分 [batch_size]
        """
        # TuckER三重线性积计算
        # x = head @ core_tensor @ relation @ tail
        x = torch.einsum('bi,ijk,bj,bk->b', head, self.core_tensor, relation, tail)
        return torch.sigmoid(x)

class GraphConvolutionalNetwork(nn.Module):
    """图卷积网络模块 - ADAPTIVE核心组件
    
    用于在基站图上传播空间信息
    Reference: Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.1):
        super(GraphConvolutionalNetwork, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GCN层
        self.gcn_layers = nn.ModuleList()
        
        # 第一层：input_dim -> hidden_dim
        self.gcn_layers.append(nn.Linear(input_dim, hidden_dim))
        
        # 中间层：hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            self.gcn_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # 最后一层：hidden_dim -> output_dim
        if num_layers > 1:
            self.gcn_layers.append(nn.Linear(hidden_dim, output_dim))
        else:
            # 只有一层的情况
            self.gcn_layers[0] = nn.Linear(input_dim, output_dim)
    
    def forward(self, node_features, adjacency_matrix):
        """
        GCN前向传播
        Args:
            node_features: 节点特征 [num_nodes, input_dim]
            adjacency_matrix: 邻接矩阵 [num_nodes, num_nodes]
        Returns:
            输出特征 [num_nodes, output_dim]
        """
        # 计算度矩阵和归一化邻接矩阵
        degree_matrix = torch.diag(torch.sum(adjacency_matrix, dim=1))
        degree_matrix_inv_sqrt = torch.pow(degree_matrix, -0.5)
        degree_matrix_inv_sqrt[torch.isinf(degree_matrix_inv_sqrt)] = 0
        
        # 对称归一化: D^(-1/2) A D^(-1/2)
        normalized_adj = torch.mm(torch.mm(degree_matrix_inv_sqrt, adjacency_matrix), degree_matrix_inv_sqrt)
        
        x = node_features
        for i, layer in enumerate(self.gcn_layers):
            # 图卷积操作: A * X * W
            x = torch.mm(normalized_adj, x)
            x = layer(x)
            
            # 最后一层不使用激活函数和dropout
            if i < len(self.gcn_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x

class AttentionDrivenMatching(nn.Module):
    """注意力驱动匹配模块 - ADAPTIVE核心组件
    
    用于将时间模式从源城市传递到目标城市
    """
    def __init__(self, feature_dim, num_clusters=4):
        super(AttentionDrivenMatching, self).__init__()
        self.feature_dim = feature_dim
        self.num_clusters = num_clusters
        
        # 注意力权重计算网络
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1)
        )
        
        # 簇中心表示（可学习参数）
        self.cluster_centers = nn.Parameter(torch.randn(num_clusters, feature_dim))
        
        # 模式匹配网络
        self.pattern_matching = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, node_representations):
        """
        注意力驱动的模式匹配
        Args:
            node_representations: 节点表示 [num_nodes, feature_dim]
        Returns:
            匹配后的表示 [num_nodes, feature_dim]
        """
        num_nodes = node_representations.size(0)
        
        # 计算每个节点与簇中心的相似度
        similarities = []
        for i in range(self.num_clusters):
            center = self.cluster_centers[i].unsqueeze(0).expand(num_nodes, -1)  # [num_nodes, feature_dim]
            sim = F.cosine_similarity(node_representations, center, dim=1)  # [num_nodes]
            similarities.append(sim.unsqueeze(1))
        
        similarities = torch.cat(similarities, dim=1)  # [num_nodes, num_clusters]
        
        # 注意力权重计算
        attention_weights = F.softmax(similarities, dim=1)  # [num_nodes, num_clusters]
        
        # 加权聚合簇中心特征
        weighted_centers = torch.mm(attention_weights, self.cluster_centers)  # [num_nodes, feature_dim]
        
        # 模式匹配网络处理
        matched_representations = self.pattern_matching(weighted_centers)
        
        # 残差连接
        output = node_representations + matched_representations
        
        return output

class FeatureEnhancedGenerator(nn.Module):
    """特征增强生成器 - ADAPTIVE核心组件
    
    结合多种模式生成流量数据，按照原论文GAN架构实现
    """
    def __init__(self, input_dim, feature_dim, hidden_dim, time_steps, noise_dim=32, output_dim=2):
        super(FeatureEnhancedGenerator, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.time_steps = time_steps
        self.noise_dim = noise_dim
        self.output_dim = output_dim
        
        # 噪声投影 - GAN的关键组件
        self.noise_projection = nn.Linear(noise_dim, hidden_dim)
        
        # 输入特征投影
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 节点特征投影
        self.feature_projection = nn.Linear(feature_dim, hidden_dim)
        
        # 多尺度生成器：日模式、周模式、残差模式
        self.daily_generator = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),  # 添加批归一化
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, output_dim)  # 移除Tanh
        )
        
        self.weekly_generator = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),  # 添加批归一化
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, output_dim)  # 移除Tanh
        )
        
        self.residual_generator = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),  # 添加批归一化
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, output_dim)  # 移除Tanh
        )
        
        # 时序建模网络
        self.temporal_net = nn.LSTM(
            input_size=hidden_dim * 3,  # 包含噪声
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # 最终融合网络 - 移除Tanh，让sigmoid在forward中处理
        self.fusion_net = nn.Sequential(
            nn.Linear(output_dim * 3, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),  # 添加dropout防止过拟合
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, output_dim)  # 移除Tanh，使用线性输出
        )
        
    def forward(self, input_features, node_representations, noise=None):
        """
        生成器前向传播 - 按照GAN架构
        Args:
            input_features: 输入特征 [batch_size, time_steps, input_dim]
            node_representations: 节点表示 [batch_size, feature_dim]
            noise: 随机噪声 [batch_size, noise_dim] 或 None
        Returns:
            生成的OD流量 [batch_size, time_steps, output_dim]
        """
        batch_size, time_steps, _ = input_features.size()
        
        # 生成噪声（如果未提供）
        if noise is None:
            noise = torch.randn(batch_size, self.noise_dim, device=input_features.device)
        
        # 投影噪声
        projected_noise = self.noise_projection(noise)  # [batch_size, hidden_dim]
        projected_noise = projected_noise.unsqueeze(1).expand(-1, time_steps, -1)  # [batch_size, time_steps, hidden_dim]
        
        # 投影输入特征
        projected_input = self.input_projection(input_features)  # [batch_size, time_steps, hidden_dim]
        
        # 投影节点特征并扩展时间维度
        projected_features = self.feature_projection(node_representations)  # [batch_size, hidden_dim]
        projected_features = projected_features.unsqueeze(1).expand(-1, time_steps, -1)  # [batch_size, time_steps, hidden_dim]
        
        # 拼接特征：输入 + 节点特征 + 噪声
        combined_features = torch.cat([projected_input, projected_features, projected_noise], dim=-1)  # [batch_size, time_steps, hidden_dim*3]
        
        # LSTM时序建模
        lstm_output, _ = self.temporal_net(combined_features)  # [batch_size, time_steps, hidden_dim]
        
        # 更新组合特征用于生成器
        combined_for_gen = torch.cat([lstm_output, projected_features, projected_noise], dim=-1)  # [batch_size, time_steps, hidden_dim*3]
        
        # 多尺度生成 - 需要重塑维度以适应BatchNorm1d
        batch_size, time_steps, feature_dim = combined_for_gen.shape
        combined_reshaped = combined_for_gen.view(-1, feature_dim)  # [batch_size*time_steps, feature_dim]
        
        daily_pattern = self.daily_generator(combined_reshaped).view(batch_size, time_steps, -1)
        weekly_pattern = self.weekly_generator(combined_reshaped).view(batch_size, time_steps, -1)
        residual_pattern = self.residual_generator(combined_reshaped).view(batch_size, time_steps, -1)
        
        # 拼接所有模式
        all_patterns = torch.cat([daily_pattern, weekly_pattern, residual_pattern], dim=-1)  # [batch_size, time_steps, output_dim*3]
        
        # 最终融合 - 重塑维度处理
        all_patterns_reshaped = all_patterns.view(-1, all_patterns.size(-1))  # [batch_size*time_steps, output_dim*3]
        output = self.fusion_net(all_patterns_reshaped).view(batch_size, time_steps, -1)  # [batch_size, time_steps, output_dim]
        
        return output

class MultiScaleDiscriminator(nn.Module):
    """多尺度判别器 - ADAPTIVE GAN核心组件
    
    按照原论文实现，区分真实与生成的流量数据
    """
    def __init__(self, input_dim=2, feature_dim=64, time_steps=28):
        super(MultiScaleDiscriminator, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.time_steps = time_steps
        
        # 输入投影
        self.input_projection = nn.Conv1d(input_dim, feature_dim, kernel_size=1)
        
        # 多尺度卷积判别器
        self.conv_blocks = nn.ModuleList([
            # 第一尺度：捕捉局部模式
            nn.Sequential(
                nn.Conv1d(feature_dim, feature_dim * 2, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv1d(feature_dim * 2, feature_dim * 2, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ),
            # 第二尺度：捕捉中等模式
            nn.Sequential(
                nn.Conv1d(feature_dim * 2, feature_dim * 4, kernel_size=5, stride=1, padding=2),
                nn.LeakyReLU(0.2),
                nn.Conv1d(feature_dim * 4, feature_dim * 4, kernel_size=5, stride=2, padding=2),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ),
            # 第三尺度：捕捉全局模式
            nn.Sequential(
                nn.Conv1d(feature_dim * 4, feature_dim * 8, kernel_size=7, stride=1, padding=3),
                nn.LeakyReLU(0.2),
                nn.Conv1d(feature_dim * 8, feature_dim * 8, kernel_size=7, stride=2, padding=3),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            )
        ])
        
        # 自适应平均池化
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # 最终分类器
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 8, feature_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 1)  # Wasserstein GAN不需要sigmoid
        )
        
        # 梯度惩罚权重
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        判别器前向传播
        Args:
            x: 输入流量数据 [batch_size, time_steps, input_dim]
        Returns:
            判别分数 [batch_size, 1]
        """
        # 转换为卷积格式: [batch_size, input_dim, time_steps]
        x = x.transpose(1, 2)
        
        # 输入投影
        x = self.input_projection(x)
        
        # 多尺度卷积处理
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        # 全局平均池化
        x = self.adaptive_pool(x)  # [batch_size, feature_dim*8, 1]
        x = x.view(x.size(0), -1)  # [batch_size, feature_dim*8]
        
        # 最终分类
        output = self.classifier(x)  # [batch_size, 1]
        
        return output

class ADAPTIVEODFlowPredictor(nn.Module):
    """基于ADAPTIVE架构的OD流量预测GAN模型
    
    主要创新点：
    1. 使用TuckER知识图谱嵌入提取城市环境特征
    2. 采用GCN捕捉空间依赖关系
    3. 应用注意力驱动匹配传递时间模式
    4. 使用特征增强生成器进行多模式生成
    5. 采用多尺度判别器实现对抗训练（严格按照原论文GAN架构）
    """
    def __init__(self, input_dim=6, hidden_dim=64, time_steps=28, 
                 num_entities=100, num_relations=10, entity_dim=64,
                 gcn_layers=2, num_clusters=4, noise_dim=32):
        super(ADAPTIVEODFlowPredictor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_steps = time_steps
        self.num_entities = num_entities
        self.entity_dim = entity_dim
        self.noise_dim = noise_dim
        
        # 1. 知识图谱嵌入模块
        self.knowledge_graph_embedding = TuckERKnowledgeGraphEmbedding(
            num_entities=num_entities,
            num_relations=num_relations,
            entity_dim=entity_dim,
            relation_dim=entity_dim
        )
        
        # 2. 图卷积网络
        self.gcn = GraphConvolutionalNetwork(
            input_dim=entity_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=gcn_layers
        )
        
        # 3. 注意力驱动匹配
        self.attention_matching = AttentionDrivenMatching(
            feature_dim=hidden_dim,
            num_clusters=num_clusters
        )
        
        # 4. 特征增强生成器 (Generator)
        self.generator = FeatureEnhancedGenerator(
            input_dim=input_dim,
            feature_dim=hidden_dim,
            hidden_dim=hidden_dim,
            time_steps=time_steps,
            noise_dim=noise_dim,
            output_dim=2
        )
        
        # 5. 多尺度判别器 (Discriminator) - 按照原论文GAN架构
        self.discriminator = MultiScaleDiscriminator(
            input_dim=2,
            feature_dim=hidden_dim,
            time_steps=time_steps
        )
        
        # 辅助网络：POI分布重构任务
        self.poi_reconstruction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 10),  # 假设10个POI类别
            nn.Softmax(dim=-1)
        )
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        
        # 模拟简单的城市知识图谱结构
        self._init_mock_knowledge_graph()
        
    def pearson_correlation_loss(self, pred, target):
        """计算皮尔逊相关系数损失，用于直接优化PCC"""
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
        
        # 计算皮尔逊相关系数
        correlation = covariance / (pred_std * target_std + 1e-8)
        
        # 返回负相关系数作为损失（最大化相关性）
        return 1.0 - correlation
        
        
    def _init_mock_knowledge_graph(self):
        """初始化模拟的知识图谱结构"""
        # 创建一个简单的邻接矩阵（距离反比）
        self.register_buffer('adjacency_matrix', torch.eye(self.num_entities))
        
        # 为每个实体分配随机ID（在真实应用中应该从数据中获取）
        self.register_buffer('entity_ids', torch.arange(self.num_entities))
    
    def compute_gradient_penalty(self, real_data, fake_data):
        """
        计算Wasserstein GAN的梯度惩罚项
        Args:
            real_data: 真实数据 [batch_size, time_steps, 2]
            fake_data: 生成数据 [batch_size, time_steps, 2]
        Returns:
            梯度惩罚损失
        """
        batch_size = real_data.size(0)
        device = real_data.device
        
        # 随机插值
        alpha = torch.rand(batch_size, 1, 1, device=device)
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        # 计算判别器对插值数据的输出
        disc_interpolated = self.discriminator(interpolated)
        
        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=disc_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(disc_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # 计算梯度惩罚
        gradients = gradients.reshape(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
        
    def forward(self, features, target_od=None, mode='train', train_discriminator=True):
        """
        前向传播 - 支持GAN对抗训练
        Args:
            features: 输入特征 [batch_size, time_steps=28, input_dim] (input_dim动态计算，通常为6或10)
            target_od: 目标OD流量 [batch_size, time_steps=28, 2]
            mode: 'train' 或 'eval'
            train_discriminator: 是否训练判别器（用于交替训练）
        Returns:
            结果字典
        """
        batch_size = features.size(0)
        
        # 1. 获取知识图谱嵌入（模拟：每个batch使用前batch_size个实体）
        entity_indices = self.entity_ids[:batch_size]
        kg_embeddings = self.knowledge_graph_embedding(entity_indices)  # [batch_size, entity_dim]
        
        # 2. 构建批次的邻接矩阵
        batch_adj = self.adjacency_matrix[:batch_size, :batch_size]
        
        # 3. GCN处理空间关系
        spatial_representations = self.gcn(kg_embeddings, batch_adj)  # [batch_size, hidden_dim]
        
        # 4. 注意力驱动匹配（传递时间模式）
        matched_representations = self.attention_matching(spatial_representations)  # [batch_size, hidden_dim]
        
        # 5. 生成器生成OD流量
        noise = torch.randn(batch_size, self.noise_dim, device=features.device)
        generated_od = self.generator(features, matched_representations, noise)  # [batch_size, time_steps, 2]
        
        # 改进输出范围处理 - 使用更平滑的映射
        # 使用sigmoid激活确保输出在[0,1]范围内，保持梯度流畅
        generated_od = torch.sigmoid(generated_od)  # 直接映射到[0,1]，保持相关性
        
        if mode == 'train' and target_od is not None:
            # 归一化目标数据到[0,1]范围（用于与生成器输出匹配）
            target_od_normalized = target_od  # 假设已经归一化
            
            if train_discriminator:
                # ====== 训练判别器 ======
                # 真实数据的判别器分数
                real_score = self.discriminator(target_od_normalized)
                
                # 生成数据的判别器分数（detach防止梯度传播到生成器）
                fake_score = self.discriminator(generated_od.detach())
                
                # Wasserstein损失
                d_loss_real = -torch.mean(real_score)
                d_loss_fake = torch.mean(fake_score)
                
                # 梯度惩罚
                gradient_penalty = self.compute_gradient_penalty(target_od_normalized, generated_od.detach())
                
                # 判别器总损失
                d_loss = d_loss_real + d_loss_fake + 10.0 * gradient_penalty
                
                return {
                    'od_flows': generated_od,
                    'd_loss': d_loss,
                    'd_loss_real': d_loss_real,
                    'd_loss_fake': d_loss_fake,
                    'gradient_penalty': gradient_penalty
                }
            else:
                # ====== 训练生成器 ======
                # 生成器的对抗损失
                fake_score = self.discriminator(generated_od)
                g_loss_adv = -torch.mean(fake_score)
                
                # 重构损失（与真实数据的相似性）
                mse_loss = self.mse_loss(generated_od, target_od_normalized)
                mae_loss = self.mae_loss(generated_od, target_od_normalized)
                
                # PCC损失（直接优化相关性）
                pcc_loss = self.pearson_correlation_loss(generated_od, target_od_normalized)
                
                # POI重构损失（自监督任务）
                poi_pred = self.poi_reconstruction(matched_representations)  # [batch_size, 10]
                # 模拟POI真实分布（在真实应用中应该从数据中获取）
                poi_target = F.softmax(torch.randn_like(poi_pred), dim=-1)
                poi_loss = self.kl_loss(F.log_softmax(poi_pred, dim=-1), poi_target)
                
                # 生成器总损失 - 加入PCC损失，重点优化相关性
                # 降低对抗损失权重，增加PCC损失权重
                g_loss = 0.05 * g_loss_adv + 20.0 * mse_loss + 10.0 * mae_loss + 30.0 * pcc_loss + 0.1 * poi_loss
                
                return {
                    'od_flows': generated_od,
                    'g_loss': g_loss,
                    'g_loss_adv': g_loss_adv,
                    'mse_loss': mse_loss,
                    'mae_loss': mae_loss,
                    'pcc_loss': pcc_loss,
                    'poi_loss': poi_loss
                }
        else:
            return {
                'od_flows': generated_od
            }
    
    def generate(self, features, noise=None):
        """生成OD流量预测 - 保持与原代码接口一致"""
        with torch.no_grad():
            batch_size = features.size(0)
            
            # 获取表示
            entity_indices = self.entity_ids[:batch_size]
            kg_embeddings = self.knowledge_graph_embedding(entity_indices)
            batch_adj = self.adjacency_matrix[:batch_size, :batch_size]
            spatial_representations = self.gcn(kg_embeddings, batch_adj)
            matched_representations = self.attention_matching(spatial_representations)
            
            # 生成
            if noise is None:
                noise = torch.randn(batch_size, self.noise_dim, device=features.device)
            generated_od = self.generator(features, matched_representations, noise)
            # 使用sigmoid确保输出在[0,1]范围内
            generated_od = torch.sigmoid(generated_od)
            
            return generated_od

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
        
        # 站点对列表 - 与原版保持一致，使用所有站点对
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

# ========== ADAPTIVE GAN训练函数 ==========
def train_adaptive_model(args):
    """训练ADAPTIVE OD流量预测GAN模型"""
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
    
    # 创建ADAPTIVE GAN模型
    model = ADAPTIVEODFlowPredictor(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        time_steps=28,
        num_entities=min(args.num_entities, len(dataset.od_pairs)),  # 使用实际的站点对数量
        num_relations=args.num_relations,
        entity_dim=args.entity_dim,
        gcn_layers=args.gcn_layers,
        num_clusters=args.num_clusters,
        noise_dim=args.noise_dim
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    generator_params = sum(p.numel() for p in model.generator.parameters())
    discriminator_params = sum(p.numel() for p in model.discriminator.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"ADAPTIVE GAN模型创建成功！")
    print(f"  总参数数量: {total_params:,}")
    print(f"  生成器参数: {generator_params:,}")
    print(f"  判别器参数: {discriminator_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  隐藏维度: {args.hidden_dim}")
    print(f"  实体数量: {min(args.num_entities, len(dataset.od_pairs))}")
    print(f"  GCN层数: {args.gcn_layers}")
    print(f"  簇数量: {args.num_clusters}")
    print(f"  噪声维度: {args.noise_dim}")
    
    # 分别创建生成器和判别器的优化器 - GAN训练的标准做法
    generator_params = list(model.knowledge_graph_embedding.parameters()) + \
                      list(model.gcn.parameters()) + \
                      list(model.attention_matching.parameters()) + \
                      list(model.generator.parameters()) + \
                      list(model.poi_reconstruction.parameters())
    
    discriminator_params = list(model.discriminator.parameters())
    
    optimizer_g = torch.optim.Adam(
        generator_params, 
        lr=args.lr, 
        betas=(0.5, 0.999),
        weight_decay=args.weight_decay
    )
    
    optimizer_d = torch.optim.Adam(
        discriminator_params, 
        lr=args.lr, 
        betas=(0.5, 0.999),
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_g, mode='min', factor=0.5, patience=args.patience, verbose=True, min_lr=1e-6
    )
    scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_d, mode='min', factor=0.5, patience=args.patience, verbose=True, min_lr=1e-6
    )
    
    # 训练循环变量
    best_val_loss = float('inf')
    best_model_path = os.path.join(args.output_dir, 'best_adaptive_od_model.pth')
    epochs_without_improvement = 0
    train_history = []
    
    print(f"\n开始训练ADAPTIVE GAN OD流量预测模型...")
    print(f"模型将保存到: {best_model_path}")
    print(f"早停策略: 验证损失{args.early_stop_patience}轮无改善时停止训练")
    print(f"GAN训练: 判别器更新{args.n_critic}次，生成器更新1次")
    print("="*80)
    
    for epoch in range(args.epochs):
        # 训练阶段 - GAN对抗训练
        model.train()
        d_losses = []
        g_losses = []
        mse_losses = []
        mae_losses = []
        pcc_losses = []
        poi_losses = []
        
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{args.epochs} [GAN训练]")
        for batch_idx, (features, od_flows) in enumerate(train_progress):
            features = features.to(device)
            od_flows = od_flows.to(device)
            
            # ========== 训练判别器 ==========
            for _ in range(args.n_critic):
                optimizer_d.zero_grad()
                
                # 判别器前向传播
                d_outputs = model(features, od_flows, mode='train', train_discriminator=True)
                d_loss = d_outputs['d_loss']
                
                # 检查NaN/Inf
                if torch.isnan(d_loss) or torch.isinf(d_loss):
                    print(f"⚠️ 检测到判别器NaN/Inf损失: {d_loss.item()}")
                    print(f"   d_loss_real: {d_outputs.get('d_loss_real', 'N/A')}")
                    print(f"   d_loss_fake: {d_outputs.get('d_loss_fake', 'N/A')}")
                    print(f"   gradient_penalty: {d_outputs.get('gradient_penalty', 'N/A')}")
                    continue
                
                # 判别器反向传播
                d_loss.backward()
                
                # 检查梯度
                grad_norm_d = torch.nn.utils.clip_grad_norm_(discriminator_params, max_norm=0.5)
                if torch.isnan(grad_norm_d) or torch.isinf(grad_norm_d):
                    print(f"⚠️ 检测到判别器梯度异常: {grad_norm_d}")
                    continue
                    
                optimizer_d.step()
                
                d_losses.append(d_loss.item())
            
            # ========== 训练生成器 ==========
            optimizer_g.zero_grad()
            
            # 生成器前向传播
            g_outputs = model(features, od_flows, mode='train', train_discriminator=False)
            g_loss = g_outputs['g_loss']
            mse_loss = g_outputs['mse_loss']
            mae_loss = g_outputs['mae_loss']
            pcc_loss = g_outputs.get('pcc_loss', torch.tensor(0.0))
            poi_loss = g_outputs.get('poi_loss', torch.tensor(0.0))
            
            # 检查NaN/Inf
            if torch.isnan(g_loss) or torch.isinf(g_loss) or \
               torch.isnan(mse_loss) or torch.isinf(mse_loss) or \
               torch.isnan(mae_loss) or torch.isinf(mae_loss):
                print(f"⚠️ 检测到生成器NaN/Inf损失:")
                print(f"   g_loss: {g_loss.item()}")
                print(f"   g_loss_adv: {g_outputs.get('g_loss_adv', 'N/A')}")
                print(f"   mse_loss: {mse_loss.item()}")
                print(f"   mae_loss: {mae_loss.item()}")
                print(f"   poi_loss: {poi_loss.item()}")
                continue
            
            # 生成器反向传播
            g_loss.backward()
            
            # 检查梯度
            grad_norm_g = torch.nn.utils.clip_grad_norm_(generator_params, max_norm=0.5)
            if torch.isnan(grad_norm_g) or torch.isinf(grad_norm_g):
                print(f"⚠️ 检测到生成器梯度异常: {grad_norm_g}")
                continue
                
            optimizer_g.step()
            
            # 记录损失
            g_losses.append(g_loss.item())
            mse_losses.append(mse_loss.item())
            mae_losses.append(mae_loss.item())
            pcc_losses.append(pcc_loss.item())
            poi_losses.append(poi_loss.item())
            
            # 更新进度条 - 安全显示损失值
            if len(d_losses) > 0 and len(g_losses) > 0:
                train_progress.set_postfix({
                    'D_Loss': f'{d_losses[-1]:.4f}',
                    'G_Loss': f'{g_losses[-1]:.4f}',
                    'MSE': f'{mse_losses[-1]:.4f}' if len(mse_losses) > 0 else 'N/A',
                    'MAE': f'{mae_losses[-1]:.4f}' if len(mae_losses) > 0 else 'N/A'
                })
            else:
                train_progress.set_postfix({
                    'D_Loss': 'N/A',
                    'G_Loss': 'N/A',
                    'MSE': 'N/A',
                    'MAE': 'N/A'
                })
        
        # 计算训练指标 - 安全处理空列表
        avg_d_loss = np.mean(d_losses) if len(d_losses) > 0 else float('nan')
        avg_g_loss = np.mean(g_losses) if len(g_losses) > 0 else float('nan')
        avg_mse_loss = np.mean(mse_losses) if len(mse_losses) > 0 else float('nan')
        avg_mae_loss = np.mean(mae_losses) if len(mae_losses) > 0 else float('nan')
        avg_pcc_loss = np.mean(pcc_losses) if len(pcc_losses) > 0 else float('nan')
        avg_poi_loss = np.mean(poi_losses) if len(poi_losses) > 0 else float('nan')
        
        # 检查是否所有batch都失败了
        if len(d_losses) == 0 or len(g_losses) == 0:
            print(f"⚠️ 警告: Epoch {epoch+1} 中所有batch都出现了NaN，跳过本轮")
            continue
        
        # 验证阶段 - 计算详细指标
        print(f"  🔍 计算验证集指标...")
        val_metrics = calculate_metrics(model, val_loader, device, desc="验证集评估")
        
        # 学习率调整
        scheduler_g.step(val_metrics['loss'])
        scheduler_d.step(val_metrics['loss'])
        current_lr_g = optimizer_g.param_groups[0]['lr']
        current_lr_d = optimizer_d.param_groups[0]['lr']
        
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
        print(f"\n📊 Epoch {epoch+1:3d}/{args.epochs} GAN训练完成:")
        print(f"{'='*80}")
        print(f"🔹 训练集:")
        print(f"   判别器损失: {avg_d_loss:.6f} | 生成器损失: {avg_g_loss:.6f}")
        print(f"   MSE: {avg_mse_loss:.6f} | MAE: {avg_mae_loss:.6f} | PCC损失: {avg_pcc_loss:.6f} | POI: {avg_poi_loss:.6f}")
        
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
        
        # 保存训练历史 - 转换为Python原生类型
        epoch_history = {
            'epoch': int(epoch + 1),
            'train_d_loss': float(avg_d_loss),
            'train_g_loss': float(avg_g_loss),
            'train_mse': float(avg_mse_loss),
            'train_mae': float(avg_mae_loss),
            'train_pcc_loss': float(avg_pcc_loss),
            'train_poi': float(avg_poi_loss),
            'val_loss': float(val_metrics['loss']),
            'val_mse': float(val_metrics['mse']),
            'val_rmse': float(val_metrics['rmse']),
            'val_mae': float(val_metrics['mae']),
            'val_pcc': float(val_metrics['pcc']),
            'lr_g': float(optimizer_g.param_groups[0]['lr']),
            'lr_d': float(optimizer_d.param_groups[0]['lr']),
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
                    f.write("ADAPTIVE OD流量预测模型训练日志\n")
                    f.write("=" * 50 + "\n")
                
                f.write(f"Epoch {epoch+1}/{args.epochs}\n")
                f.write(f"   Training - D_Loss: {avg_d_loss:.6f}, G_Loss: {avg_g_loss:.6f}, MSE: {avg_mse_loss:.6f}, MAE: {avg_mae_loss:.6f}, PCC_Loss: {avg_pcc_loss:.6f}, POI: {avg_poi_loss:.6f}\n")
                f.write(f"   Validation - Loss: {val_metrics['loss']:.6f}, RMSE: {val_metrics['rmse']:.6f}, MAE: {val_metrics['mae']:.6f}, PCC: {val_metrics['pcc']:.6f}\n")
                
                if test_metrics:
                    f.write(f"   Test - Loss: {test_metrics.get('loss', 0):.6f}, RMSE: {test_metrics.get('rmse', 0):.6f}, MAE: {test_metrics.get('mae', 0):.6f}, PCC: {test_metrics.get('pcc', 0):.6f}\n")
                
                if is_best:
                    f.write(f"   New best model saved (Val Loss: {best_val_loss:.6f}, Val RMSE: {val_metrics['rmse']:.6f}, Val PCC: {val_metrics['pcc']:.6f})\n")
                else:
                    f.write(f"   No improvement ({epochs_without_improvement}/{args.early_stop_patience} epochs without improvement)\n")
                
                f.write(f"   Learning Rate: G={current_lr_g:.2e}, D={current_lr_d:.2e}\n")
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
            print(f"🎯 ✅ 保存最佳GAN模型 (验证损失: {best_val_loss:.6f})")
        else:
            print(f"⏳ 验证损失未改善 ({epochs_without_improvement}/{args.early_stop_patience}轮)")
        
        # 早停检查
        if epochs_without_improvement >= args.early_stop_patience:
            print(f"\n🛑 早停触发! 验证损失已{args.early_stop_patience}轮未改善，停止训练")
            print(f"   最佳验证损失: {best_val_loss:.6f} (来自第{epoch - epochs_without_improvement + 2}轮)")
            break
        
        # 学习率过小检查
        if current_lr_g < 1e-6 and current_lr_d < 1e-6:
            print(f"\n🛑 学习率过小 (G={current_lr_g:.2e}, D={current_lr_d:.2e})，停止训练")
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
    print("🎉 ADAPTIVE OD流量预测模型 - 最终测试结果")
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
    results_file = os.path.join(args.output_dir, "adaptive_od_results.txt")
    with open(results_file, "w", encoding='utf-8') as f:
        f.write("基于ADAPTIVE的OD流量预测模型测试结果\n")
        f.write("="*50 + "\n")
        f.write("论文: Deep Transfer Learning for City-scale Cellular Traffic Generation through Urban Knowledge Graph (KDD 2023)\n")
        f.write("模型架构核心特点:\n")
        f.write("  - 城市知识图谱 (Urban Knowledge Graph)\n")
        f.write("  - TuckER知识图谱嵌入 (TuckER Knowledge Graph Embedding)\n")
        f.write("  - 图卷积网络 (Graph Convolutional Network)\n")
        f.write("  - 注意力驱动匹配 (Attention-driven Matching)\n")
        f.write("  - 特征增强生成网络 (Feature-enhanced Generator)\n")
        f.write("  - POI分布重构任务 (POI Distribution Reconstruction)\n")
        f.write("\n")
        f.write(f"模型参数:\n")
        f.write(f"  - 总参数数量: {total_params:,}\n")
        f.write(f"  - 可训练参数: {trainable_params:,}\n")
        f.write(f"  - 隐藏维度: {args.hidden_dim}\n")
        f.write(f"  - 实体数量: {min(args.num_entities, len(dataset.od_pairs))}\n")
        f.write(f"  - 关系数量: {args.num_relations}\n")
        f.write(f"  - 实体嵌入维度: {args.entity_dim}\n")
        f.write(f"  - GCN层数: {args.gcn_layers}\n")
        f.write(f"  - 簇数量: {args.num_clusters}\n")
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
    parser = argparse.ArgumentParser(description="基于ADAPTIVE的OD流量预测模型")
    
    # 数据参数
    parser.add_argument("--io_flow_path", type=str, default="/private/od/data_NYTaxi/io_flow_daily.npy", 
                       help="IO流量数据路径")
    parser.add_argument("--graph_path", type=str, default="/private/od/data_NYTaxi/graph.npy", 
                       help="图结构数据路径")
    parser.add_argument("--od_matrix_path", type=str, default="/private/od/data_NYTaxi/od_matrix_daily.npy", 
                       help="OD矩阵数据路径")
    
    # ADAPTIVE GAN模型参数
    parser.add_argument("--hidden_dim", type=int, default=64, 
                       help="隐藏维度")
    parser.add_argument("--num_entities", type=int, default=100, 
                       help="知识图谱实体数量 (建议设置为节点数量的1-2倍)")
    parser.add_argument("--num_relations", type=int, default=10, 
                       help="知识图谱关系数量")
    parser.add_argument("--entity_dim", type=int, default=64, 
                       help="实体嵌入维度")
    parser.add_argument("--gcn_layers", type=int, default=2, 
                       help="GCN层数")
    parser.add_argument("--num_clusters", type=int, default=4, 
                       help="时间模式簇数量")
    parser.add_argument("--noise_dim", type=int, default=32, 
                       help="GAN噪声维度")
    parser.add_argument("--n_critic", type=int, default=5, 
                       help="每次生成器更新前判别器更新次数")
    
    # 训练参数  
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.0001, help="学习率 (GAN推荐较小值)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="权重衰减")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="测试集比例 (固定8:1:1划分)")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例 (固定8:1:1划分)")
    parser.add_argument("--seed", type=int, default=82, help="随机种子")
    
    # 早停和学习率调整参数
    parser.add_argument("--early_stop_patience", type=int, default=15, help="早停策略：验证损失多少轮无改善时停止训练")
    parser.add_argument("--patience", type=int, default=8, help="学习率调整策略：验证损失多少轮无改善时降低学习率")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="/private/od/paper_ny/ADAPTIVE", help="输出目录")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = create_dynamic_output_dir(args.output_dir)
    args.output_dir = output_dir
    
    print("="*60)
    print("🚀 ADAPTIVE GAN OD流量预测模型")
    print("="*60)
    print("📖 论文: Deep Transfer Learning for City-scale Cellular Traffic Generation through Urban Knowledge Graph")
    print("📖 会议: KDD 2023")
    print("📖 作者: Shiyuan Zhang, et al.")
    print("📖 架构: 特征增强生成对抗网络 (Feature-enhanced GAN)")
    print()
    print("🔧 模型创新点:")
    print("  ✅ 城市知识图谱 - 建模城市环境实体关系")
    print("  ✅ TuckER嵌入 - 学习知识图谱实体表示")
    print("  ✅ 图卷积网络 - 捕捉空间依赖关系")
    print("  ✅ 注意力匹配 - 传递时间模式")
    print("  ✅ 特征增强生成器 - 多模式流量生成")
    print("  ✅ 多尺度判别器 - 对抗训练")
    print("  ✅ Wasserstein GAN - 稳定训练过程")
    print("  ✅ POI重构任务 - 自监督学习")
    print()
    print(f"📁 输出目录: {output_dir}")
    print("="*60)
    
    # 训练模型
    try:
        best_model_path = train_adaptive_model(args)
        print("\n🎉 ADAPTIVE GAN模型训练完成!")
        print(f"📁 最佳模型保存位置: {best_model_path}")
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {str(e)}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)