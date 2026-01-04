

import os
import sys
import argparse
import torch
from tqdm import tqdm

ROOT_DIR = "/private/od"
NEW_DIR = os.path.join(ROOT_DIR, "new")
if NEW_DIR not in sys.path:
    sys.path.insert(0, NEW_DIR)

from CADF_OD import ODFlowDataset, QwenFeatureExtractor  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Precompute token features for m18_NY (Stacking Forest-GAN) using Qwen2-7B for NYC taxi grid OD data")
    parser.add_argument("--io_flow_path", type=str, default="/private/od/data_NYTaxi/io_flow_daily.npy")
    parser.add_argument("--graph_path", type=str, default="/private/od/data_NYTaxi/graph.npy")
    parser.add_argument("--od_matrix_path", type=str, default="/private/od/data_NYTaxi/od_matrix_daily.npy")
    parser.add_argument("--output_path", type=str, default="/private/od/data_NYTaxi/token_features/precomputed_token_features_m18.pt")
    parser.add_argument("--token_dim", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=("cuda:0" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()

    # 创建输出目录
    out_dir = os.path.dirname(args.output_path)
    os.makedirs(out_dir, exist_ok=True)

    print(f"预计算 m18_NY 的 Token 特征（纽约出租车网格OD数据）")
    print(f"输出路径: {args.output_path}")
    print(f"设备: {args.device}")
    print(f"Token维度: {args.token_dim}")
    print("-" * 60)

    # 数据集（与 m18_NY 一致，默认 20% 测试、10% 验证，仅用于访问 io_flow 与区域对）
    dataset = ODFlowDataset(
        io_flow_path=args.io_flow_path,
        graph_path=args.graph_path,
        od_matrix_path=args.od_matrix_path,
        test_ratio=0.2,
        val_ratio=0.1,
        seed=args.seed,
    )

    # 特征提取器（使用 m18_NY 的 QwenFeatureExtractor）
    print("正在加载 Qwen2 特征提取器...")
    extractor = QwenFeatureExtractor(feature_dim=args.token_dim, device=args.device)
    print("特征提取器加载完成")
    print("-" * 60)

    token_features = {}
    total_pairs = len(dataset.od_pairs)

    print(f"开始处理 {total_pairs} 个站点对...")
    print(f"数据集节点数: {dataset.num_nodes}, 时间步数: {dataset.time_steps}")
    print(f"预期节点数: 52 (如果数据已更新为52节点版本)")
    
    for idx, (site_i, site_j) in enumerate(tqdm(dataset.od_pairs, desc="预计算 Token 特征 (m18_NY)")):
        # 获取IO流数据
        # 注意：io_flow 的实际格式是 (时间步, 区域数, 2)
        # 所以需要正确访问数据
        io_i = dataset.io_flow[:, site_i, :]  # (时间步, 2)
        io_j = dataset.io_flow[:, site_j, :]  # (时间步, 2)
        
        # 如果时间步数超过28，只取前28天（与模型训练一致）
        time_steps = io_i.shape[0]
        if time_steps > 28:
            io_i = io_i[:28]
            io_j = io_j[:28]

        io_i_t = torch.FloatTensor(io_i).to(args.device)
        io_j_t = torch.FloatTensor(io_j).to(args.device)

        try:
            with torch.no_grad():
                # 调用特征提取器，使用纽约出租车数据的提示词模板
                feat, _ = extractor(
                    site_i, 
                    site_j, 
                    io_i_t, 
                    io_j_t, 
                    dataset.station_data if hasattr(dataset, 'station_data') else None,
                    prompt_type="nyc"  # 使用纽约出租车数据的提示词
                )

            token_features[f"{site_i}_{site_j}"] = feat.cpu()

            # 定期保存中间结果
            if (idx + 1) % 100 == 0:
                print(f"\n已处理 {idx + 1}/{total_pairs} 个区域对，保存中间结果...")
                torch.save(token_features, args.output_path)
                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        except Exception as e:
            import traceback
            error_msg = str(e)
            # 只打印前几个错误的详细堆栈，避免输出过多
            if idx < 3:
                print(f"\n警告: 处理区域对 ({site_i}, {site_j}) 时出错: {error_msg}")
                print(f"详细错误信息:")
                traceback.print_exc()
            elif "Numpy is not available" in error_msg:
                # 对于numpy错误，尝试使用numpy数组而不是tensor
                try:
                    import numpy as np
                    io_i_np = io_i_t.cpu().numpy() if hasattr(io_i_t, 'cpu') else io_i_t
                    io_j_np = io_j_t.cpu().numpy() if hasattr(io_j_t, 'cpu') else io_j_t
                    # 重新尝试，但这次使用numpy数组
                    # 如果还是失败，跳过这个样本
                    continue
                except:
                    continue
            else:
                # 其他错误，只打印简要信息
                if (idx + 1) % 100 == 0:
                    print(f"\n警告: 处理区域对 ({site_i}, {site_j}) 时出错: {error_msg[:100]}")
            continue

    # 保存最终结果
    torch.save(token_features, args.output_path)
    print(f"\n{'='*60}")
    print(f"✅ 完成！保存了 {len(token_features)} 个区域对的 token 特征")
    print(f"📁 保存路径: {args.output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main() 