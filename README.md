# OD生成项目

## 目录结构

```
.
├── CADF_OD.py                          # 主实验代码
├── precompute_token_features_m18_NY.py # 语义特征计算脚本（运行主实验前需先执行）
├── data_NYTaxi/                        # 纽约出租车数据集
│   ├── od_matrix_daily.npy            # OD矩阵数据
│   ├── io_flow_daily.npy              # IO流量数据
│   ├── graph.npy                      # 距离矩阵
│   └── grid_population_density_52nodes.json  # 区域人口密度数据
└── baselines/                          # 基线模型训练代码
    ├── ADAPTIVE/
    ├── DT_VAE/
    ├── Flow_VAE/
    ├── ForestGAN/
    ├── IVP_VAE/
    ├── LMGU_DDPM/
    ├── MCGAN/
    ├── PSA_GAN/
    ├── SeNM_VAE/
    ├── Spd_DDPM/
    └── TimeGAN/
```

## 使用说明

1. 下载语义特征嵌入模型（可使用任意语义嵌入模型，如Qwen2-7B-embed-base、Word2Vec等）

2. 运行语义特征计算脚本：
   ```
   python precompute_token_features_m18_NY.py
   ```

3. 运行主实验：
   ```
   python CADF_OD.py
   ```

## 数据集

`data_NYTaxi/` 目录包含纽约出租车相关数据：
- OD矩阵（起点-终点流量）
- IO流量（流入-流出）
- 距离矩阵
- 区域人口密度

## 基线模型

`baselines/` 目录包含多个对比基线模型的实现代码。
