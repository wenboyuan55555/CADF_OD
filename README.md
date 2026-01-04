# CADF-OD: Cross-Modal Alignment with Ensemble Adversarial Learning for OD Flow Generation

This repository contains the official implementation of **CADF-OD**, a novel framework for Origin-Destination (OD) flow generation.

---

## Abstract

Origin-Destination flow prediction has become increasingly vital for urban transportation planning and traffic management. **CADF-OD** leverages a cross-modal alignment mechanism bridging semantic representations from pre-trained language models with temporal sequences, and a discriminator forest architecture to mitigate mode collapse and capture nuanced inter-location relationships.

---

## Project Structure

```plaintext
.
├── CADF_OD.py                         # Main experimental script
├── precompute_token_features_m18_NY.py# Semantic feature extraction (run this first)
├── data_NYTaxi/                       # New York City Taxi Dataset
│   ├── od_matrix_daily.npy            # OD matrix data
│   ├── io_flow_daily.npy              # Inflow-Outflow data
│   ├── graph.npy                      # Distance / adjacency matrix
│   └── grid_population_density_52nodes.json  # Regional population density
└── baselines/                         # Implementations of 11 state-of-the-art baselines
    ├── ForestGAN/
    ├── TimeGAN/
    ├── Spd_DDPM/
    └── ...
Data Availability
We evaluate our model on two major real-world datasets:

New York City (NYC) Dataset
The data is obtained from the NYC Open Data portal and is fully open-source.
All processed files required for reproduction are included in the data_NYTaxi/ directory.

Beijing Dataset
Due to data privacy and security regulations, the Beijing dataset contains sensitive urban information and is not publicly available in this repository.

Usage
1. Prerequisites
Download a semantic embedding model (e.g., Qwen2-7B-embed-base, Word2Vec, or any LLM-based encoder) to extract textual features.

2. Feature Precomputation
Run the following script to generate semantic embeddings for the NYC dataset:

bash
Copy code
python precompute_token_features_m18_NY.py
3. Training & Evaluation
Execute the main script to train the CADF-OD model and evaluate performance:

bash
Copy code
python CADF_OD.py
Baselines
The baselines/ directory includes implementations of 11 competitive models used for comparison, covering a wide range of generative paradigms, including:

GAN-based models (e.g., ForestGAN)

Sequential generative models (e.g., TimeGAN)

Diffusion-based methods (e.g., Spd-DDPM)

Other VAE- and hybrid-based approaches
