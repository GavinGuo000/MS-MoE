# MS-MoE: Multi-Scale Mixture of Experts for Time Series Forecasting

## Project Introduction
**MS-MoE** is a Multi-Scale Mixture of Experts model designed for time series forecasting, aiming to resolve the contradiction between the multi-scale characteristics of real-world time series and the structural uniformity of existing models.

By integrating **multi-scale decomposition**, an **adaptive expert network pool**, and a **dynamic fusion mechanism**, the model achieves accurate modeling of:
- fine-scale high-frequency fluctuations  
- medium-scale periodic patterns  
- coarse-scale long-term trends  

It delivers **state-of-the-art performance** on benchmark datasets across energy, meteorology, transportation, and other domains.

### Core Advantage
On **8 multi-domain datasets**, compared with mainstream baseline models:
- **MSE ↓ ~15%**
- **MAE ↓ ~12%**

Meanwhile, a **sparse expert activation mechanism** balances prediction accuracy and computational efficiency.

---

## Key Features

### 1. Multi-Scale Intelligent Decomposition
Combines **average downsampling** and **seasonal-trend decomposition** to decouple the original sequence into sub-sequences of different granularities.

### 2. Granularity-Adaptive MoE Module
- Dedicated expert pools for each scale  
  - CNN-based experts (fine-scale)  
  - LSTM-based experts (coarse-scale)  
- **Sigmoid-Softmax mixed gating** enables dynamic routing and sparse activation.

### 3. Multi-Scale Collaborative Fusion
Aggregates expert outputs via **attention weighting**, dynamically adjusting the contribution of each scale.

---

## Installation Requirements

- Python 3.8+  
- PyTorch 2.0+  
- NumPy 1.21+  
- Pandas 1.4+  
- Scikit-learn 1.0+  
- Matplotlib 3.5+  
- Tqdm 4.64+  

Install dependencies:
```bash
pip install torch numpy pandas scikit-learn matplotlib tqdm

## Quick Start

### Data Preparation
Download benchmark datasets (ETT, Weather, Solar, Electricity, Traffic) or preprocess using:
```bash
python data_preprocess.py --data_dir ./data --dataset ETTh1

## Experimental Configuration

### Datasets
Validated on **8 benchmark datasets** across multiple domains:

- **Energy/Power**: ETT-h1, ETT-h2, ETT-m1, ETT-m2  
- **Meteorology**: Weather  
- **Energy Consumption**: Electricity  
- **Transportation**: Traffic  
- **Solar Energy**: Solar  

---

### Baseline Models
Compared against **9 state-of-the-art (SOTA) models**:

**TimeMixer, PatchTST, Crossformer, TimesNet, Autoformer, Informer, MTSMixer, TSMixer, DLinear**

---

### Evaluation Metrics
- **MSE** — emphasizes large errors  
- **MAE** — robust to outliers  

---

## Experimental Results

MS-MoE achieves **top performance** across all datasets and forecasting horizons (96 / 192 / 336 / 720).

| Dataset     | MSE (MS-MoE) | MAE (MS-MoE) | Performance Gain |
|-------------|--------------|--------------|------------------|
| ETTh1       | 0.430        | 0.436        | ↓15.3%           |
| Weather     | 0.242        | 0.273        | ↓14.8%           |
| Electricity | 0.183        | 0.271        | ↓16.2%           |
| Traffic     | 0.484        | 0.297        | ↓13.9%           |

For full results and ablation study details, please refer to the original paper.


## Hyperparameter Tuning Guidelines

- **Number of Scales**: `n_scales = 4`  
- **Experts per Scale**: 8–16  
- **Learning Rate**: `5e-4`, cosine annealing recommended  
- **Model Dimension**: `d_model = 512`  
- **Encoder Layers**: 1 layer is usually sufficient  

---

## License
This project is released under the **MIT License**.

