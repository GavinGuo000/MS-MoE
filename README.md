# MS-MoE: Multi-Scale Mixture of Experts for Time Series Forecasting

## Project Introduction
**MS-MoE** is a Multi-Scale Mixture of Experts model designed for time series forecasting, aiming to resolve the contradiction between the multi-scale characteristics of real-world time series and the structural uniformity of existing models.

By integrating **adaptive frequency-domain signal decoupling**, **scale-specific expert network pools**, and **cross-scale attention fusion**, the model achieves accurate modeling of:
- **Micro** — high-frequency fluctuations  
- **Meso** — mid-frequency periodic patterns  
- **Macro** — low-frequency long-term trends  

It delivers strong performance on benchmark datasets across energy, meteorology, transportation, and other domains.

### Core Advantage
- Adaptive frequency cutoff learning — no manual scale threshold definition required  
- Fully isolated expert pools per scale — parameter independence guaranteed  
- **Sigmoid + Softmax mixed gating** — dynamic routing with sparse activation  
- Cross-scale multi-head attention fusion — adaptive complementary weighting  
- Auxiliary losses (MoE load-balancing + scale-branch prediction) — enhanced training stability  

---

## Key Features

### 1. Adaptive Signal Decoupling
Uses **FFT-based frequency-domain decomposition** with **learnable cutoff frequencies** to adaptively separate the input signal into three sub-sequences:
- **Micro** (high-frequency, cutoff upper bound ∈ [0.25, 0.70])  
- **Meso** (mid-frequency, between the two cutoffs)  
- **Macro** (low-frequency, cutoff lower bound ∈ [0.05, 0.25])  

Per-channel adaptive modulation and residual reconstruction ensure expressive power.

### 2. Scale-Specific MoE Module
- **Independent expert pools** for Micro / Meso / Macro — parameters fully isolated  
- Each expert is an MLP block (SwiGLU-style: gate_proj + up_proj + down_proj with GELU activation)  
- **Sigmoid + Softmax mixed gating**: softmax routing × sigmoid gating for dynamic sparse activation  
- Shared expert with sigmoid gate for common knowledge across scales  
- Top-K sparse activation balances accuracy and efficiency  

### 3. Cross-Scale Attention Fusion
- Multi-head attention computes **cross-scale complementary weights**  
- Learnable correction network adaptively adjusts per-scale contributions  
- Residual connections preserve scale-specific information  
- Active when `down_sampling_layers >= 2` (multi-resolution outputs)  

### 4. Loss Function
- **Main loss**: MSE (or L1 for PEMS dataset)  
- **MoE auxiliary loss**: load-balancing loss encourages uniform expert utilization  
- **Branch auxiliary loss**: MSE on each scale's independent prediction branch  

### 5. Training Strategy
- **Optimizer**: AdamW with weight decay  
- **Scheduler**: CosineAnnealingLR  
- **Gradient clipping**: configurable max norm (default: 1.0)  
- **Early stopping**: patience-based on validation loss  

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
```


## Quick Start

### Data Preparation
Download benchmark datasets (ETT, Weather, Solar, Electricity, Traffic) and place them under `./dataset/`.

### Run Training
```bash
# Example: ETTh1 dataset, predict 96 steps ahead
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model MsMoE \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --c_out 7 \
  --d_model 16 \
  --d_ff 32 \
  --learning_rate 0.01 \
  --train_epochs 10 \
  --patience 10 \
  --batch_size 128 \
  --down_sampling_layers 3 \
  --down_sampling_method avg \
  --down_sampling_window 2 \
  --num_experts 8 \
  --num_experts_per_tok 2 \
  --aux_loss_weight 0.01 \
  --branch_loss_weight 0.1 \
  --des 'Exp' \
  --itr 1
```

Or use the provided shell scripts:
```bash
bash scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1_unify.sh
```

## License
This project is released under the **MIT License**.
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
```


## Quick Start

### Data Preparation
Download benchmark datasets (ETT, Weather, Solar, Electricity, Traffic) or preprocess using:
```bash
python data_preprocess.py --data_dir ./data --dataset ETTh1
```


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

