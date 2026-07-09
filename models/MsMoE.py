import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize


class TimeMoeTemporalBlock(nn.Module):
    """Base module for expert networks"""
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str = 'gelu'):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if hidden_act == 'gelu':
            self.act_fn = nn.GELU()
        elif hidden_act == 'silu':
            self.act_fn = nn.SiLU()
        else:
            self.act_fn = nn.ReLU()

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class AdaptiveSignalDecoupler(nn.Module):
    """
    Module 1: Adaptive frequency/time-domain hierarchical decoupling.
    Learns frequency cutoff thresholds adaptively to separate the signal into:
    - Micro (high-frequency fluctuations)
    - Meso (mid-frequency periodic patterns)
    - Macro (low-frequency long-term trends)
    No manual scale threshold definition required.
    """
    def __init__(self, d_model, seq_len):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        # Learnable logit cutoff frequencies (sigmoid maps to [0,1])
        self.cutoff_high_logit = nn.Parameter(torch.zeros(1))     # Micro upper bound
        self.cutoff_mid_logit = nn.Parameter(torch.zeros(1) - 1)  # Meso upper bound

        # Learnable per-channel modulation
        self.channel_proj = nn.Linear(d_model, 3 * d_model)

        # Band reconstruction projections
        self.micro_proj = nn.Linear(d_model, d_model)
        self.meso_proj = nn.Linear(d_model, d_model)
        self.macro_proj = nn.Linear(d_model, d_model)

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: [B, T, D]
        Returns: micro, meso, macro each [B, T, D]
        """
        B, T, D = x.shape

        # Compute adaptive cutoffs via sigmoid → [0, 1] range
        cutoff_high = torch.sigmoid(self.cutoff_high_logit) * 0.45 + 0.25  # [0.25, 0.70]
        cutoff_low = torch.sigmoid(self.cutoff_mid_logit) * 0.20 + 0.05   # [0.05, 0.25]

        # Ensure cutoff_low < cutoff_high
        if cutoff_low >= cutoff_high:
            cutoff_low = cutoff_high * 0.5

        # FFT-based frequency decomposition
        x_freq = torch.fft.rfft(x, dim=1)  # [B, T//2+1, D]
        freq_bins = x_freq.shape[1]

        # Normalized frequency indices [0, 1]
        freq_indices = torch.arange(freq_bins, device=x.device, dtype=x.dtype) / max(freq_bins - 1, 1)

        # Create soft masks for each band using sigmoid for differentiability
        sharpness = 30.0
        micro_mask = torch.sigmoid(sharpness * (freq_indices - cutoff_high))
        macro_mask = torch.sigmoid(sharpness * (cutoff_low - freq_indices))
        meso_mask = 1.0 - micro_mask - macro_mask
        meso_mask = meso_mask.clamp(min=0.0)

        # Apply masks to frequency domain
        micro_freq = x_freq * micro_mask.unsqueeze(0).unsqueeze(-1)
        meso_freq = x_freq * meso_mask.unsqueeze(0).unsqueeze(-1)
        macro_freq = x_freq * macro_mask.unsqueeze(0).unsqueeze(-1)

        # Inverse FFT to time domain
        micro = torch.fft.irfft(micro_freq, n=T, dim=1)
        meso = torch.fft.irfft(meso_freq, n=T, dim=1)
        macro = torch.fft.irfft(macro_freq, n=T, dim=1)

        # Channel-adaptive modulation
        channel_weights = self.channel_proj(x)  # [B, T, 3*D]
        micro_w, meso_w, macro_w = channel_weights.chunk(3, dim=-1)  # each [B, T, D]
        micro_w = torch.sigmoid(micro_w)
        meso_w = torch.sigmoid(meso_w)
        macro_w = torch.sigmoid(macro_w)

        # Reconstruct with projection + residual
        micro_out = self.micro_proj(micro * micro_w) + micro
        meso_out = self.meso_proj(meso * meso_w) + meso
        macro_out = self.macro_proj(macro * macro_w) + macro

        return self.layer_norm(micro_out), self.layer_norm(meso_out), self.layer_norm(macro_out)


class MultiScaleMoELayer(nn.Module):
    """
    Module 2: Scale-specific MoE expert layer.
    - Each scale (Micro/Meso/Macro) has an independent expert pool with fully isolated parameters.
    - Uses Sigmoid + Softmax fused sparse gating.
    """
    NUM_SCALES = 3  # Micro, Meso, Macro

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.num_experts = getattr(configs, 'num_experts', 8)
        self.top_k = getattr(configs, 'num_experts_per_tok', 2)
        self.hidden_size = configs.d_model
        expert_intermediate_size = configs.d_ff // max(self.top_k, 1)

        # === Adaptive signal decoupler ===
        self.decoupler = AdaptiveSignalDecoupler(configs.d_model, configs.seq_len)

        # === Scale-specific expert pools (fully isolated parameters) ===
        self.micro_experts = nn.ModuleList([
            TimeMoeTemporalBlock(
                hidden_size=self.hidden_size,
                intermediate_size=expert_intermediate_size,
                hidden_act='gelu'
            ) for _ in range(self.num_experts)
        ])
        self.meso_experts = nn.ModuleList([
            TimeMoeTemporalBlock(
                hidden_size=self.hidden_size,
                intermediate_size=expert_intermediate_size,
                hidden_act='gelu'
            ) for _ in range(self.num_experts)
        ])
        self.macro_experts = nn.ModuleList([
            TimeMoeTemporalBlock(
                hidden_size=self.hidden_size,
                intermediate_size=expert_intermediate_size,
                hidden_act='gelu'
            ) for _ in range(self.num_experts)
        ])
        self.scale_expert_pools = [self.micro_experts, self.meso_experts, self.macro_experts]

        # === Scale-specific routing networks ===
        self.micro_gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.meso_gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.macro_gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.scale_gates = [self.micro_gate, self.meso_gate, self.macro_gate]

        # === Shared expert + Sigmoid gating ===
        self.shared_expert = TimeMoeTemporalBlock(
            hidden_size=self.hidden_size,
            intermediate_size=configs.d_ff,
            hidden_act='gelu'
        )
        self.shared_expert_gate = nn.Linear(self.hidden_size, 1, bias=False)

    def _route_and_compute(self, x, gate, expert_pool):
        """
        Sigmoid + Softmax fused sparse gating routing.
        x: [B, T, D]
        """
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)

        # Softmax routing + Sigmoid gating
        router_logits = gate(x_flat)                          # [B*T, num_experts]
        softmax_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        sigmoid_gate = torch.sigmoid(router_logits.float())
        combined_weights = softmax_weights * sigmoid_gate     # Sigmoid + Softmax fusion

        routing_weights, selected_experts = torch.topk(combined_weights, self.top_k, dim=-1)
        routing_weights = routing_weights.to(x.dtype)

        # Initialize output buffer
        final_hidden_states = torch.zeros_like(x_flat)

        # Expert computation
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            expert_layer = expert_pool[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if len(top_x) > 0:
                current_state = x_flat[top_x]
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(x.dtype))

        # Shared expert output (Sigmoid gating)
        shared_expert_output = self.shared_expert(x_flat)
        shared_expert_output = torch.sigmoid(self.shared_expert_gate(x_flat)) * shared_expert_output

        final_output = final_hidden_states + shared_expert_output
        final_output = final_output.view(B, T, D)

        return final_output, router_logits

    def forward(self, x_list):
        """
        x_list: list of tensors, each [B, T, D] (multi-resolution inputs).
        For each resolution, first apply adaptive tri-scale decoupling,
        then run scale-specific MoE.
        """
        output_list = []
        router_logits_list = []
        branch_outputs = {'micro': [], 'meso': [], 'macro': []}

        for x in x_list:
            # Adaptive frequency-domain tri-scale decoupling
            micro, meso, macro = self.decoupler(x)

            # Independent routing + independent expert pool per scale
            micro_out, micro_logits = self._route_and_compute(micro, self.micro_gate, self.micro_experts)
            meso_out, meso_logits = self._route_and_compute(meso, self.meso_gate, self.meso_experts)
            macro_out, macro_logits = self._route_and_compute(macro, self.macro_gate, self.macro_experts)

            # Sum three scales to get combined output for this resolution
            combined = micro_out + meso_out + macro_out
            output_list.append(combined)

            # Collect router logits for auxiliary loss
            router_logits_list.extend([micro_logits, meso_logits, macro_logits])

            # Collect branch outputs for scale-branch auxiliary loss (first resolution only)
            if len(branch_outputs['micro']) == 0:
                branch_outputs['micro'].append(micro_out)
                branch_outputs['meso'].append(meso_out)
                branch_outputs['macro'].append(macro_out)

        return output_list, router_logits_list, branch_outputs


class CrossScaleAttentionFusion(nn.Module):
    """
    Module 3: Cross-scale attention fusion.
    Computes complementary weights across three scales via multi-head attention,
    adaptively corrects bias.
    Includes auxiliary output: three scale-specific prediction branches.
    """
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        # Per-scale Q projections
        self.query_micro = nn.Linear(d_model, d_model)
        self.query_meso = nn.Linear(d_model, d_model)
        self.query_macro = nn.Linear(d_model, d_model)

        # Shared K/V projections (cross-scale interaction)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        # Fusion output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # Learnable scale correction weights
        self.scale_alpha = nn.Parameter(torch.ones(3) / 3.0)

        # Bias correction network
        self.correction_net = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3)  # one correction coefficient per scale
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, micro_out, meso_out, macro_out):
        """
        micro_out, meso_out, macro_out: [B, T, D]
        Returns:
            fused: [B, T, D] - fused output
            branch_outputs: dict - three scale-specific prediction branches (auxiliary output)
        """
        B, T, D = micro_out.shape

        # Concatenate three scales as shared KV context
        combined_kv = torch.cat([micro_out, meso_out, macro_out], dim=1)  # [B, 3T, D]
        K = self.key_proj(combined_kv)
        V = self.value_proj(combined_kv)

        # Compute cross-scale attention for each scale
        scale_outputs = []
        scale_queries = [self.query_micro(micro_out), self.query_meso(meso_out), self.query_macro(macro_out)]
        scale_inputs = [micro_out, meso_out, macro_out]

        for q, x_in in zip(scale_queries, scale_inputs):
            # Multi-head attention
            q_mh = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k_mh = K.view(B, 3 * T, self.n_heads, self.head_dim).transpose(1, 2)
            v_mh = V.view(B, 3 * T, self.n_heads, self.head_dim).transpose(1, 2)

            # Scaled dot-product attention
            scale_factor = self.head_dim ** -0.5
            attn_weights = torch.matmul(q_mh, k_mh.transpose(-2, -1)) * scale_factor
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)

            attn_output = torch.matmul(attn_weights, v_mh)
            attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)
            attn_output = self.out_proj(attn_output)

            # Residual connection
            scale_out = x_in + attn_output
            scale_outputs.append(scale_out)

        # Adaptive complementary weights: compute correction coefficients from tri-scale info
        combined_features = torch.cat(scale_inputs, dim=-1)  # [B, T, 3D]
        correction_weights = F.softmax(self.correction_net(combined_features), dim=-1)  # [B, T, 3]

        # Weighted fusion + adaptive correction
        fused = torch.zeros_like(micro_out)
        for i, out in enumerate(scale_outputs):
            fused = fused + correction_weights[:, :, i:i + 1] * out

        fused = self.layer_norm(fused)

        # Auxiliary output: three scale-specific prediction branches
        branch_outputs = {
            'micro': scale_outputs[0],
            'meso': scale_outputs[1],
            'macro': scale_outputs[2],
        }

        return fused, branch_outputs


class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window

        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_independence = configs.channel_independence

        if configs.channel_independence == 0:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                nn.GELU(),
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
            )

        # MultiScaleMoELayer with adaptive decoupling + independent expert pools
        self.multi_scale_moe = MultiScaleMoELayer(configs)

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
            nn.GELU(),
            nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
        )

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Preprocess inputs
        processed_list = []
        for x in x_list:
            if self.channel_independence == 0:
                x = self.cross_layer(x)
            processed_list.append(x)

        # Process inputs with MoE (adaptive decoupling + independent experts + Sigmoid+Softmax gating)
        out_list, router_logits, branch_outputs = self.multi_scale_moe(processed_list)

        # Postprocess outputs
        final_out_list = []
        for ori, out, length in zip(x_list, out_list, length_list):
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            final_out_list.append(out[:, :length, :])

        return final_out_list, router_logits, branch_outputs


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence

        # MoE-related parameters
        self.num_experts = getattr(configs, 'num_experts', 8)
        self.num_experts_per_tok = getattr(configs, 'num_experts_per_tok', 2)
        self.aux_loss_weight = getattr(configs, 'aux_loss_weight', 0.01)
        self.branch_loss_weight = getattr(configs, 'branch_loss_weight', 0.1)
        self.apply_aux_loss = getattr(configs, 'apply_aux_loss', True)

        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs)
                                         for _ in range(configs.e_layers)])

        self.preprocess = series_decomp(configs.moving_avg)
        self.enc_in = configs.enc_in
        self.use_future_temporal_feature = configs.use_future_temporal_feature

        if self.channel_independence == 1:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)

        self.layer = configs.e_layers

        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.pred_len,
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

            if self.channel_independence == 1:
                self.projection_layer = nn.Linear(
                    configs.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(
                    configs.d_model, configs.c_out, bias=True)

                self.out_res_layers = torch.nn.ModuleList([
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ])

                self.regression_layers = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(
                            configs.seq_len // (configs.down_sampling_window ** i),
                            configs.pred_len,
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )

            # === Module 3: Cross-scale attention fusion ===
            self.cross_scale_fusion = CrossScaleAttentionFusion(
                d_model=configs.d_model,
                n_heads=getattr(configs, 'n_heads', 4),
                dropout=configs.dropout
            )

            # === Tri-scale branch prediction heads (auxiliary output) ===
            proj_out_dim = 1 if self.channel_independence == 1 else configs.c_out
            self.branch_predict_layers = nn.Linear(configs.seq_len, configs.pred_len)
            self.branch_projection = nn.Linear(configs.d_model, proj_out_dim, bias=True)

        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            if self.channel_independence == 1:
                self.projection_layer = nn.Linear(
                    configs.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(
                    configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def compute_moe_aux_loss(self, router_logits_list):
        """
        Compute MoE load-balancing auxiliary loss (fixed undefined variable bug).
        """
        if not self.apply_aux_loss or not router_logits_list:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        total_aux_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for router_logits in router_logits_list:
            if router_logits is not None:
                # Compute routing weights
                routing_weights = F.softmax(router_logits, dim=-1)
                _, selected_experts = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
                expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).float()

                # Per-expert selection frequency
                tokens_per_expert = torch.mean(expert_mask, dim=0)
                # Average routing probability per expert
                router_prob_per_expert = torch.mean(routing_weights, dim=0)

                # Load-balancing loss
                aux_loss = torch.sum(tokens_per_expert * router_prob_per_expert) * self.num_experts
                total_aux_loss = total_aux_loss + aux_loss

        return total_aux_loss * self.aux_loss_weight

    def compute_branch_aux_loss(self, branch_outputs, batch_y, criterion):
        """
        Compute tri-scale branch prediction auxiliary loss.
        branch_outputs: dict with keys 'micro', 'meso', 'macro', each a list of [B, T, D]
        batch_y: [B, pred_len, C]
        """
        if not branch_outputs or not branch_outputs.get('micro'):
            return torch.tensor(0.0, device=next(self.parameters()).device)

        total_branch_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        f_dim = -1 if self.configs.features == 'MS' else 0
        target = batch_y[:, -self.pred_len:, f_dim:]

        for scale_name in ['micro', 'meso', 'macro']:
            scale_out = branch_outputs[scale_name][0]  # [B, T, D]
            # Generate prediction via branch prediction head
            branch_pred = self.branch_predict_layers(scale_out.permute(0, 2, 1)).permute(0, 2, 1)
            branch_pred = self.branch_projection(branch_pred)
            if self.channel_independence == 1:
                B = branch_pred.shape[0]
                branch_pred = branch_pred.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()

            branch_loss = criterion(branch_pred, target)
            total_branch_loss = total_branch_loss + branch_loss

        return total_branch_loss * self.branch_loss_weight

    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(self, x_list):
        if self.channel_independence == 1:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            return x_enc, x_mark_enc
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc

        return x_enc, x_mark_enc

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        if self.use_future_temporal_feature:
            if self.channel_independence == 1:
                B, T, N = x_enc.size()
                x_mark_dec = x_mark_dec.repeat(N, 1, 1)
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)
            else:
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)

        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_mark = x_mark.repeat(N, 1, 1)
                x_list.append(x)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding
        enc_out_list = []
        x_list = self.pre_enc(x_list)
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)  # [B,T,C]
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)  # [B,T,C]
                enc_out_list.append(enc_out)

        # Past Decomposable Mixing (adaptive decoupling + independent expert MoE)
        all_router_logits = []
        all_branch_outputs = {'micro': [], 'meso': [], 'macro': []}
        for i in range(self.layer):
            result = self.pdm_blocks[i](enc_out_list)
            if isinstance(result, tuple) and len(result) == 3:
                enc_out_list, router_logits, branch_outputs = result
                all_router_logits.extend(router_logits)
                # Keep the last layer's branch outputs
                all_branch_outputs = branch_outputs

        # Future Multipredictor Mixing as decoder for future
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)

        # === Module 3: Cross-scale attention fusion (replaces simple summation) ===
        if len(dec_out_list) >= 3:
            # Multi-resolution outputs: use first 3 for cross-scale fusion
            fused, fusion_branches = self.cross_scale_fusion(
                dec_out_list[0], dec_out_list[1], dec_out_list[2]
            )
            # Sum remaining resolutions if more than 3
            for extra in dec_out_list[3:]:
                fused = fused + extra
            dec_out = fused
        elif len(dec_out_list) == 1:
            dec_out = dec_out_list[0]
            # Single resolution: use directly
            fusion_branches = None
        else:
            # 2 resolutions: sum them
            dec_out = dec_out_list[0]
            for extra in dec_out_list[1:]:
                dec_out = dec_out + extra
            fusion_branches = None

        dec_out = self.normalize_layers[0](dec_out, 'denorm')

        # Return prediction, router logits, and branch outputs
        return dec_out, all_router_logits, all_branch_outputs

    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        if self.channel_independence == 1:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                if self.use_future_temporal_feature:
                    dec_out = dec_out + self.x_mark_dec
                    dec_out = self.projection_layer(dec_out)
                else:
                    dec_out = self.projection_layer(dec_out)
                dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)

        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        return dec_out_list

    def classification(self, x_enc, x_mark_enc):
        x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)
        x_list = x_enc

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-CrissCrossAttention  as encoder for past
        for i in range(self.layer):
            result = self.pdm_blocks[i](enc_out_list)
            if isinstance(result, tuple):
                enc_out_list = result[0]

        enc_out = enc_out_list[0]
        # Output
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def anomaly_detection(self, x_enc):
        B, T, N = x_enc.size()
        x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)

        x_list = []

        for i, x in zip(range(len(x_enc)), x_enc, ):
            B, T, N = x.size()
            x = self.normalize_layers[i](x, 'norm')
            if self.channel_independence == 1:
                x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(x)

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-CrissCrossAttention  as encoder for past
        for i in range(self.layer):
            result = self.pdm_blocks[i](enc_out_list)
            if isinstance(result, tuple):
                enc_out_list = result[0]

        dec_out = self.projection_layer(enc_out_list[0])
        dec_out = dec_out.reshape(B, self.configs.c_out, -1).permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def imputation(self, x_enc, x_mark_enc, mask):
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        B, T, N = x_enc.size()
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)
                x_mark = x_mark.repeat(N, 1, 1)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-CrissCrossAttention  as encoder for past
        for i in range(self.layer):
            result = self.pdm_blocks[i](enc_out_list)
            if isinstance(result, tuple):
                enc_out_list = result[0]

        dec_out = self.projection_layer(enc_out_list[0])
        dec_out = dec_out.reshape(B, self.configs.c_out, -1).permute(0, 2, 1).contiguous()

        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            result = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return result
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        else:
            raise ValueError('Other tasks implemented yet')
