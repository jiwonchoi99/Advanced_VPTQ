from torch.autograd import Function 
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
import os
from pathlib import Path

import pdb
from .precision import FloatingPoint
import csv

class CustomNVFPSavesLinear(nn.Linear):
    def __init__(self, *args, precision=[4, 8], block_size = 16, layer_name : str = '', layer_num : int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        # FP Quantization Unit
        self.weight_scale = 1
        self.block_size = block_size
        self.fbw,   self.bsw    = precision  
        self.layer_name = layer_name
        self.layer_num = layer_num
    def forward(self, x):
        x = CustomNVFPSaves.apply(x, self.weight.T, self.bias, self)
        return x

    def __repr__(self):
        return f"CustomNVFPSavesLinear(in_features={self.in_features}, "\
            f"out_features={self.out_features}, "\
            f"precision={self.fbw, self.bsw}, "\
            f"block_size={self.block_size}, "\
            f"bias={self.bias is not None}, "\
            f"layer_name={self.layer_name}, "\
            f"layer_num={self.layer_num}), "
 


def replace_all_linear_with_nvfp_for_saves(model, precision=[4, 8], block_size=16, layer_name : str = '', layer_num : int = 0, verbose=True): 
    replaced_count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, CustomNVFPSavesLinear):
            continue
        if not isinstance(module, nn.Linear):
            continue
        
        parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
        attr_name = name.rsplit('.', 1)[-1]

        # Skip language modeling head from replacement
        if attr_name == 'lm_head':
            if verbose:
                print(f"Skip NVFP replacement for lm_head at: {name}")
            continue
        
        if parent_name:
            parent = dict(model.named_modules())[parent_name]
        else:
            parent = model
        
        in_features = module.in_features
        out_features = module.out_features
        bias = module.bias is not None
        
        new_module = CustomNVFPSavesLinear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            precision=precision,
            block_size=block_size,
            layer_name=layer_name,
            layer_num=layer_num,
        )
        
        with torch.no_grad():
            new_module.weight.data.copy_(module.weight.data)
            if bias and module.bias is not None:
                new_module.bias.data.copy_(module.bias.data)
        
        # Freeze base weights for LoRA training
        new_module.weight.requires_grad = False
        if bias and new_module.bias is not None:
            new_module.bias.requires_grad = False
        
        new_module.to(device=module.weight.device, dtype=module.weight.dtype)
        
        setattr(parent, attr_name, new_module)
        replaced_count += 1
        
        if verbose:
            print(f"Replaced {name} with CustomNVFPSavesLinear")
    
    if verbose:
        print(f"Total {replaced_count} Linear layers replaced with CustomNVFPSavesLinear")
    
    return replaced_count

class CustomNVFPSaves(Function): 
    @staticmethod
    def forward(x, w, b, L):
        x = quant(x, 'nvfp4', L.block_size, L.layer_name, L.layer_num, 'input')
        w = quant(w, 'nvfp4', L.block_size, L.layer_name, L.layer_num, 'weight')
        out = torch.matmul(x, w)
        if b is not None:
            out = out + b
        return out.to(torch.float32)
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        x, w, b, L = inputs
        ctx.constant = (x, w, L)
        ctx.has_bias = b is not None
    
    @staticmethod
    def backward(ctx, grad_output):
        x, w, L = ctx.constant
        # Convert all tensors to float32 to ensure dtype consistency
        grad_output = grad_output.to(torch.float32)
        x = x.to(torch.float32)
        w = w.to(torch.float32)
        
        err = torch.matmul(grad_output, w.T)
        if x.ndim > 2:
            x_flat = x.reshape(-1, x.shape[-1])
            grad_flat = grad_output.reshape(-1, grad_output.shape[-1])
            w_grad = torch.matmul(x_flat.T, grad_flat)
        else:
            w_grad = torch.matmul(x.T, grad_output)
        
        if ctx.has_bias:
            b_grad = grad_output.sum(dim=tuple(range(grad_output.ndim - 1)))
        else:
            b_grad = None
        return err, w_grad, b_grad, None

def fake_fp8_quant(x: torch.Tensor, exp_bias: int = 7):
    """
    OFP8 E4M3 quantization (bias=7), no Inf; top-exponent uses mantissa<=110 only.
    Subnormals supported. Max finite = ±448.
    """
    exp_bits, frac_bits = 4, 3
    efield_max = (1 << exp_bits) - 1          # 15 (top exponent used for finite; 111 mantissa reserved -> cap)
    efield_max_finite = efield_max            # OFP8: 15도 유한수 지수로 사용 (단, mantissa 111 금지)
    efield_min_norm = 1                       # 최소 정규 지수 필드

    # 범위/상수
    emax_unbiased = efield_max_finite - exp_bias   # = 8
    min_norm_val  = 2.0 ** (1 - exp_bias)          # = 2^-6
    scale_sub     = 2.0 ** (1 - exp_bias)          # 서브노멀 스케일
    max_val       = (2.0 - 2.0**(-(frac_bits))) * (2.0 ** emax_unbiased)  # = 1.75 * 256 = 448
    # pdb.set_trace()
    eps = torch.finfo(x.dtype).eps

    # 부호/절대값 + 오버플로우 포화
    sign = torch.sign(x)
    ax   = torch.clamp(torch.abs(x), max=max_val)

    # 마스크
    is_zero   = ax == 0
    is_normal = ax >= min_norm_val
    is_sub    = (~is_zero) & (~is_normal)

    out = torch.zeros_like(ax)

    # -------- 정규수 경로 --------
    if is_normal.any():
        xn = ax
        e  = torch.floor(torch.log2(xn + eps))          # 실제 지수
        efield_raw = e + exp_bias                       # 저장 지수(반올림/캐리 전)

        # 정규 mantissa (1.m), 3bit로 반올림
        m = xn / (2.0 ** e) - 1.0                       # m ∈ [0, 1.999..)
        s = torch.round(m * (2 ** frac_bits))           # 정수 스텝(0..8)로 반올림

        # 최상 지수(=15)에서는 mantissa=111 금지 → s 최대 6으로 캡, 캐리 금지
        top_mask = (efield_raw >= efield_max_finite)
        s_top    = torch.clamp(s, 0, (2 ** frac_bits) - 2)        # 0..6
        efield_t = efield_raw                                     # 캐리 없음

        # 비-최상 지수에서는 s==8이면 캐리 발생(efield+1, s=0)
        carry_mask = (~top_mask) & (s >= (2 ** frac_bits))
        s_nontop   = torch.where(carry_mask, torch.zeros_like(s), torch.clamp(s, 0, (2 ** frac_bits) - 1))
        efield_nt  = torch.where(carry_mask, efield_raw + 1, efield_raw)

        # 지수/가수 최종 선택
        s_final = torch.where(top_mask, s_top, s_nontop)
        efield  = torch.where(top_mask, efield_t, efield_nt)

        # 정규 지수 범위로 클램프 (1..15) — 15에서도 s_final<=6이면 유한수
        efield = torch.clamp(efield, efield_min_norm, efield_max_finite)

        # 값 복원: (1 + s/2^3) * 2^(efield - bias)
        xn_q = (1.0 + s_final / (2 ** frac_bits)) * (2.0 ** (efield - exp_bias))
        out = torch.where(is_normal, xn_q, out)

    # -------- 서브노멀 경로 --------
    if is_sub.any():
        xs = ax
        # f ∈ {0..7} (3bit frac), 반올림
        f = torch.round(xs / scale_sub * (2 ** frac_bits))
        promote = f >= (2 ** frac_bits)                          # 반올림으로 최소 정규수 승격

        xs_promote = torch.full_like(xs, min_norm_val)
        f_clamped  = torch.clamp(f, 0, (2 ** frac_bits) - 1)     # 0..7
        xs_stay    = (f_clamped / (2 ** frac_bits)) * scale_sub  # (0.f)*2^(1-bias)
        xs_q = torch.where(promote, xs_promote, xs_stay)

        out = torch.where(is_sub, xs_q, out)

    # 부호 복원
    return out * sign

def e8m0_quant(scale_fp, ceil_mode=False):
    """
    scale_fp: torch.Tensor (floating-point scale, fp16/fp32)
    ceil_mode: True → always ceil to avoid clipping
    return: torch.Tensor (E8M0 quantized scale, float32)
    """
    eps = 1e-8
    log2_val = torch.log2(scale_fp + eps)

    if ceil_mode:
        exp = torch.ceil(log2_val)   # ceil (clipping error x)
    else:
        exp = torch.round(log2_val)  # rtn (clipping error o)

    # exponent range clamp (E8 Range: -126 ~ +127)
    exp = torch.clamp(exp, -126, 127)
    scale_fake_q = torch.pow(2.0, exp)
    scale_q = exp + 127

    return scale_fake_q, scale_q


def nvfp4_sf(x,group):
    G = group
    maxq = 6

    init_shape = x.shape
    if x.dim() == 3:
        reshaped_x = x.reshape(-1, x.shape[-2], x.shape[-1] // G, G)
    elif x.dim() == 2:
        reshaped_x = x.reshape(x.shape[0], x.shape[1] // G, G)

    xmax = torch.amax(reshaped_x, dim=-1, keepdim=True)
    xmin = torch.amin(reshaped_x, dim=-1, keepdim=True)
    
    x_abs_max = torch.maximum(torch.abs(xmin), xmax)
    scale = (x_abs_max / maxq).to(torch.float32)

    # === Per-tensor normalization ===
    s_max = scale.max()
    fp8_max = 448.0   # e4m3 with bias=7
    norm_factor = fp8_max / (s_max + 1e-8)
    scale_norm = (scale * norm_factor)

    # === Fake FP8 quantization (fp32) ===
    fp8_scale = (fake_fp8_quant(scale_norm) / norm_factor).to(torch.float16)
    
    if x.dim() == 3:
        fp8_scale = fp8_scale.expand(-1, -1, -1, G)
    elif x.dim() == 2:
        fp8_scale = fp8_scale.expand(-1, -1, G)
     
    scale = fp8_scale.reshape(init_shape)
    
    tmp = scale == 0
    scale[tmp] = 1e-5

    return scale

def mxfp4_sf(x,group):
    G = group
    maxq = 6

    init_shape = x.shape
    if x.dim() == 3:
        reshaped_x = x.reshape(-1, x.shape[-2], x.shape[-1] // G, G)
    elif x.dim() == 2:
        reshaped_x = x.reshape(x.shape[0], x.shape[1] // G, G)
   

    xmax = torch.amax(reshaped_x, dim=-1, keepdim=True) 
    xmin = torch.amin(reshaped_x, dim=-1, keepdim=True) 
    
    x_abs_max = torch.maximum(torch.abs(xmin), xmax)

    scale = x_abs_max / maxq

    e8m0_scale, exp_val = e8m0_quant(scale)
    # e8m0_scale = scale
    if x.dim() == 3:
        e8m0_scale = e8m0_scale.expand(-1, -1, -1, G)
    elif x.dim() == 2:
        e8m0_scale = e8m0_scale.expand(-1, -1, G)
    
    
    scale = e8m0_scale.reshape(init_shape)
    
    tmp = scale == 0
    scale[tmp] = 1e-5

    return scale


def quant(x:torch.Tensor, method , group, layer_name : str = '', layer_num : int = 0, type : str = 'input'):
    dtype_org = x.dtype
    with torch.no_grad():

        if(method == 'nvfp4' or method == 'mxfp4'):
            x.to(torch.float32)
            if method == 'nvfp4':
                scale = nvfp4_sf(x,group)
            elif method == 'mxfp4':
                scale = mxfp4_sf(x,group)
            else:
                pdb.set_trace()
                print("Wrong format")
            
            exp_bias =  1
            eps = 1e-8

            # Extract sign
            sign = torch.sign(x/scale)
            x_abs = torch.abs(x/scale)

            # Handle zeros separately (to avoid log2 issues)
            is_zero = (x_abs < eps)

            #Exponent quantization (2 bits, range 0~3)
            exp_unbiased = torch.floor(torch.log2(x_abs + eps))
            exp_field = torch.clamp(exp_unbiased + exp_bias, 0, 3)  # stored exponent
            exp_val = exp_field - exp_bias                          # actual exponent

            ################### Save exp_field as numpy array ###################
            # Save exp_val as numpy array
            if layer_name and layer_num is not None:
                save_dir = Path("../04_data/exp_val_saves")
                save_dir.mkdir(exist_ok=True)
                
                # Create filename with layer_name and layer_num
                # Note: x is input, w is weight - we save both separately if needed
                # For now, save exp_val from the current tensor
                filename = f"nvfp4_exp_{layer_name}_layer{layer_num}_{type}.npy"
                filepath = save_dir / filename
                
                # Convert to numpy and save (only save once per layer)
                # Check if file already exists to avoid overwriting during training
                if not filepath.exists():
                    exp_field_np = exp_field.detach().cpu().numpy()
                    np.save(str(filepath), exp_field_np)
                print(f"Saved exp_field for layer: {layer_name} layer {layer_num}")
            ######################################################################

            # # Mantissa quantization (1 bit → {0.0, 0.5})
            # Normalized case (exp_field > 0): hidden-1 포함
            mantissa_norm = x_abs / (2 ** exp_val) - 1.0
            mantissa_q_norm = torch.clamp(torch.round(mantissa_norm * 2) / 2, 0.0, 0.5)

            # Subnormal case (exp_field == 0): hidden-1 제거
            mantissa_sub = x_abs / (2.0 ** (1 - exp_bias))
            mantissa_q_sub = torch.clamp(torch.round(mantissa_sub * 2) / 2, 0.0, 0.5)

            # Reconstruct
            # Normalized case (exp_field > 0): hidden-1 포함
            xq_norm = (1.0 + mantissa_q_norm) * (2.0 ** exp_val)
            # Subnormal case (exp_field == 0): hidden-1 제거
            xq_sub = mantissa_q_sub * (2.0 ** (1 - exp_bias))  # e.g. 0.0 or 0.25

            # Select normal/subnormal
            xq = torch.where(exp_field > 0, xq_norm, xq_sub)
            xq = xq * sign

            # Handle exact zero
            xq[is_zero] = 0.0

            fake_quant = scale * xq

            return fake_quant.to(dtype_org)

            
        if(method == 'no'):
            return x
