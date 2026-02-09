from torch.autograd import Function 
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor

import pdb
from .precision import FloatingPoint
import csv

FP_dict= {
    32  : (8, 23)       ,
    31  : (8, 22)       ,
    19  : (8, 10)       ,
    16  : (6, 9)        ,
    16.1: (5, 10)       ,   # IEEE Half Precision
    16.2: (8, 7)        ,   # bfloat16
    8   : (4, 3)        ,
    6   : (3, 2)        ,
    4   : (2, 1)
}

def get_exp_man(x):
    man, exp = torch.frexp(x.abs())
    man, exp = man*2, exp-1 
    
    return exp.float(), man-1

class Scaler(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale 
        
    def forward(self, input: Tensor) -> Tensor : 
        return input * self.scale 
    
class CustomNVFPLinear(nn.Linear):
    def __init__(self, *args, precision=[4, 8], block_size = 16, **kwargs):
        super().__init__(*args, **kwargs)
        # FP Quantization Unit
        self.weight_scale = 1
        self.block_size = block_size
        self.fbw,   self.bsw    = precision  
        self.NVFP_PrecisionUnit = FloatingPoint( FP_dict[self.fbw][0] , FP_dict[self.fbw][1] )
        self.Block_Scaling_PrecisionUnit = FloatingPoint( FP_dict[self.bsw][0] , FP_dict[self.bsw][1] )
    
    def forward(self, x):
        x = CustomNVFP.apply(x, self.weight.T, self.bias, self)
        return x

    def __repr__(self):
        return f"CustomNVFPLinear(in_features={self.in_features}, "\
            f"out_features={self.out_features}, "\
            f"precision={self.fbw, self.bsw}, "\
            f"block_size={self.block_size}, "\
            f"bias={self.bias is not None}), "
 


def replace_all_linear_with_nvfp(model, precision=[4, 8], block_size=16, verbose=True): 
    replaced_count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, CustomNVFPLinear):
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
        
        new_module = CustomNVFPLinear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            precision=precision,
            block_size=block_size
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
            print(f"Replaced {name} with CustomNVFPLinear")
    
    if verbose:
        print(f"Total {replaced_count} Linear layers replaced with CustomNVFPLinear")
    
    return replaced_count

class CustomNVFP(Function): 
    @staticmethod
    def forward(x, w, b, L):
        # Convert input to float32 to prevent overflow
        x = x.to(torch.float32)
        w = w.to(torch.float32)
        if b is not None:
            b = b.to(torch.float32)
        if b is not None:
            original_x = x.clone()
            original_w = w.clone()
            original_b = b.clone()
            original_out = torch.matmul(original_x, original_w) + original_b
        else:
            original_x = x.clone()
            original_w = w.clone()
            original_out = torch.matmul(original_x, original_w)
            
        # Handle input shape: support both 2D (batch*seq, hidden) and 3D (batch, seq, hidden)
        if len(x.shape) == 2:
            # Reshape 2D to 3D: (batch*seq, hidden) -> (batch*seq, 1, hidden)
            x = x.unsqueeze(1)
            was_2d = True
        elif len(x.shape) == 3:
            was_2d = False
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}. Expected 2D or 3D tensor.")
        
        # Tensor Level Scaling Calculation with zero protection
        eps = 1e-12
        x_max = torch.max(x.abs())
        w_max = torch.max(w.abs())
        x_t_scaling = (6*448) / torch.clamp(x_max, min=eps)
        w_t_scaling = (6*448) / torch.clamp(w_max, min=eps)
        
        # Handle bias (may be None)
        has_bias = b is not None

        # Input Block Level Scaling Calculation
        batch, token, ich = x.shape
        block_size = L.block_size

        num_blocks = (ich + block_size - 1) // block_size
        x_abs = x.abs()
        padded_ich = num_blocks * block_size
        if ich < padded_ich:
            pad_size = padded_ich - ich
            x_abs_padded = torch.nn.functional.pad(x_abs, (0, pad_size), mode='constant', value=0.0)
        else:
            x_abs_padded = x_abs
        
        x_abs_reshaped = x_abs_padded.view(batch, token, num_blocks, block_size)
        x_block_maxs = torch.amax(x_abs_reshaped, dim=-1)  # (batch, token, num_blocks)
        
        x_b_scaling = x_block_maxs / 6.0 # (batch, token, num_blocks)
        x_b_scaling_quantized = L.Block_Scaling_PrecisionUnit.quantize(x_b_scaling*x_t_scaling)  # (batch, token, num_blocks)
        # Zero-block protection: if amax_b==0, set quantized decode scale to 1 to avoid division by 0
        x_b_scaling_quantized = torch.where(
            (x_block_maxs == 0),
            torch.ones_like(x_b_scaling_quantized),
            x_b_scaling_quantized
        )

        # Weight Block Level Scaling Calculation
        w_och = w.shape[-1]
        w_abs = w.abs()
        w_padded_ich = num_blocks * block_size
        if ich < w_padded_ich:
            w_pad_size = w_padded_ich - ich
            w_abs_padded = torch.nn.functional.pad(w_abs, (0, 0, 0, w_pad_size), mode='constant', value=0.0)
        else:
            w_abs_padded = w_abs
        
        w_abs_reshaped = w_abs_padded.view(num_blocks, block_size, w_och)
        w_block_maxs = torch.amax(w_abs_reshaped, dim=1)  # (num_blocks, w_och)

        w_b_scaling = w_block_maxs / 6.0 # (num_blocks, w_och)
        w_b_scaling_quantized = L.Block_Scaling_PrecisionUnit.quantize(w_b_scaling*w_t_scaling)  # (num_blocks, w_och)
        w_b_scaling_quantized = torch.where(
            (w_block_maxs == 0),
            torch.ones_like(w_b_scaling_quantized),
            w_b_scaling_quantized
        )

        # Quantization
        x_padded = torch.nn.functional.pad(x, (0, padded_ich - ich), mode='constant', value=0.0) if ich < padded_ich else x
        x_reshaped = x_padded.view(batch, token, num_blocks, block_size)
        x_b_scaling_expanded = x_b_scaling_quantized.unsqueeze(-1)  # (batch, token, num_blocks, 1)

        w_padded = torch.nn.functional.pad(w, (0, 0, 0, padded_ich - ich), mode='constant', value=0.0) if ich < padded_ich else w
        w_reshaped = w_padded.view(num_blocks, block_size, w_och)
        w_b_scaling_expanded = w_b_scaling_quantized.unsqueeze(1)  # (num_blocks, 1, w_och)

        x_scaled = x_reshaped * (x_t_scaling/torch.clamp(x_b_scaling_expanded, min=eps))
        w_scaled = w_reshaped * (w_t_scaling/torch.clamp(w_b_scaling_expanded, min=eps))
        x_quantized = L.NVFP_PrecisionUnit.quantize(x_scaled)
        w_quantized = L.NVFP_PrecisionUnit.quantize(w_scaled)

        x = (x_quantized * x_b_scaling_expanded / torch.clamp(x_t_scaling, min=eps)).view(batch, token, padded_ich)[:, :, :ich]
        w = (w_quantized * w_b_scaling_expanded / torch.clamp(w_t_scaling, min=eps)).view(padded_ich, w_och)[:ich, :]

        # Conversion: Apply block scaling with proper reshaping
        out = torch.matmul(x, w)
        if has_bias:
            out = out + b
        
        # Reshape back to original shape if input was 2D
        if was_2d:
            out = out.squeeze(1)  # (batch*seq, 1, out_features) -> (batch*seq, out_features)
        
        if out.isnan().sum() > 0 or out.isinf().sum():
            pdb.set_trace()  # JG DEBUG
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
