# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import warnings
from typing import Any, Optional

import bitsandbytes as bnb
import torch
import pdb

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import dequantize_bnb_weight
from peft.utils.other import transpose

from .LoRA_custom_layer import LoraLayer, LoraVariant
from .custom_nvfp_for_saves import CustomNVFPSavesLinear

class LoRA_NVFPSaves(torch.nn.Module, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer: torch.nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        layer_num: int = 0,
        layer_name : str = '',
        lora_skip: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer)
        self.fan_in_fan_out = False
        self._active_adapter = adapter_name
        self.layer_num = layer_num
        self.lora_skip = lora_skip
        self.layer_name = layer_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            lora_bias=lora_bias,
        )
    def resolve_lora_variant(self, *, use_dora: bool, **kwargs) -> Optional[LoraVariant]:
        if not use_dora:
            return None
        from .LoRA_custom_variants import DoraLinearVariant
        return DoraLinearVariant()
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)
        result = self.base_layer(x, *args, **kwargs)
        # As per Tim Dettmers, for 4bit, we need to defensively clone here.
        # The reason is that in some cases, an error can occur that backprop
        # does not work on a manipulated view. This issue may be solved with
        # newer PyTorch versions but this would need extensive testing to be
        # sure.
        result = result.clone()
        base_result = result.clone()
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                x = self._cast_input_dtype(x, lora_A.weight.dtype)
            if active_adapter not in self.lora_variant:  # vanilla LoRA
                output = lora_B(lora_A(dropout(x))) * scaling
                if requires_conversion:
                    output = output.to(expected_dtype)
                result = result + output
            else:
                result = self.lora_variant[active_adapter].forward(
                    self,
                    active_adapter=active_adapter,
                    x=x,
                    result=result,
                )
                if requires_conversion:
                    result = result.to(expected_dtype)
        ###################### [LJG] Top-K LoRA Based Sampling ######################
        if self.lora_skip:
            # output shape: (batch_size, token_length, channel)
            # Sum over token_length to get scores for each channel
            lora_score = base_result.sum(dim=1)  # (batch_size, channel)
            K = lora_score.shape[1] // 16 * 6
            topk_values, topk_indices = torch.topk(lora_score.abs(), K, dim=1)  # (batch_size, K)
            
            # Create mask: (batch_size, token_length, channel)
            batch_size, token_length, channel = base_result.shape
            mask = torch.zeros(batch_size, token_length, channel, 
                            dtype=torch.bool, device=base_result.device)
            # Set top-k channel indices to True for each batch
            batch_indices = torch.arange(batch_size, device=base_result.device).unsqueeze(1)  # (batch_size, 1)
            mask[batch_indices, :, topk_indices] = True # Broadcasting: (batch_size, token_length, K)
            
            # Apply mask: keep top-k channels, set others to 0
            result = result * mask.float() + base_result * (1 - mask.float())
        #############################################################################
        return result
    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


def dispatch_nvfp_for_saves(target: torch.nn.Module, adapter_name: str, **kwargs) -> Optional[torch.nn.Module]:
    """
    Dispatcher function for CustomNVFPSavesLinear layers.
    When a CustomNVFPSavesLinear layer is detected, returns LoRA_NVFPSaves wrapper.
    """
    from peft.tuners.tuners_utils import BaseTunerLayer
    
    new_module = None
    
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target
    
    # Check if the base layer is CustomNVFPSavesLinear
    if isinstance(target_base_layer, CustomNVFPSavesLinear):
        nvfp_for_saves_kwargs = kwargs.copy()
        # Extract any additional kwargs that might be needed
        # (e.g., layer_num, lora_skip from the base layer if available)
        if hasattr(target_base_layer, 'block_size'):
            # Preserve NVFP-specific attributes if needed
            pass
        new_module = LoRA_NVFPSaves(target, adapter_name, **nvfp_for_saves_kwargs)

    return new_module