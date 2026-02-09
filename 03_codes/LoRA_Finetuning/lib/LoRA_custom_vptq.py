# Custom dispatcher to support VPTQ's VQuantLinear with PEFT LoRA.

from __future__ import annotations

from typing import Any, Optional

import os.path as osp
import sys
import torch

from .LoRA_custom_layer import LoraLayer, LoraVariant


class LoRA_VPTQ(torch.nn.Module, LoraLayer):
    """LoRA wrapper for VPTQ's VQuantLinear (structure aligned with LoRA_custom_nvfp.py)."""

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
        **kwargs: Any,
    ) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer)
        self.fan_in_fan_out = False
        self._active_adapter = adapter_name
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

    def resolve_lora_variant(self, *, use_dora: bool, **kwargs: Any) -> Optional[LoraVariant]:
        if not use_dora:
            return None
        from .LoRA_custom_variants import DoraLinearVariant

        return DoraLinearVariant()

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        result = self.base_layer(x, *args, **kwargs)
        result = result.clone()

        if self.disable_adapters:
            return result

        if adapter_names is not None:
            # Keep consistent with other wrappers in this repo
            return self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)

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
                x_cast = self._cast_input_dtype(x, lora_A.weight.dtype)
            else:
                x_cast = x

            if active_adapter not in self.lora_variant:  # vanilla LoRA
                output = lora_B(lora_A(dropout(x_cast))) * scaling
                if requires_conversion:
                    output = output.to(expected_dtype)
                result = result + output
            else:
                result = self.lora_variant[active_adapter].forward(
                    self,
                    active_adapter=active_adapter,
                    x=x_cast,
                    result=result,
                )
                if requires_conversion:
                    result = result.to(expected_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


def dispatch_vptq(
    target: torch.nn.Module, adapter_name: str, lora_config=None, **kwargs: Any
) -> Optional[torch.nn.Module]:
    """
    Dispatcher function for VPTQ VQuantLinear layers.
    When a VQuantLinear layer is detected, returns LoRA_VPTQ wrapper.
    """
    try:
        from peft.tuners.tuners_utils import BaseTunerLayer
    except Exception:
        BaseTunerLayer = None

    # NOTE: VQuantLinear may come from different sys.path entries. Avoid relying purely on isinstance.
    VQuantLinear = None
    try:
        _vptq_path = osp.abspath(
            osp.join(osp.dirname(__file__), "/home/jingu_pim6/project/2026_jw_maintapeout/llm_frameworks/04_quant/VPTQ")
        )
        if _vptq_path not in sys.path:
            sys.path.insert(0, _vptq_path)
        from vptq_qwen_vl.layers.vqlinear import VQuantLinear as _VQL  # type: ignore[import]

        VQuantLinear = _VQL
    except Exception:
        VQuantLinear = None

    if BaseTunerLayer is not None and isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    is_vptq_vqlinear = False
    if VQuantLinear is not None and isinstance(target_base_layer, VQuantLinear):
        is_vptq_vqlinear = True
    elif target_base_layer.__class__.__name__ == "VQuantLinear":
        is_vptq_vqlinear = True
    else:
        # Heuristic: VQuantLinear exposes these attrs in this repo
        if (
            hasattr(target_base_layer, "centroids")
            and hasattr(target_base_layer, "indices")
            and hasattr(target_base_layer, "in_features")
            and hasattr(target_base_layer, "out_features")
        ):
            is_vptq_vqlinear = True

    if not is_vptq_vqlinear:
        return None

    vptq_kwargs = kwargs.copy()
    vptq_kwargs["fan_in_fan_out"] = False

    return LoRA_VPTQ(target, adapter_name, **vptq_kwargs)


