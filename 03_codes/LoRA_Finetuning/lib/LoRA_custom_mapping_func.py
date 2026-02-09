# Copyright 2024-present the HuggingFace Inc. team.
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
from typing import Optional

from transformers import PreTrainedModel

from peft.auto import MODEL_TYPE_TO_PEFT_MODEL_MAPPING
from peft.config import PeftConfig
from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING, PEFT_TYPE_TO_PREFIX_MAPPING
from peft.mixed_model import PeftMixedModel
from peft.peft_model import PeftModel
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import _prepare_prompt_learning_config

# Patch LoraModel to add VPTQ(VQuantLinear) dispatcher (+ NVFP optionally).
# NOTE: PEFT version differences:
# - Some versions export LoraModel from peft.tuners.lora
# - Others only have it in peft.tuners.lora.model
#
# IMPORTANT: NVFP code path may depend on optional deps (e.g., bitsandbytes).
# We must not let NVFP import failure disable VPTQ patching.
try:
    from .LoRA_custom_vptq import dispatch_vptq

    try:
        from .LoRA_custom_nvfp import dispatch_nvfp
    except Exception:
        dispatch_nvfp = None

    _LoraModel = None
    try:
        from peft.tuners.lora.model import LoraModel as _LM  # preferred / most common

        _LoraModel = _LM
    except Exception:
        try:
            from peft.tuners.lora import LoraModel as _LM  # older PEFT

            _LoraModel = _LM
        except Exception:
            _LoraModel = None

    if _LoraModel is not None:
        _original_create_new_module = _LoraModel._create_new_module

        @staticmethod
        def _create_new_module_with_custom_dispatchers(lora_config, adapter_name, target, **kwargs):
            """Wrapper that checks custom dispatchers first, then falls back to original."""
            # Try VPTQ dispatcher first (VQuantLinear)
            vptq_module = dispatch_vptq(target, adapter_name, lora_config=lora_config, **kwargs)
            if vptq_module is not None:
                return vptq_module

            # Try NVFP dispatcher (optional)
            if dispatch_nvfp is not None:
                nvfp_module = dispatch_nvfp(target, adapter_name, **kwargs)
                if nvfp_module is not None:
                    return nvfp_module

            # Fall back to original dispatcher chain
            return _original_create_new_module(lora_config, adapter_name, target, **kwargs)

        _LoraModel._create_new_module = _create_new_module_with_custom_dispatchers
except Exception:
    # If patching fails, we silently continue (will fall back to vanilla PEFT behavior).
    pass


def get_peft_model(
    model: PreTrainedModel,
    peft_config: PeftConfig,
    adapter_name: str = "default",
    mixed: bool = False,
    autocast_adapter_dtype: bool = True,
    revision: Optional[str] = None,
    low_cpu_mem_usage: bool = False,
) -> PeftModel | PeftMixedModel:
    """
    Returns a Peft model object from a model and a config, where the model will be modified in-place.

    Args:
        model ([`transformers.PreTrainedModel`]):
            Model to be wrapped.
        peft_config ([`PeftConfig`]):
            Configuration object containing the parameters of the Peft model.
        adapter_name (`str`, `optional`, defaults to `"default"`):
            The name of the adapter to be injected, if not provided, the default adapter name is used ("default").
        mixed (`bool`, `optional`, defaults to `False`):
            Whether to allow mixing different (compatible) adapter types.
        autocast_adapter_dtype (`bool`, *optional*):
            Whether to autocast the adapter dtype. Defaults to `True`. Right now, this will only cast adapter weights
            using float16 or bfloat16 to float32, as this is typically required for stable training, and only affect
            select PEFT tuners.
        revision (`str`, `optional`, defaults to `main`):
            The revision of the base model. If this isn't set, the saved peft model will load the `main` revision for
            the base model
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading process. Leave this setting as
            False if you intend on training the model, unless the adapter weights will be replaced by different weights
            before training starts.
    """
    model_config = BaseTuner.get_model_config(model)
    old_name = peft_config.base_model_name_or_path
    new_name = model.__dict__.get("name_or_path", None)
    peft_config.base_model_name_or_path = new_name

    # Especially in notebook environments there could be a case that a user wants to experiment with different
    # configuration values. However, it is likely that there won't be any changes for new configs on an already
    # initialized PEFT model. The best we can do is warn the user about it.
    if any(isinstance(module, BaseTunerLayer) for module in model.modules()):
        warnings.warn(
            "You are trying to modify a model with PEFT for a second time. If you want to reload the model with a "
            "different config, make sure to call `.unload()` before."
        )

    if (old_name is not None) and (old_name != new_name):
        warnings.warn(
            f"The PEFT config's `base_model_name_or_path` was renamed from '{old_name}' to '{new_name}'. "
            "Please ensure that the correct base model is loaded when loading this checkpoint."
        )

    if revision is not None:
        if peft_config.revision is not None and peft_config.revision != revision:
            warnings.warn(
                f"peft config has already set base model revision to {peft_config.revision}, overwriting with revision {revision}"
            )
        peft_config.revision = revision

    if (
        (isinstance(peft_config, PEFT_TYPE_TO_CONFIG_MAPPING["LORA"]))
        and (peft_config.init_lora_weights == "eva")
        and not low_cpu_mem_usage
    ):
        warnings.warn(
            "lora with eva initialization used with low_cpu_mem_usage=False. "
            "Setting low_cpu_mem_usage=True can improve the maximum batch size possible for eva initialization."
        )

    prefix = PEFT_TYPE_TO_PREFIX_MAPPING.get(peft_config.peft_type)
    if prefix and adapter_name in prefix:
        warnings.warn(
            f"Adapter name {adapter_name} should not be contained in the prefix {prefix}."
            "This may lead to reinitialization of the adapter weights during loading."
        )

    if mixed:
        # note: PeftMixedModel does not support autocast_adapter_dtype, so don't pass it
        return PeftMixedModel(model, peft_config, adapter_name=adapter_name)

    # We explicitly exclude prompt learning here since prompt learning is specific to the task and needs special
    # handling in the PEFT model's forward method.
    if peft_config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys() and not peft_config.is_prompt_learning:
        return PeftModel(
            model,
            peft_config,
            adapter_name=adapter_name,
            autocast_adapter_dtype=autocast_adapter_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )

    if peft_config.is_prompt_learning:
        peft_config = _prepare_prompt_learning_config(peft_config, model_config)
    return MODEL_TYPE_TO_PEFT_MODEL_MAPPING[peft_config.task_type](
        model,
        peft_config,
        adapter_name=adapter_name,
        autocast_adapter_dtype=autocast_adapter_dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
