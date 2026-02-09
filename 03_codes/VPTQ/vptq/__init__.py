# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import importlib.metadata

from vptq.layers import AutoModelForCausalLM, VQuantLinear

#HJ(1.21) 오류로 인해 수정
#__version__ = importlib.metadata.version("vptq")
__version__ = "0.0.1" #HJ(1.21) 임의버전 강제 할당

__all__ = ["AutoModelForCausalLM", "VQuantLinear"]
