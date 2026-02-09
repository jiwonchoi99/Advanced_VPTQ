# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import json
import os.path as osp
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from sentence_transformers import SentenceTransformer
from torch.multiprocessing import set_start_method
from transformers import AutoTokenizer, HfArgumentParser, set_seed

from vptq.models.llama import eval_llama, get_llama, quant_llama
from vptq.models.mistral import get_mistral
from vptq.models.nvembed import get_nvembed, quant_nvembed
from vptq.models.qwen import eval_qwen, get_qwen, quant_qwen
from vptq.models.phi import eval_phi, get_phi, quant_phi
from vptq.quantizer import QuantizationArguments
from vptq.utils.data import get_data_loader
from vptq.utils.pack import absorb_perm, pack_model


@dataclass
class VPTQArguments:
    model_name: str = field(default="meta-llama/Llama-3.1-8B")
    seq_len: Optional[int] = field(default=2048)
    quant_step: int = field(default=1)
    percdamp: float = field(default=0.01)
    blocksize: int = field(default=128)
    output_dir: str = field(default="/home/sslunder52/project/vptq/01_outputs/output01.21")
    seed: int = field(default=0)
    eval: bool = field(default=False)
    new_eval: bool = field(default=False)
    save_model: bool = field(default=True)
    save_packed_model: bool = field(default=False)
    disable_actorder: bool = field(default=False)
    hessian_path: Optional[str] = field(default="/home/sslunder41/project/VPTQ/hess/Hessians-Llama-31-8B-Instruct-6144-8k")
    inv_hessian_path: Optional[str] = field(default="/home/sslunder41/project/VPTQ/invhess/InvHessians-Llama-31-8B-Instruct-6144-8k")
    num_gpus: int = field(default=3)
    eval_nsamples: int = field(default=128)
    save_qlinear: bool = field(default=False)
    absorb_perm: bool = field(default=False)
    enable_residual: bool = field(default=False)
    eval_mode: bool = field(default=False) ##True only when you have fully quantized model and you want to evaluate
    outlier_size: int = field(default=0)


if __name__ == "__main__":
    parser = HfArgumentParser((VPTQArguments, QuantizationArguments))
    args, quant_args = parser.parse_args_into_dataclasses()

    # set output folder based on time
    args.output_dir = osp.join(args.output_dir, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    #args.output_dir = osp.join(args.output_dir, time.strftime("2026-02-05-12-50-37", time.localtime())) #중간에 이어서 진행하려면 이 코멘트를 풀고 진행하면 된다
    set_start_method("spawn")

    set_seed(args.seed)

    if "llama" in args.model_name.lower():
        model = get_llama(args.model_name)
    elif "qwen" in args.model_name.lower():
        model = get_qwen(args.model_name)
    elif "mistral" in args.model_name.lower():
        model = get_mistral(args.model_name)
    elif "nv-embed" in args.model_name.lower():
        model = get_nvembed(args.model_name)
    elif "phi" in args.model_name.lower():
        model = get_phi(args.model_name)
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

    # set sequence length
    if args.seq_len or model.seqlen is None:
        model.seqlen = args.seq_len
    print(f"model sequence length: {model.seqlen}")

    model.eval()

    tick = time.time()
    print(f'exp time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
    print(f"args: {args}")
    print(f"quant_args: {quant_args}")

    if "llama" in args.model_name.lower() or "mistral" in args.model_name.lower():
        model, quantizers = quant_llama(model, args, quant_args)
    elif "qwen" in args.model_name.lower():
        model, quantizers = quant_qwen(model, args, quant_args)
    elif "nv-embed" in args.model_name.lower():
        model, quantizers = quant_nvembed(model, args, quant_args)
    elif "phi" in args.model_name.lower():
        model, quantizers = quant_phi(model, args, quant_args)
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

    # save model, not for inference
    """
    HJ(1.20)
    - model, config, 그리고 tokenizer를 지정된 경로로 저장한다.
    """
    if args.save_model:
        model_path = osp.join(args.output_dir, 'model/')
        if (args.eval_mode == False):
            model.save_pretrained(model_path)
            
            # save config 
            config_path = osp.join(args.output_dir, 'model/config.json')
            with open(config_path, 'w') as f:
                json.dump(model.config.to_dict(), f)

            print(f'save config to {config_path}')
            print(f'save model to {model_path}')
            tokenizer = AutoTokenizer.from_pretrained(f'{args.model_name}', legacy=False)

            tokenizer.save_pretrained(model_path)
            print(f"save tokenizer to {model_path}")

    
    # save packed model for inference
    if args.save_packed_model:
        model_path = osp.join(args.output_dir, "packed_model/")
        if (args.eval_mode == False):
            model = pack_model(model, from_type=torch.uint16, to_type=torch.uint16, as_type=torch.int16)

            if args.absorb_perm:
                model = absorb_perm(model)
            model.save_pretrained(model_path, safe_serialization=True)
            print(f"save packed model to {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(f"{args.model_name}", legacy=False)

            tokenizer.save_pretrained(model_path)
            print(f"save tokenizer to {model_path}")

    print("\n[DEBUG] 양자화 결과 확인 중...")
    try:
        # 첫 번째 레이어의 q_proj에 indices가 붙었는지 확인
        check_layer = model.model.layers[0].self_attn.q_proj
        if hasattr(check_layer, 'indices') and check_layer.indices is not None:
            print(f"[SUCCESS] Indices found! Shape: {check_layer.indices.shape}")
        else:
            print("[FAIL] Indices are MISSING (None or not attr).")
            print(f"Layer type is still: {type(check_layer)}")
    except Exception as e:
        print(f"[ERROR] Checking failed: {e}")
    
    end_time = time.time()
    #print(f"Total time: {(end_time - start_time)/60:.1f}min")

    model.eval() #HJ(1.20) .eval()을 사용하면 dropout 과 고정된 batchnorm 을 사용하고, 모든 neuron 을 이용해 inference 를 진행한다.

    # TODO: add evaluation for SentenceTransformer (MTEB or other)
    if isinstance(model, SentenceTransformer):
        sentence1 = ["The cat saw the mat"]
        sentence2 = ["The cat sat on the mat"]

        embeddings1 = model.encode(sentence1, convert_to_tensor=True)
        embeddings2 = model.encode(sentence2, convert_to_tensor=True)

        cosine_scores = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)
        print(f"cosine similarity: {cosine_scores.item()}")
        exit()

    if args.eval:
        datasets = ["wikitext2", "c4"]
    if args.new_eval:
        datasets = ["wikitext2", "c4-new"]

    # seqlens = [2048, 4096, 8192]
    seqlens = [2048]

    # store results
    results = {}

    for seqlen in seqlens:
        model.seqlen = seqlen
        for dataset in datasets:
            dataloader, testloader = get_data_loader(
                dataset, seed=args.seed, model=args.model_name, seqlen=model.seqlen
            )
            if "llama" in args.model_name.lower() or "mistral" in args.model_name.lower():
                ppl = eval_llama(model, testloader, "cuda")
            elif "qwen" in args.model_name.lower():
                ppl = eval_qwen(model, testloader, "cuda")
            elif "phi" in args.model_name.lower():
                ppl = eval_phi(model, testloader, "cuda")
            else:
                raise ValueError(f"Unsupported model: {args.model_name}")

            if f"ctx_{seqlen}" not in results:
                results[f"ctx_{seqlen}"] = {}
            results[f"ctx_{seqlen}"][dataset] = ppl

        with open(osp.join(args.output_dir, "ppl_results.json"), "w") as f:
            json.dump(results, f, indent=2)
