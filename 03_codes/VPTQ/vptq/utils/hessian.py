# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# From https://github.com/Cornell-RelaxML/quip-sharp/

import torch
from vptq.utils.reshape import reshape
import torch.nn.functional as F
import pdb

# load Hessian from files
def load_hessian(quant_args, hessian_path, logger=None):
    if logger is None:
        print(f'load Hessian from {hessian_path}')
    else:
        logger.info(f'load Hessian from {hessian_path}')

    H_data = torch.load(f'{hessian_path}', weights_only=False)

    # convert H to sym matrix
    def flat_to_sym(V, N):
        A = torch.zeros(N, N, dtype=V.dtype, device=V.device)
        idxs = torch.tril_indices(N, N, device=V.device)
        A[idxs.unbind()] = V
        A[idxs[1, :], idxs[0, :]] = V
        return A

    def regularize_H(H, n, sigma_reg):
        H.div_(torch.diag(H).mean())
        idx = torch.arange(n)
        H[idx, idx] += sigma_reg
        return H

    def basic_preprocess(H, mu, n):
        H.add_(mu[None, :] * mu[:, None])
        H = regularize_H(H, n, 1e-2)
        return H, mu

    H = flat_to_sym(H_data['flatH'], H_data['n'])
    mu = H_data['mu']
    n = H_data['n']
    H, mu = basic_preprocess(H, mu, n)
    if quant_args.enable_transpose == False:
        if quant_args.vector_lens[0] != -1:
            reshaper = reshape(vector_len=quant_args.vector_lens[1], outlier_vector_len=quant_args.vector_lens[0], enable_transpose=False, group_num = quant_args.group_num)
            H, _, quant_col = reshaper.add_hessian_padding(H)

    return H, mu


# load inverse Hessian from files
# TODO: reduce tensor size
def load_inv_hessian(quant_args, inv_hessian_path, logger=None):
    if logger is None:
        print(f'load inv Hessian from {inv_hessian_path}')
    else:
        logger.info(f'load inv Hessian from {inv_hessian_path}')
    if quant_args.vector_lens[0] != -1:
        reshaper = reshape(vector_len=quant_args.vector_lens[1], outlier_vector_len=quant_args.vector_lens[0],  enable_transpose=False, group_num = quant_args.group_num)
    else:
        reshaper = reshape(vector_len=quant_args.vector_lens[1], outlier_vector_len=0,  enable_transpose=False, group_num = quant_args.group_num)
    H_data = torch.load(f'{inv_hessian_path}', weights_only=False)


    inv_hessian = H_data['invH']
    perm = H_data['perm']
    zero_idx = H_data['zero_idx']

    if quant_args.enable_transpose:
        #inv_hessian, _, quant_col = reshaper.add_invhessian_padding(inv_hessian)
        current_dim = perm.numel()      # 3
        #target_dim = current_dim + quant_col                  # 목표 길이

        #padding_indices = torch.arange(current_dim, target_dim, device=perm.device)
        #perm = torch.cat([perm, padding_indices])   
        
        #zero_idx = F.pad(zero_idx, (0, quant_col), value=True)

        if zero_idx is None:
            zero_idx = torch.diag(inv_hessian) == 0
            inv_hessian[zero_idx, zero_idx] = 1	    

    return inv_hessian, perm, zero_idx
