# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch
import math


# reshape class for matrix to vectors
class reshape():

    def __init__(self, vector_len, group_num, outlier_vector_len = 0, enable_transpose=False):
        self.vector_len = vector_len
        self.outlier_vector_len = outlier_vector_len
        self.enable_transpose = enable_transpose
        self.is_padded = False
        self.pad_cols = 0
        self.group_num = group_num
        self.npercent = 1

    """
    HJ(1.21)
    
    """

    def matrix2vectors(self, data):
        if data is None:
            return None, None
        if self.enable_transpose:
            data = data.T
            data, self.is_padded, self.pad_cols = self.add_padding(data)
        self.padded_shape = data.shape
        sub_vectors = data.reshape(-1, self.vector_len)
        return sub_vectors, self.padded_shape


    def add_padding(self, data):
        '''
        Check if data need padding columns
        Returns (padded data, is_padded, pad_cols)
        '''
        remainder = data.shape[1] % self.vector_len
        if remainder != 0:
            padded_tensor = torch.zeros((data.shape[0], self.vector_len - remainder),
                                        dtype=data.dtype,
                                        device=data.device)
            return torch.cat((data, padded_tensor), dim=1), True, self.vector_len - remainder
        return data, False, 0


    def add_whole_padding(self, data):
        '''
        Check if data need padding columns
        Returns (padded data, is_padded, pad_cols)
        '''
        remainder = int(data.shape[1] / self.group_num) % self.vector_len
        padding_size = (self.vector_len - remainder) * self.group_num

        outlier_size = int(math.ceil((self.npercent / 100) * (data.shape[1] + padding_size)))
        if self.outlier_vector_len != 0:
            outlier_remainder = outlier_size % self.outlier_vector_len
            outlier_size = outlier_size - outlier_remainder
        self.outlier_size = outlier_size

        print(f"outlier_size: {outlier_size}")

        if padding_size != 0:
            padded_tensor = torch.zeros((data.shape[0], padding_size + outlier_size),
                                        dtype=data.dtype,
                                        device=data.device)
            return torch.cat((data, padded_tensor), dim=1), True, padding_size + outlier_size, outlier_size
        return data, False, 0

    def add_hessian_padding(self, data):
        '''
        Check if data need padding columns
        Returns (padded data, is_padded, pad_cols)
        '''
        
        remainder = int(data.shape[1] / self.group_num) % self.vector_len
        padding_size = (self.vector_len - remainder) * self.group_num

        outlier_size = int(math.ceil((self.npercent / 100) * (data.shape[1] + padding_size)))
        if self.outlier_vector_len != 0:
            outlier_remainder = outlier_size % self.outlier_vector_len
            outlier_size = outlier_size - outlier_remainder
        self.outlier_size = outlier_size

        if padding_size != 0 and self.outlier_vector_len == 0:
            padded_tensor_col = torch.zeros((data.shape[0], padding_size + outlier_size),
                                        dtype=data.dtype,
                                        device=data.device)
            padded_tensor_row = torch.zeros((padding_size + outlier_size, data.shape[1]+padded_tensor_col.shape[1] ),
                                        dtype=data.dtype,
                                        device=data.device)
            return torch.cat(((torch.cat((data, padded_tensor_col), dim=1)), padded_tensor_row), dim=0), True, padding_size + outlier_size
        return data, False, 0

    def add_invhessian_padding(self, data):
        '''
        Check if data need padding columns
        Returns (padded data, is_padded, pad_cols)
        '''
        remainder = int(data.shape[1] / self.group_num) % self.vector_len
        padding_size = (self.vector_len - remainder) * self.group_num

        outlier_size = int(math.ceil((self.npercent / 100) * (data.shape[1] + padding_size)))
        if self.outlier_vector_len != 0:
            outlier_remainder = outlier_size % self.outlier_vector_len
            outlier_size = outlier_size - outlier_remainder
        self.outlier_size = outlier_size

        if padding_size != 0:
            padded_tensor_col = torch.zeros((data.shape[0], padding_size + outlier_size),
                                        dtype=data.dtype,
                                        device=data.device)
            padded_tensor_row = torch.zeros((padding_size + outlier_size, data.shape[1]+padded_tensor_col.shape[1] ),
                                        dtype=data.dtype,
                                        device=data.device)
            return torch.cat(((torch.cat((data, padded_tensor_col), dim=1)), padded_tensor_row), dim=0), True, padding_size + outlier_size
        return data, False, 0

    #def add_padding(self, data):
    #    '''
    #    Check if data need padding columns
    #    Returns (padded data, is_padded, pad_cols)
    #    '''
    #    remainder = int(data.shape[1])% self.vector_len
    #
    #    padding_size = (self.vector_len - remainder) * self.group_num
    #
    #    outlier_size = int(math.ceil((self.npercent / 100) * (data.shape[1] + padding_size)))
    #    if self.outlier_vector_len != 0:
    #        outlier_remainder = outlier_size % self.outlier_vector_len
    #        outlier_size = outlier_size - outlier_remainder
    #    self.outlier_size = outlier_size
    #
    #    if remainder != 0:
    #        padded_tensor = torch.zeros((data.shape[0], padding_size + outlier_size),
    #                                    dtype=data.dtype,
    #                                    device=data.device)
    #        return torch.cat((data, padded_tensor), dim=1), True, padding_size + outlier_size
    #    return data, False, 0

    def remove_padding(self, data):
        '''
        Remove padding
        '''
        if self.is_padded:
            if self.enable_transpose:
                data = data[:, :-self.pad_cols].T
            else:
                data = data[:, :-self.pad_cols]
        else:
            if self.enable_transpose:
                data = data.T
            else:
                data = data
        return data
