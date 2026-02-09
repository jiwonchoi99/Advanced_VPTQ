# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch


# reshape class for matrix to vectors
class reshape():

    def __init__(self, vector_len,group_num, enable_transpose=False ):
        self.vector_len = vector_len
        self.enable_transpose = enable_transpose
        self.is_padded = False
        self.pad_cols = 0
        self.group_num = group_num

    """
    HJ(1.21)
    
    """
    def matrix2vectors(self, data):
        if data is None:
            return None, None
        if self.enable_transpose:
            data = data.T
        print(data.shape)
        data, self.is_padded, self.pad_cols = self.add_padding(data)
        print(data.shape)
        self.padded_shape = data.shape
        sub_vectors = data.reshape(-1, self.vector_len)
        print(sub_vectors.shape)
        return sub_vectors, self.padded_shape

    def add_whole_padding(self, data):
        '''
        Check if data need padding columns
        Returns (padded data, is_padded, pad_cols)
        '''
        print("add padding working")
        print(f"vector len {self.vector_len}")
        print(f"data.shape[1] {data.shape[1]}")
        print(f"self.group_num {self.group_num}")
        remainder = int(data.shape[1] / self.group_num) % self.vector_len
        print(f"remainder is {remainder}")
        padding_size = (self.vector_len - remainder) * self.group_num
        print(f"padding size is {padding_size}")
        if padding_size != 0:
            padded_tensor = torch.zeros((data.shape[0], padding_size),
                                        dtype=data.dtype,
                                        device=data.device)
            return torch.cat((data, padded_tensor), dim=1), True, padding_size
        return data, False, 0

    def add_hessian_padding(self, data):
        '''
        Check if data need padding columns
        Returns (padded data, is_padded, pad_cols)
        '''
        print("add padding working")
        print(f"vector len {self.vector_len}")
        print(f"data.shape[1] {data.shape[1]}")
        print(f"self.group_num {self.group_num}")
        remainder = int(data.shape[1] / self.group_num) % self.vector_len
        print(f"remainder is {remainder}")
        padding_size = (self.vector_len - remainder) * self.group_num
        print(f"padding size is {padding_size}")
        if padding_size != 0:
            padded_tensor_col = torch.zeros((data.shape[0], padding_size),
                                        dtype=data.dtype,
                                        device=data.device)
            padded_tensor_row = torch.zeros((padding_size, data.shape[1]+padded_tensor_col.shape[1] ),
                                        dtype=data.dtype,
                                        device=data.device)
            return torch.cat(((torch.cat((data, padded_tensor_col), dim=1)), padded_tensor_row), dim=0), True, padding_size
        return data, False, 0

    def add_padding(self, data):
        '''
        Check if data need padding columns
        Returns (padded data, is_padded, pad_cols)
        '''
        print("add padding working")
        print(f"vector len {self.vector_len}")
        print(f"data.shape[1] {data.shape[1]}")
        print(f"self.group_num {self.group_num}")
        remainder = int(data.shape[1])% self.vector_len
        print(f"remainder is {remainder}")
        padding_size = (self.vector_len - remainder) * self.group_num
        print(f"padding size is {padding_size}")
        if remainder != 0:
            padded_tensor = torch.zeros((data.shape[0], padding_size),
                                        dtype=data.dtype,
                                        device=data.device)
            return torch.cat((data, padded_tensor), dim=1), True, padding_size
        return data, False, 0

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
