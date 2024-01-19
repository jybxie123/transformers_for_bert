import torch
import numpy as np


def input2sparse(input, keep_frac):
    # 选择需要稀疏几个维度，这里一维。
    def shp(t):
        return tuple(t.size())
    if len(shp(input)) == 4:
        batch_size = shp(input)[0] * shp(input)[1] * shp(input)[2]
        feature_len = shp(input)[3]
    elif len(shp(input)) == 2:
        batch_size = shp(input)[0]
        feature_len = shp(input)[1]
    elif len(shp(input)) == 3: # test->(batch_size, feature_len, 1)
        batch_size = shp(input)[0]*shp(input)[1] 
        feature_len = shp(input)[2] 
    input = input.contiguous()

    kept_feature_size = int(feature_len * keep_frac + 0.999)
    gather_index = get_batch_score(input, batch_size, feature_len, kept_feature_size)
    gathered_input = select_columns(input.view(batch_size,feature_len), gather_index)
    # print('input1 shape, batch size, feature len, index: ',input.shape, batch_size, feature_len, gather_index.shape)
    # 区别，这里是直接把0的位置去掉了，而不是稀疏化。
    # with torch.autograd.grad_mode.no_grad():
        # gathered_input = torch.index_select(input.view(batch_size, feature_len), 1, gather_index.squeeze()).clone()
    sparse_input = denseToSparse(gathered_input)
    return sparse_input, gather_index

# def sparse2input(gathered_input, input_shape, gather_index):
def sparse2input(sparse_input, input_shape):

    """
    Inverse of input2sparse. Accepts the outputted reduced tensor from input2sparse along with
    the expected size of the input.

    One and only one of gather_index or random_seed must be provided.
    This method must take either the gather_index outputted by input2sparse, or the random seed
    used by input2sparse. If the random seed is provided, this method will reconstruct gather_index, which
    contains the random indices used to sample the input.

    Arguments:
        gathered_input: The outputted reduced tensor from input2sparse.
        input_shape: The shape of the input tensor fed into input2sparse.
        gather_index: The random indices generated by input2sparse.
    """

    # def shp(t):
    #     return tuple(t.size())

    # if len(input_shape) == 4:
    #     batch_size = input_shape[0] * input_shape[1]
    #     feature_len = input_shape[2] * input_shape[3]
    # elif len(input_shape) == 2:
    #     batch_size = input_shape[0]
    #     feature_len = input_shape[1]
    # elif len(input_shape) == 3: # test->(batch_size, feature_len, 1)
    #     batch_size = input_shape[0]
    #     feature_len = input_shape[1] * input_shape[2]
    input = sparseToDense(sparse_input) 
    input = input.view(input_shape)
    # # 这是原来的恢复函数
    # with torch.autograd.grad_mode.no_grad():
    #     input = torch.zeros(batch_size, feature_len, device=gathered_input.device).to(gathered_input.dtype)
    #     # batch_index = torch.arange(batch_size).view(-1, 1).expand(-1, feature_len)
    #     # input.index_put_((batch_index, gather_index.expand(batch_size, -1)), gathered_input, accumulate=True)
    #     # input = input.view(input_shape)
    #     batch_index = torch.arange(batch_size).view(batch_size, 1)
    #     input.index_put_((batch_index, gather_index), gathered_input, accumulate=True)
    #     input_shape = torch.Size(input_shape)
    #     input = input.view(input_shape)
    return input

def get_ref_dim_reduced_input(input, gather_index, input_shape):
    def shp(t):
        return tuple(t.size())
    if len(shp(input)) == 4:
        batch_size = shp(input)[0] * shp(input)[1]*shp(input)[2]
        feature_len =  shp(input)[3]
    elif len(shp(input)) == 2:
        batch_size = shp(input)[0]
        feature_len = shp(input)[1]
    elif len(shp(input)) == 3: # test->(batch_size, feature_len, 1)
        batch_size = shp(input)[0]*shp(input)[1] 
        feature_len = shp(input)[2] 
    input = input.contiguous()
    # print('input2 shape, batch size, feature len, index: ',input.shape, batch_size, feature_len, gather_index.shape)
    
    if feature_len != input_shape[-1]:
        # print('feature_len : ',feature_len, 'input_shape[-1] : ',input_shape[-1])
        raise ValueError("feature_len is not equal to input_shape[-1]")
    gathered_input = select_columns(input.view(batch_size,feature_len), gather_index)
    sparse_input = denseToSparse(gathered_input)
    return sparse_input



def get_batch_score(input, batch_size, feature_len, kept_feature_size):
    # print('input shape, batch size, feature len, kept: ',input.shape, batch_size, feature_len, kept_feature_size)
    temp_input = input.view(batch_size, feature_len)
    temp_input_norm = torch.norm(temp_input, dim=0) # 对列求范数    
    temp_input_std = torch.std(temp_input, dim=0)*np.sqrt(feature_len)
    sf_temp_input_norm = torch.softmax(temp_input_norm, dim=0)
    sf_temp_input_std = torch.softmax(temp_input_std, dim=0)
    sf_temp_relative_std = torch.softmax(temp_input_std/temp_input_norm, dim=0)
    # score = sf_temp_input_norm + sf_temp_input_std + sf_temp_relative_std # 三个指标的加权和
    score = sf_temp_input_norm
    _, gather_index = torch.topk(score, kept_feature_size, dim=0, largest=True, sorted=True, out=None)
    return gather_index

# ===========================back razor===========================

# def get_batch_score(input, batch_size, feature_len, kept_feature_size):
#     activation_mag = torch.abs(input)
#     threshold, _ = torch.kthvalue(activation_mag.flatten(1), kept_feature_size)
#     while len(threshold.shape) < len(activation_mag.shape):
#         threshold = threshold.unsqueeze(-1)
#     mask = activation_mag >= threshold

#     # print("mask density is {}".format(mask.float().mean()))
#     # idle mask
#     # mask = torch.ones_like(activation).to(torch.bool)
#     return mask

# dense tensor to sparse tensor
def denseToSparse(x):
    indices = torch.nonzero(x, as_tuple=True)
    values = x[indices]
    stacked = torch.stack(indices)
    sparse_x = torch.sparse_coo_tensor(stacked, values, x.size())
    return sparse_x

# sparse tensor to dense tensor
def sparseToDense(sparse_x):
    dense_tensor = sparse_x.to_dense()
    return dense_tensor

# input should be viewed into [xxx, feature_len], then we give the output with masked zero.
def select_columns(input, col_idx):
    new_input = input.clone()
    mask = torch.zeros(new_input.size(1), dtype=torch.bool)
    mask[col_idx] = True
    new_input[:, ~mask] = 0
    return new_input

'''
流程：
input原维度
变为b f类型的维度
先在f上找出需要稀疏的维度 index
再将原矩阵稀疏为目标矩阵，但是0没有去掉
将目标矩阵转为稀疏矩阵类型，存起来。
稀疏矩阵回头可以恢复为目标矩阵，但是这里需要再view为原shape才可以作乘法。
'''

# test

import torch.nn.functional as F

if __name__ == "__main__":
    with torch.no_grad():
        # # test index column generation
        # input = torch.rand(3)
        # print("input is {}".format(input))
        # y = input.repeat(4,1)
        # print("y is {}".format(y))

        # test select columns 
        x = torch.randint(3, (3,3,3))
        print("x is {}".format(x))

        idx = torch.tensor([0,1])
        new_w = select_columns(x, idx)
        print("x is {}".format(x))
        print("new_w is {}".format(new_w))

        # # test dense to sparse
        # x = torch.randint(3, (3,3,3))
        # x[1,2] = 0
        # x[2,1] = 0
        # print(x)
        # dim_len = len(x.shape)
        # indices = torch.nonzero(x, as_tuple=True)

        # values = x[indices]
        # print("indices : ", indices)
        # print("values : ", values)
        # # indices = torch.tensor(indices)
        # # print("indices : ", indices)

        # stacked = torch.stack(indices)
        # print("stacked : ", stacked)

        # sparse_x = torch.sparse_coo_tensor(stacked, values, x.size())
        # dense_tensor = sparse_x.to_dense()
        # print(sparse_x)
        # print(dense_tensor)
        # print('type: ', type(sparse_x), type(dense_tensor), type(indices), type(values))

        # # test input2sparse
        # input = torch.rand(3,4)
        # weight = torch.rand(2,4)
        # bias = torch.rand(3,2)
        # print("input is {}".format(input))
        # keep_frac = 0.5
        # # gather_index = get_batch_score(input, 3, 4, kept_feature_size)
        # sparse, gather_index = input2sparse(input, keep_frac)
        # print("gather_index is {}".format(gather_index))
        # print("sparse is {}".format(sparse))
        # new_input = sparse2input(sparse, (3,4), gather_index)
        # print("new_input is {}".format(new_input))

        # def cln(t):
        #     if t is None:
        #         return None
        #     ct = t.clone().detach()
        #     ct.requires_grad_(True)
        #     return ct

        # cinput = cln(input)
        # cweight = cln(weight)
        # cbias = cln(bias)
        # grad_output = torch.rand(3,2)
        # grad_output.requires_grad_(True)
        # with torch.autograd.grad_mode.enable_grad():
        #     output = F.linear(cinput, cweight, bias=cbias)
        # # bias_grad_input, input_grad_input, weight_grad_input = output.grad_fn(grad_output)
        # input_grad_input, weight_grad_input, bias_grad_input = torch.autograd.grad(output, (cinput, cweight, cbias), grad_output)
        # print('backward grad_output : ',grad_output)
        # print('grad : ', input_grad_input, weight_grad_input, bias_grad_input)
        # print('backward shape : ',input_grad_input.shape,weight_grad_input.shape,bias_grad_input.shape)




# ================== back razor(not used) ==================
from pdb import set_trace
class Masker(object):
    def __init__(self, prune_ratio):
        self.prune_ratio = prune_ratio
    @torch.no_grad()
    def __call__(self, activation):
        num_small = int(np.clip(activation[0].numel() * self.prune_ratio, 1, activation[0].numel()))
        activation_mag = torch.abs(activation)
        threshold, _ = torch.kthvalue(activation_mag.flatten(1), num_small)
        while len(threshold.shape) < len(activation_mag.shape):
            threshold = threshold.unsqueeze(-1)
        mask = activation_mag >= threshold
        return mask

def sparsify(tensor, mask, with_batch_size=False):
    shape = tensor.shape
    shape = torch.tensor(shape)

    mask = mask.reshape(-1)
    sparse = tensor.reshape(-1)[mask]
    if with_batch_size:
        sparse = sparse.reshape(shape[0], -1)
    else:
        sparse = sparse.unsqueeze(0)

    # add bits to make it divisible by 8
    if mask.shape[0] % 8 != 0:
        add_bits = 8 - (mask.shape[0] % 8)
        mask = torch.cat([mask, torch.zeros(add_bits, dtype=mask.dtype, device=mask.device)], dim=0)

    # mask = packbit.packbits_padded(mask)

    # idle value
    # mask = torch.ones(1, device=tensor.device)
    # sparse = tensor

    return shape, mask, sparse

def unsparsify(shape, mask, sparse, with_batch_size=False):
    # mask = packbit.unpackbits_padded(mask).to(dtype=torch.bool)
    mask = mask.to(dtype=torch.bool)

    if with_batch_size:
        sparse = sparse.view(-1)
    else:
        sparse = sparse.squeeze(0)

    shape = torch.Size(shape)
    dense = torch.zeros(shape.numel(), device=sparse.device, dtype=sparse.dtype)
    dense[mask[:shape.numel()]] = sparse

    return dense.reshape(shape)

    # idle
    # return sparse

