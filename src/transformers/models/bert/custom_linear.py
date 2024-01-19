
import sys
import logging

# from mesa import custom_quant
# from mesa import native
# /disk3/Haonan/yanbo_random/bert_finetune/transformers/src/transformers/models/bert/custom_linear.py
sys.path.insert(0, '/disk3/Haonan/yanbo_random/bert_finetune/transformers/src/transformers/models/bert')
from rand_layers import sparsify, unsparsify
from pdb import set_trace
import torch
import torch.nn as nn
import torch.nn.functional as F

# ================== our method ==================
class OurSparseLinear(torch.nn.Linear):
    def __init__(self, *args, keep_frac=0.5, linear_idx = None, act_type = None, **kwargs):
        super(OurSparseLinear, self).__init__(*args, **kwargs)
        self.keep_frac = keep_frac
        self.step_idx = 0
        self.linear_idx = linear_idx
        self.act_type = act_type

    def forward(self, input, retain=False, skip_rand=False):
        if not retain:
            self.random_seed = torch.randint(low=10000000000, high=99999999999, size=(1,))

        if skip_rand:
            keep_frac = 1.0
        else:
            keep_frac = self.keep_frac

        result = SparseMatMul.apply(input, self.weight, self.bias, keep_frac)
        cal_zero_ratio(result, self.linear_idx, self.step_idx, self.act_type)
        # print('Ourlinear step idx + 1')
        self.step_idx += 1
        return result

def cal_zero_ratio(layer_output, layer_idx, iteration, act_type):
    temp_total = float(layer_output.view(-1).shape[0])
    temp_act = torch.sum(torch.eq(layer_output, 0).float()) #eq: equal to 0
    ratio = temp_act/temp_total
    with open(f'zero-ratio-of-bert-layer{act_type}.txt', 'a+') as file:
        file.write(f"iteration:{iteration};layer{layer_idx}:{ratio}\n")


import rand_layers as rl
# 单个hidden state
class SparseMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, keep_frac):
        # Calculate dimensions according to input and keep_frac
        num_activations = input.size()[-1]

        ctx.input_shape = tuple(input.size())
        ctx.keep_frac = keep_frac
        if ctx.keep_frac == 1.0:
            ctx.save_for_backward(input, weight, bias)
            linear_out = F.linear(input, weight, bias=bias)
            return linear_out

        sparse_input,_ = rl.input2sparse(input, keep_frac)
        ctx.save_for_backward(sparse_input, weight, bias)

        with torch.autograd.grad_mode.no_grad():
            return F.linear(input, weight, bias=bias)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.keep_frac < 1.0:
            sparse_input, weight, bias = ctx.saved_tensors
            input = rl.sparse2input(sparse_input, ctx.input_shape)
        else:
            input, weight, bias = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            input_grad = grad_output.matmul(weight.to(dtype=grad_output.dtype))
        if ctx.needs_input_grad[1]:
            weight_grad = grad_output.transpose(-2, -1).matmul(input.to(dtype=grad_output.dtype))
        if bias is not None and ctx.needs_input_grad[2]:
            bias_grad = grad_output.sum(0)
        return input_grad, weight_grad, bias_grad, None, None, None, None


class OurSparseMatMul(nn.Module):
    def __init__(self, args=None, keep_frac=0.5, linear_idx = None, act_type = None ):
        super(OurSparseMatMul, self).__init__()
        self.keep_frac = keep_frac
        self.step_idx = 0
        self.linear_idx = linear_idx
        self.act_type = act_type
    def forward(self, x1, x2):
        y = DoubleSparseMatMul.apply(x1, x2, self.keep_frac)
        return y

import time

class DoubleSparseMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, to_matmul_input2, keep_frac):
        input2 = to_matmul_input2.transpose(-2, -1).contiguous() # input2 (64, 512)->(512, 64)
        ctx.input_shape1 = tuple(input1.size())
        ctx.input_shape2 = tuple(input2.size())
        sparse_input1, gather_index = rl.input2sparse(input1, keep_frac)
        # 这里为什么不能直接将1计算得到的index用到2，因为1计算为0的维度，乘以2必为0.因此可以从2去掉。
        # 原因是如果1为512，512， 而2为512，64，那么他们的index的值大小是不同的，对1的索引在2中可能会溢出。
        # 因此这里要么维度严格一致，要么就得重新计算2的维度。
        # 如果维度严格一致，就只能稀疏最后一维。
        sparse_input2 = rl.get_ref_dim_reduced_input(input2, gather_index, input1.shape)
        # sparse_input2 = rl.input2sparse(input2, keep_frac) # here we reuse the dim-reduced index of input1
        ctx.save_for_backward(sparse_input1, sparse_input2)
        output = input1.matmul(to_matmul_input2)
        return output
# backward很慢。
    @staticmethod
    def backward(ctx, grad_output):
        grad_input1 = grad_input2 = None    
        sparse_input1, sparse_input2 = ctx.saved_tensors
        input_shape1 = ctx.input_shape1
        input_shape2 = ctx.input_shape2
        input1 = rl.sparse2input(sparse_input1, input_shape1)
        input2 = rl.sparse2input(sparse_input2, input_shape2)
        # print('grad shape : ',grad_output.shape, input1.shape, input2.shape)
        if ctx.needs_input_grad[0]:
            grad_input1 = grad_output.matmul(input2.to(dtype=grad_output.dtype))
        if ctx.needs_input_grad[1]:
            grad_input2 = input1.transpose(-2, -1).to(dtype=grad_output.dtype).matmul(grad_output)
        return grad_input1, grad_input2, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None



# ================== back razor ==================
class linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None, mask=None, quantize=True, half=False, clip_val=None, level=256, iteration=None, ema_decay=None, quant_groups=None, shift=None):
        shape_x, mask_x, sparse_x = sparsify(x, mask, with_batch_size=False)

        if half and (not quantize):
            sparse_x = sparse_x.half()

        # if quantize:
        #     custom_quant.Quant.forward(ctx, sparse_x, clip_val, level, iteration, ema_decay, quant_groups, shift)
        #     ctx.save_for_backward(weight, bias, shape_x, mask_x)
        # else:
        ctx.save_for_backward(weight, bias, shape_x, mask_x, sparse_x)

        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_weight = grad_bias = None

        tensors = ctx.saved_tensors

        # if len(tensors) == 5:
        weight, bias, shape_x, mask_x, sparse_x = tensors
        # else:
        #     weight, bias, shape_x, mask_x = tensors
        #     sparse_x = custom_quant.Quant.restore(ctx)

        sparse_x = sparse_x.float()
        input = unsparsify(shape_x, mask_x, sparse_x, with_batch_size=False)

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight.to(dtype=grad_output.dtype))

        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.transpose(-2, -1).matmul(input.to(dtype=grad_output.dtype))

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None


# class LinearSparse(nn.Linear, custom_quant.Quant):
class LinearSparse(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, args=None, logger=None, quant_groups=1, masker=None,
                 quantize=True, half=False, act_prune=False):
        super(LinearSparse, self).__init__(in_features, out_features, bias=bias)
        # custom_quant.Quant.__init__(self, args=args, logger=logger, quant_groups=quant_groups)
        self.masker = masker
        self.quantize = quantize
        self.act_prune = act_prune
        self.half = half
        self.tag = 'fc'

    def __repr__(self):
        return self.__str__()

    def forward(self, x):
        # print("type(x) is {}".format(type(x)))
        if self.masker is not None and self.training:
            mask = self.masker(x)
            # print("mask sum is {}".format((~mask).sum()))
            if self.act_prune:
                x = x * mask
            y = linear.apply(x, self.weight, self.bias, mask, self.quantize, self.half, self.clip_val, self.level,
                             self.iteration, self.ema_decay, self.quant_groups, self.shift)
        else:
            y = F.linear(x, self.weight, self.bias)
        return y

