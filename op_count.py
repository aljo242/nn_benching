import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd

multiply_adds = 1


def zero_ops(m, x, y):
    m.total_ops += torch.Tensor([int(0)])


def count_convNd(m: _ConvNd, x: (torch.Tensor,), y: torch.Tensor):
    x = x[0]

    kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh
    bias_ops = 1 if m.bias is not None else 0

    # N x Cout x H x W x  (Cin x Kw x Kh + bias) 
    # here we count the FP add as a FLOP
    total_ops = y.nelement() * (m.in_channels // m.groups * kernel_ops + bias_ops)

    m.total_ops += torch.Tensor([int(total_ops)])



# batch normalization 
def count_bn(m, x, y):
    x = x[0]

    nelements = x.numel() 
    if not m.training:
        # we already will have learned a mean and var 
        # and therefore do not calculate one at inference time 
        # xi_hat = (xi - mu)/sqrt(sigma + epsilon)
        # simiplifies  to 2 operations (sub and div)
        total_ops = 2 * nelements

    m.total_ops += torch.Tensor([int(total_ops)])


# relu is just the number of elements in the input tensor
def count_relu(m, x, y):
    x = x[0]

    nelements = x.numel()

    m.total_ops += torch.Tensor([int(nelements)])


# get shape of the tensor then implement as follows
def count_softmax(m, x, y):
    x = x[0]

    batch_size, nfeatures = x.size()

    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)

    m.total_ops += torch.Tensor([int(total_ops)])


def count_avgpool(m, x, y):
    # total_add = torch.prod(torch.Tensor([m.kernel_size]))
    # total_div = 1
    # kernel_ops = total_add + total_div
    kernel_ops = 1
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])



def count_adap_avgpool(m, x, y):
    kernel = torch.Tensor([*(x[0].shape[2:])]) // torch.Tensor(list((m.output_size,))).squeeze()
    total_add = torch.prod(kernel)
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])


# in PyTorch we call FC layers linear
def count_fullyConnected(m, x, y):
	# one add and one mul per input node
    total_mul = m.in_features
    total_add = m.in_features - 1
    total_add += 1 if m.bias is not None else 0
    output_size = y.numel()
    # each input node is connected (output_size) times 
    total_ops = (total_mul + total_add) * output_size

    m.total_ops += torch.Tensor([int(total_ops)])


def count_upsample(m, x, y):
    if m.mode not in ("nearest", "linear", "bilinear", "bicubic",):  # "trilinear"
        return zero_ops(m, x, y)

    if m.mode == "nearest":
        return zero_ops(m, x, y)

    x = x[0]
    if m.mode == "linear":
        total_ops = y.nelement() * 5  # 2 muls + 3 add
    elif m.mode == "bilinear":
        # https://en.wikipedia.org/wiki/Bilinear_interpolation
        total_ops = y.nelement() * 11  # 6 muls + 5 adds
    elif m.mode == "bicubic":
        # https://en.wikipedia.org/wiki/Bicubic_interpolation
        # Product matrix [4x4] x [4x4] x [4x4]
        ops_solve_A = 224  # 128 muls + 96 adds
        ops_solve_p = 35  # 16 muls + 12 adds + 4 muls + 3 adds
        total_ops = y.nelement() * (ops_solve_A + ops_solve_p)
    elif m.mode == "trilinear":
        # https://en.wikipedia.org/wiki/Trilinear_interpolation
        # can viewed as 2 bilinear + 1 linear
        total_ops = y.nelement() * (13 * 2 + 5)

    m.total_ops += torch.Tensor([int(total_ops)])