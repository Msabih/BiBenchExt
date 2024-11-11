import torch
from math import ceil
import torch.nn as nn
import torch.nn.functional as F
from .coding import *
import random

class CBNNConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, dilation=0, transposed=False, output_padding=None, groups=1,k_bits=12):
        super(CBNNConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.k_bits=k_bits
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True)
        else:
            self.bias = None
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.number_of_weights = in_channels * out_channels * kernel_size * kernel_size
        self.shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.rand(*self.shape) * 0.002 - 0.001, requires_grad=True)
        #self.coder = BinaryCodebook(k_bits=self.k_bits)
        self.register_buffer('codebook', torch.empty(256,k_bits))
        self.register_buffer('encoded_vector', torch.empty(ceil(self.number_of_weights/k_bits)))
        self.first_iter=True

    def forward(self, x):
        if self.training:
            if self.first_iter:
                self.coder = BinaryCodebook(k_bits=self.k_bits)
                self.codebook, self.encoded_vector = self.coder.process_weights(self.weight)
                self.first_iter=False
            else:
                if random.random()>.9: 
                 self.encoded_vector=CodebookReplacer.replace_with_codebook( self.weight, self.codebook,self.coder)
            binary_input_no_grad = torch.sign(x)
            cliped_input = torch.clamp(x, -1.0, 1.0)
            x = binary_input_no_grad.detach() - cliped_input.detach() + cliped_input

            real_weights = self.weight.view(self.shape)
            binary_weights_no_grad = CodebookReplacer.weight_builder(self.codebook,self.encoded_vector,self.shape)
            cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
            binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
            y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding,bias=self.bias)
            
            return y
        else:
            if self.first_iter and self.codebook.numel() == 0:
                self.coder = BinaryCodebook(k_bits=self.k_bits)
                self.codebook, self.encoded_vector = self.coder.process_weights(self.weight)
                self.first_iter=False
            x = torch.sign(x)
            binary_weights = CodebookReplacer.weight_builder(self.codebook,self.encoded_vector,self.shape)
            y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding,bias=self.bias)
            return y
           

class CBNNConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, dilation=0, transposed=False, output_padding=None, groups=1,k_bits=12):
        super(CBNNConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.k_bits=k_bits
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True)
        else:
            self.bias = None
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.number_of_weights = in_channels * out_channels * kernel_size
        self.shape = (out_channels, in_channels, kernel_size)
        self.weight = nn.Parameter(torch.rand(*self.shape) * 0.002 - 0.001, requires_grad=True)
        self.register_buffer('codebook', torch.empty(256,k_bits))
        self.register_buffer('encoded_vector', torch.empty(ceil(self.number_of_weights/k_bits)))
        self.first_iter=True
         
    def forward(self, x):
        if self.training:
            if self.first_iter:
                self.coder = BinaryCodebook(k_bits=self.k_bits)
                self.codebook, self.encoded_vector = self.coder.process_weights(self.weight)
                self.first_iter=False            
            else:
                if random.random()>.9: 
                    self.encoded_vector=CodebookReplacer.replace_with_codebook( self.weight, self.codebook,self.coder)
            binary_input_no_grad = torch.sign(x)
            cliped_input = torch.clamp(x, -1.0, 1.0)
            x = binary_input_no_grad.detach() - cliped_input.detach() + cliped_input

            real_weights = self.weight.view(self.shape)
            binary_weights_no_grad = CodebookReplacer.weight_builder(self.codebook,self.encoded_vector,self.shape)
            cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
            binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
            y = F.conv1d(x, binary_weights, stride=self.stride, padding=self.padding,bias=self.bias)
            return y
        else:
            if self.first_iter and self.codebook.numel() == 0:
                self.coder = BinaryCodebook(k_bits=self.k_bits)
                self.codebook, self.encoded_vector = self.coder.process_weights(self.weight)
                self.first_iter=False
            x = torch.sign(x)
            binary_weights = CodebookReplacer.weight_builder(self.codebook,self.encoded_vector,self.shape)
            y = F.conv1d(x, binary_weights, stride=self.stride, padding=self.padding,bias=self.bias)
            return y
        


class BinaryQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output
        grad_input[input[0].gt(1)] = 0
        grad_input[input[0].lt(-1)] = 0
        return grad_input


class BNNLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, binary_act=True):
        super(BNNLinear, self).__init__(in_features, out_features, bias=True)
        self.binary_act = binary_act
        #self.output_ = None
        self.coder = BinaryCodebook(k_bits=12)
        self.register_buffer('codebook', None)
        self.register_buffer('encoded_vector', None)
        self.first_iter=True

    def forward(self, input):

        if self.training:
            if self.first_iter:
                self.codebook, self.encoded_vector = self.coder.process_weights(self.weight)
            else:
                if random.random()>.9: 
                    self.encoded_vector=CodebookReplacer.replace_with_codebook( self.weight, self.codebook,self.coder)
        
            #real_weights = self.weight.view((self.out_features, self.in_features))
            binary_weights_no_grad = CodebookReplacer.weight_builder(self.codebook,self.encoded_vector,(self.out_features, self.in_features))
            cliped_weights = torch.clamp(self.weight, -1.0, 1.0)
            bw = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
          
            self.first_iter=False
            ba = input
            if self.binary_act:
                ba = BinaryQuantize().apply(ba)
            output = F.linear(ba, bw, self.bias)
             #self.output_ = output
            return output

        else:
            if self.first_iter:
                self.codebook, self.encoded_vector = self.coder.process_weights(self.weight)
            bw = CodebookReplacer.weight_builder(self.codebook,self.encoded_vector,(self.out_features, self.in_features))
            ba = input
            if self.binary_act:
                ba = BinaryQuantize().apply(ba)
            output = F.linear(ba, bw, self.bias)
            return output


        