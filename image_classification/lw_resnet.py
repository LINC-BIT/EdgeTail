'''ConvNet-AIG in PyTorch.

Residual Network is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

Adaptive Inference Graphs is from the original ConvNet-AIG paper:
[2] Andreas Veit, Serge Belognie
    Convolutional Networks with Adaptive Inference Graphs. ECCV 2018

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable
import re
from torch.autograd import Variable
from torch import autograd
from torch.nn import functional as F
from functools import partial
from torch.nn.modules.pooling import AvgPool2d


# from gumbelmodule import GumbleSoftmax


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


# class Sequential_ext(nn.Module):
#     """A Sequential container extended to also propagate the gating information
#     that is needed in the target rate loss.
#     """

#     def __init__(self, *args):
#         super(Sequential_ext, self).__init__()
#         if len(args) == 1 and isinstance(args[0], OrderedDict):
#             for key, module in args[0].items():
#                 self.add_module(key, module)
#         else:
#             for idx, module in enumerate(args):
#                 self.add_module(str(idx), module)

#     def __getitem__(self, idx):
#         if not (-len(self) <= idx < len(self)):
#             raise IndexError('index {} is out of range'.format(idx))
#         if idx < 0:
#             idx += len(self)
#         it = iter(self._modules.values())
#         for i in range(idx):
#             next(it)
#         return next(it)

#     def __len__(self):
#         return len(self._modules)

#     def forward(self, input, temperature=1, openings=None):
#         gate_activations = []
#         for i, module in enumerate(self._modules.values()):
#             input, gate_activation = module(input, temperature)
#             gate_activations.append(gate_activation)
#         return input, gate_activations


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        # Gate layers
        # self.fc1 = nn.Conv2d(in_planes, 16, kernel_size=1)
        # self.fc1bn = nn.BatchNorm2d(16)
        # self.fc2 = nn.Conv2d(16, 2, kernel_size=1)
        # initialize the bias of the last fc for 
        # initial opening rate of the gate of about 85%
        # self.fc2.bias.data[0] = 0.1
        # self.fc2.bias.data[1] = 2
        # self.gs = GumbleSoftmax()
        # self.gs.cuda()

    def forward(self, x):
        # Compute relevance score
        # w = F.avg_pool2d(x, x.size(2))
        # w = F.relu(self.fc1bn(self.fc1(w)))
        # w = self.fc2(w)
        # # Sample from Gumble Module
        # w = self.gs(w, temp=temperature, force_hard=True)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.shortcut(x) + out
        out = F.relu(out)
        # Return output of layer and the value of the gate
        # The value of the gate will be used in the target rate loss
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        # Gate layers
        # self.fc1 = nn.Conv2d(in_planes, 16, kernel_size=1)
        # self.fc1bn = nn.BatchNorm2d(16)
        # self.fc2 = nn.Conv2d(16, 2, kernel_size=1)
        # # initialize the bias of the last fc for 
        # # initial opening rate of the gate of about 85%
        # self.fc2.bias.data[0] = 0.1
        # self.fc2.bias.data[1] = 2

        # self.gs = GumbleSoftmax()
        # self.gs.cuda()

    def forward(self, x):
        # Compute relevance score
        # w = F.avg_pool2d(x, x.size(2))
        # w = F.relu(self.fc1bn(self.fc1(w)))
        # w = self.fc2(w)
        # # Sample from Gumble Module
        # w = self.gs(w, temp=temperature, force_hard=True)

        # TODO: For fast inference, check decision of gate and jump right 
        #       to the next layer if needed.

        out = F.relu(self.bn1(self.conv1(x)), inplace=False)
        out = F.relu(self.bn2(self.conv2(out)), inplace=False)
        out = self.bn3(self.conv3(out))
        out = self.shortcut(x) + out
        out = F.relu(out, inplace=False)
        # Return output of layer and the value of the gate
        # The value of the gate will be used in the target rate loss
        return out

    
class ResNet_ImageNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.in_planes = 64
        super(ResNet_ImageNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        for k, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if 'fc2' in str(k):
                    # Initialize last layer of gate with low variance
                    m.weight.data.normal_(0, 0.001)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride=1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, out):
        # gate_activations = []
        out = self.relu(self.bn1(self.conv1(out)))
        out = self.maxpool(out)
        out, a = self.layer1(out)
        # gate_activations.extend(a)
        out, a = self.layer2(out)
        # gate_activations.extend(a)
        out, a = self.layer3(out)
        # gate_activations.extend(a)
        out, a = self.layer4(out)
        # gate_activations.extend(a)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNet_cifar(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_cifar, self).__init__()
        self.in_planes = 16

        self.conv1 = conv3x3(3,16)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64*block.expansion, num_classes)

        # for k, m in self.named_modules():
        #     if isinstance(m, nn.Conv2d):
        #         if 'fc2' in str(k):
        #             # Initialize last layer of gate with low variance
        #             m.weight.data.normal_(0, 0.001)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # gate_activations = []
        #print(x)
        #print(x.size())
        if x.ndim==5:
         x=x[0]
        #print(x.size())

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        # gate_activations.extend(a)
        out = self.layer2(out)
        # gate_activations.extend(a)
        out = self.layer3(out)
        # gate_activations.extend(a)
        out = F.avg_pool2d(out, 8)
        #print(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

import torch
class ResNet_cifarF(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_cifarF, self).__init__()
        self.in_planes = 16

        self.conv1 = conv3x3(3,16)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64*block.expansion, num_classes)
        self.out1= torch.tensor(0)
        # for k, m in self.named_modules():
        #     if isinstance(m, nn.Conv2d):
        #         if 'fc2' in str(k):
        #             # Initialize last layer of gate with low variance
        #             m.weight.data.normal_(0, 0.001)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # gate_activations = []
        #print(x)
        #print(x.size())
        if x.ndim==5:
         x=x[0]
        #print(x.size())

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        # gate_activations.extend(a)
        out = self.layer2(out)
        # gate_activations.extend(a)
        out = self.layer3(out)
        # gate_activations.extend(a)
        out = F.avg_pool2d(out, 8)
        #print(out)
        self.out1 = out.view(out.size(0), -1)
        out = self.linear(self.out1)
        return out

    def features(self):
        return torch.sum(torch.abs(self.out1 ), 1).reshape(-1, 1)


    
class ResNet_cifar_w2(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_cifar_w2, self).__init__()
        self.in_planes = 16*2

        self.conv1 = conv3x3(3,16*2)
        self.bn1 = nn.BatchNorm2d(16*2)
        self.layer1 = self._make_layer(block, 16*2, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64*2, num_blocks[2], stride=2)
        self.linear = nn.Linear(64*2*block.expansion, num_classes)

        # for k, m in self.named_modules():
        #     if isinstance(m, nn.Conv2d):
        #         if 'fc2' in str(k):
        #             # Initialize last layer of gate with low variance
        #             m.weight.data.normal_(0, 0.001)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # gate_activations = []
        if x.ndim==5:
         x=x[0]
         #print('fsa')
        #print('adsdad')
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        # gate_activations.extend(a)
        out = self.layer2(out)
        # gate_activations.extend(a)
        out = self.layer3(out)
        # gate_activations.extend(a)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    
class ResNet_cifar_w0_5(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_cifar_w0_5, self).__init__()
        self.in_planes = 16//2

        self.conv1 = conv3x3(3,16//2)
        self.bn1 = nn.BatchNorm2d(16//2)
        self.layer1 = self._make_layer(block, 16//2, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32//2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64//2, num_blocks[2], stride=2)
        self.linear = nn.Linear(64//2*block.expansion, num_classes)

        # for k, m in self.named_modules():
        #     if isinstance(m, nn.Conv2d):
        #         if 'fc2' in str(k):
        #             # Initialize last layer of gate with low variance
        #             m.weight.data.normal_(0, 0.001)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # gate_activations = []
        if x.ndim==5:
         x=x[0]
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        # gate_activations.extend(a)
        out = self.layer2(out)
        # gate_activations.extend(a)
        out = self.layer3(out)
        # gate_activations.extend(a)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNet_TinyImageNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=200):
        super(ResNet_TinyImageNet, self).__init__()
        self.in_planes = 64

        # 初始卷积层和池化层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 四个 ResNet 模块
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # 平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应池化到 1x1
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.lamda = 400

        # 初始化权重
        for k, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)  # 展平
        out = self.linear(out)
        # if AUG == 4:
        #     out = self.fc(out1)
        #     out_cb = self.linear(out1)
        #     z = self.projection_head(out1)
        #     p = self.contrast_head(z)
        #     return out, out_cb, z, p
        return out
    def setGLMC(self,block=BasicBlock,num_classes=200):
        hidden_dim = 256
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.contrast_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.projection_head = nn.Sequential(
            nn.Linear(512 * block.expansion, hidden_dim),
        )

    def forwardGLMC(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out1 = torch.flatten(out, 1)  # 展平
        out = self.fc(out1)
        out_cb = self.linear(out1)
        z = self.projection_head(out1)
        p = self.contrast_head(z)
        return out, out_cb, z, p




    def ewc_loss(self, cuda=False):
        # for name, param in self.named_parameters():
        #     print(f"Parameter name: {name}, Shape: {param.shape}")

        try:
            losses = []
            for n, p in self.named_parameters():
                # retrieve the consolidated mean and fisher information.
                n = n.replace('.', '__')
                mean = getattr(self, '{}_mean'.format(n))
                fisher = getattr(self, '{}_fisher'.format(n))
                # wrap mean and fisher in variables.
                mean = Variable(mean)
                fisher = Variable(fisher)
                # calculate a ewc loss. (assumes the parameter's prior as
                # gaussian distribution with the estimated mean and the
                # estimated cramer-rao lower bound variance, which is
                # equivalent to the inverse of fisher information)
                losses.append((fisher * (p - mean) ** 2).sum())
                #print('loss:',(self.lamda / 2) * sum(losses))
            print("sumloss:",sum(losses))
            print("lamda",self.lamda)
            return (self.lamda / 2) * sum(losses)
        except AttributeError:
            # ewc loss is 0 if there's no consolidated parameters.
            #print("ewc loss is 0 if there's no consolidated parameters")
            return (
                Variable(torch.zeros(1)).cuda() if cuda else
                Variable(torch.zeros(1))
            )

    def estimate_fisher(self, dataset, sample_size,scenario, batch_size=8):
        # sample loglikelihoods from the dataset.
        data_loader = scenario.build_dataloader(dataset, batch_size,0,True, False)
        loglikelihoods = []
        for x, y in data_loader:

            x = Variable(x).cuda() if self._is_on_cuda() else Variable(x)
            y = Variable(y).cuda() if self._is_on_cuda() else Variable(y)
            loglikelihoods.append(
                F.log_softmax(self(x), dim=1)[range(batch_size), y.data]
            )
            if len(loglikelihoods) >= sample_size // batch_size:
                break
        # estimate the fisher information of the parameters.
        loglikelihoods = torch.cat(loglikelihoods).unbind()
        # loglikelihood_grads = zip(*[autograd.grad(
        #     l, self.parameters(),
        #     retain_graph=(i < len(loglikelihoods))
        # ) for i, l in enumerate(loglikelihoods, 1)])
        # for i, l in enumerate(loglikelihoods):
        #     print(f"loglikelihood {i}: requires_grad={l.requires_grad}, grad_fn={l.grad_fn}")
        #loglikelihood_grads = []

        loglikelihood_grads = zip(*[autograd.grad(
                l,self.parameters(),  # 只选择相关参数
                retain_graph=(i < len(loglikelihoods))
        )for i, l in enumerate(loglikelihoods, 1)])

        #print('loglikelihood_grads：', loglikelihood_grads)
        # for i, grad in enumerate(loglikelihood_grads):
        #     if grad is None:
        #         print(f"Element {i} is None")
        loglikelihood_grads = [torch.stack(gs) for gs in loglikelihood_grads]
        #print('loglikelihood_grads：',loglikelihood_grads)
        fisher_diagonals = [(g ** 2).mean(0) for g in loglikelihood_grads]
        #print('fisher_diagonals:',fisher_diagonals)
        param_names = [
            n.replace('.', '__') for n, p in self.named_parameters()
        ]
        #print('param_names:',param_names)
        #print('fish:',{n: f.detach() for n, f in zip(param_names, fisher_diagonals)})
        return {n: f.detach() for n, f in zip(param_names, fisher_diagonals)}

    def consolidate(self, fisher):
        #print(f"Keys in fisher: {fisher.keys()}")

        for n, p in self.named_parameters():
            n = n.replace('.', '__')
            self.register_buffer('{}_mean'.format(n), p.data.clone())
            self.register_buffer('{}_fisher'
                                 .format(n), fisher[n].data.clone())
        # print("Generated Buffers:")
        # for name, buffer in self.named_buffers():
        #     if '_mean' in name or '_fisher' in name:
        #         print(f"Buffer Name: {name}")
    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda



def ResNet110_cifar(nclass=10): # name rule: num_blocks * 2 + 2
    return ResNet_cifar(BasicBlock, [18,18,18], num_classes=nclass)

def ResNet34_cifar(nclass=10):
    return ResNet_cifar(BasicBlock, [4, 6, 6], num_classes=nclass)

def ResNet56_cifar(nclass=10):
    return ResNet_cifar(BasicBlock, [9,9,9], num_classes=nclass)

def ResNet56_cifar_w0_5(nclass=10):
    return ResNet_cifar_w0_5(BasicBlock, [9,9,9], num_classes=nclass)

def ResNet56_cifar_w2(nclass=10):
    return ResNet_cifar_w2(BasicBlock, [9,9,9], num_classes=nclass)

def ResNet20_cifar(nclass=10):
    return ResNet_cifar(BasicBlock, [3, 3, 3], num_classes=nclass)

def ResNet20_cifarF(nclass=10):
    return ResNet_cifarF(BasicBlock, [3, 3, 3], num_classes=nclass)

def ResNet50_ImageNet():
    return ResNet_ImageNet(Bottleneck, [3,4,6,3])

def ResNet101_ImageNet():
    return ResNet_ImageNet(Bottleneck, [3,4,23,3])

def ResNet152_ImageNet():
    return ResNet_ImageNet(Bottleneck, [3,8,36,3])

def ResNet18_TinyImageNet():
    return ResNet_TinyImageNet(BasicBlock, [2, 2, 2, 2], num_classes=200)

def ResNet34_TinyImageNet():
    return ResNet_TinyImageNet(BasicBlock, [3, 4, 6, 3], num_classes=200)

# ResNet56 配置
def ResNet56_TinyImageNet():
    return ResNet_TinyImageNet(BasicBlock, [9, 9, 9, 9], num_classes=200)

if __name__ == '__main__':
    i = torch.rand((1, 3, 32, 32))
    ResNet56_cifar_w0_5()(i)
    ResNet56_cifar_w2()(i)