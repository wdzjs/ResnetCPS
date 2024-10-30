import torch
import torch.nn as nn
import time

from collections import Counter

from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models
from audtorch.metrics.functional import pearsonr
from itertools import permutations

def tensor_unfold(a):
    b = model.state_dict()[a]
    print(b.shape)
    return b.reshape(b.shape[0], -1).t().contiguous()

def p_corr(a, b):
    if a.shape != b.shape:
        print('Dims are not same.')
        return
    sum = list()
    count = 0
    for i in range(a.shape[1]):
        for j in range(a.shape[1]):
            tmp = torch.abs(pearsonr(a[:, i], b[:, j])).numpy().take(0)
            sum.append(tmp)
            if tmp > 0.99:
                count += 1
    print(a.shape[1])
    print(count)
    return sum

def s_corr(a):
    sum = list()
    count = 0
    for i in range(a.shape[1]):
        for j in range(a.shape[1]):
            if i==j:
                continue
            tmp = torch.abs(pearsonr(a[:, i], a[:, j])).numpy().take(0)
            sum.append(tmp)
            if tmp > 0.99:
                count += 1
    print(a.shape[1])
    print(count)
    return sum

model = create_model(model_name="resnet50", checkpoint_path=
r"E:\code\pytorch-image-models-master\pytorch-image-models-master\util\model_best.pth-c7cc8fbe.pth")

name1 = 'layer3.1.conv2.weight'
name2 = 'layer3.2.conv2.weight'

start = time.time()
x = tensor_unfold(name1)
y = tensor_unfold(name2)
#print(model.state_dict())
#print(list(permutations(range(x.shape(0)))))
sum = s_corr(y)
result = Counter(sum)

import matplotlib.pyplot as plt
#import numpy as np

plt.hist(sum, bins=100)
plt.show()
print(max(sum))

'''
print(sum)
sum = p_corr(x, x)
print(sum)
sum = p_corr(y, y)
print(sum)
end = time.time()
print('time:', end-start)
'''