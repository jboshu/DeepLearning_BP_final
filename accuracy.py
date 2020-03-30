import numpy as np

'''def accuracy(a,y):
    mini_batch = np.size(a,1)
    idx_a = np.max(a)
    idx_y = np.max(y)
    acc = sum(idx_a==idx_y) / mini_batch;
    return acc'''
def accuracy(a,y):
    mini_batch = a.shape[1]
    ind_x = a.max(axis=0)
    ind_y = y.max(axis=0)
    total = 0
    for i in range(mini_batch):
        if ind_x[i] == ind_y[i]:
            total = total + 1
    return total/mini_batch