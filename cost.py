import numpy as np

def cost(a,y):
    '''loss=0
    log_a=np.log(a)
    return -np.sum(y*log_a)
    return np.sum(-1/np.log(a/y))'''
    return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

