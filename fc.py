import numpy as np

def fc(w,a,f):
    t = a
    
    z = np.matmul(w,t)
    
    return f(z),z