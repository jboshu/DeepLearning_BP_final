import numpy as np
def bc(w,z,delta_next,df):
    #print("z.shape=%d,%d" % z.shape)
    #print("(w[:,-(z.shape[0])-1:-1]).T.shape=%d,%d"%(w[:,w.shape[1]-(z.shape[0]):w.shape[1]]).T.shape)
    #print("delta_next.shape=%d,%d"%delta_next.shape)
    delta = df(z) * np.matmul((w[:,w.shape[1]-(z.shape[0]):w.shape[1]]).T,delta_next)
    #print("delta.shape=")
    #print(delta.shape)
    return delta