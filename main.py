import numpy as np
from fc import fc
from bc import bc
'''from cost import cost'''
from accuracy import accuracy
import time
from utils import mnist_reader

def to_categorical(y):
    ans=[]
    for idx in y:
        t=np.zeros(10)
        t[idx]=1
        ans.append(t)
    return np.array(ans).T

x_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
x_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

x_train = x_train.reshape(60000, 784).T
x_test = x_test.reshape(10000, 784).T
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

'''print(y_train)'''


input_size = 28*28
train_size = x_train.shape[1]

test_size = x_test.shape[1]

alpha = 0.1
max_iter = 2
mini_batch = 128
layer_size = np.array([[input_size,0],[0,512],[0,512],[0,10]])
layer_size = np.array([input_size,512,512,10])
L = layer_size.shape[0]

sigm = lambda s: 1/(1+np.exp(-s))
dsigm = lambda s: sigm(s)*(1-sigm(s))
lin = lambda s: s
dlin = lambda s: 1

#relu= lambda s: s if s>0 else 0

def relu(m):
    '''return np.maximum(0, m)'''
    x,y=m.shape
    for i in range(x):
        for j in range(y):
            if m[i,j]<0:
                m[i,j]=0
    return m

def drelu(m):
    '''return -np.minimum(0, m)'''
    x,y=m.shape
    for i in range(x):
        for j in range(y):
            if m[i,j]<0:
                m[i,j]=0
    return m

#relu= lambda s: np.where(s>0,s,0)
#drelu= lambda s: np.where(s>0,1,0)

def softmax(m):
    y=m.shape[1]
    for i in range(y):
        #input(m[:,i])
        max_row=np.max(m[:,i])
        #input(m[:,i]-max_row)
        #input(np.exp(m[:,i]-max_row))
        m[:,i]=np.exp(m[:,i]-max_row)/np.sum(np.exp(m[:,i]-max_row))
        #input(m[:,i])
    '''print(m.shape)'''
    '''input(m)'''
    return m

#softmax=lambda s: np.exp(s-np.max(s))/np.sum(np.exp(s-np.max(s)))
dsoftmax=lambda s: 1

fs = [[],relu,relu,softmax,sigm,sigm]
dfs = [[],drelu,drelu,dsoftmax,dsigm,dsigm]

def cost(t,y):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
w = []
for l in range(L-1):
    #w.append((np.random.rand(layer_size[l+1,1],np.sum(layer_size[l,:]))*2-1)*np.sqrt(6/(layer_size[l+1,1]+np.sum(layer_size[l,:]))))
    w.append(np.random.rand(layer_size[l+1],layer_size[l]))
w = np.array(w)
J = np.array([])
x = list(np.zeros((L,1)))
a = list(np.zeros((L,1)))
z = list(np.zeros((L,1)))
delta = list(np.zeros((L,1)))

for iter in range(max_iter):
    print('max_iter=%d' % (max_iter))
    ind = np.random.permutation(train_size)
    for k in range(int(np.ceil(train_size/mini_batch))):
        a[0] = x_train[:,ind[k*mini_batch:min((k+1)*mini_batch,train_size)]]
        y=y_train[:,ind[k*mini_batch:min((k+1)*mini_batch,train_size)]]

        for l in range(L-1):
            a[l+1],z[l+1] = fc(w[l],a[l],fs[l+1])
            tt=np.all(a[l+1]==0,axis=0) #should be all false

            #print(tt.any()) #should be false
        '''print(a[L-1])'''
        J = np.append(J,1/mini_batch*cost(a[L-1],y))
        delta[L-1] = ([a[L-1]-y]*dfs[L](z[L-1]))[0]
        for l in range(L-2,0,-1):
            delta[l] = bc(w[l],z[l],delta[l+1],dfs[l])
        for l in range(L-1):
            gw = np.matmul(delta[l+1],a[l].T)/mini_batch
            w[l] = w[l] - alpha*gw
            '''print("test wwwwwww变没变")'''
            print(w)
    if iter%1==0:
        print('%d/%d epochs: J=%.4f' % (iter,max_iter,J[-1]))

a[1] = x_train;
for l in range(L-1):
  a[l+1],z[l+1] = fc(w[l], a[l],fs[l+1])
train_acc = accuracy(a[L-1], y_train);
print('Accuracy on training dataset is %.4f%%\n' % (train_acc*100));

'''test on testing set'''
'''a{1} = X_test;
for l = 1:L-1
   a{l+1} = fc(w{l}, a{l});
end
test_acc = accuracy(a{L}, y_test);
fprintf('Accuracy on testing dataset is %f%%\n', test_acc*100);'''

