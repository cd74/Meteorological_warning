import numpy as np 
import math
data = np.load('batch.npy')[5:25]
mask = np.load('mask.npy')[5:25]
out = np.load('out.npy')
thresh = [0.5,2,5,10,30]
tp,tn,fp,fn = [0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]
def dbz(x):
    d = (x-0.5)*70/255-10
    r = 10**((d-10*math.log(58.53,10))/10/1.56)
    return r

for i in range(0,5):
    for j in range(out.shape[0]):
        for k in range (out.shape[3]):
            for l in range (out.shape[4]):                           
                r1 = dbz(out[j][0][0][k][l])
                r2 = dbz(data[j][0][0][k][l])
                a1= 1 if(r1>thresh[i]) else 0
                a2= 1 if(r2>thresh[i]) else 0
                if(mask[j][0][0][k][l]==1):
                    if(a1== 1 and a2== 1):
                        tp[i]+=1
                    if(a1== 0 and a2== 0):
                        tn[i]+=1
                    if(a1== 1 and a2== 0):
                        fp[i]+=1
                    if(a1== 0 and a2== 1):
                        fn[i]+=1
print(tp,tn,fp,fn)
for i in range (0,5):
    hss=(tp[i]*tn[i]-fp[i]*fn[i])/((tp[i]+fn[i])*(tn[i]+fn[i])+(tp[i]+fp[i])*(tn[i]+fp[i]))
    print(hss)
pass#[160207, 0, 0, 0, 0] [2036319, 2601976, 2601976, 2601976, 2601976] [65193, 0, 0, 0, 0] [340257, 0, 0, 0, 0]