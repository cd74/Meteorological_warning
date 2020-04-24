import pandas as pd
import time
import datetime
import numpy as np
mode = 'train'
f = open('hko7_rainy_{}_days.txt'.format(mode),'r')
l,l1 =[],[]
for i in f:
    s = i.split(',')
    if (float(s[1])>1):
        l.append(s[0])
        l1.append(i)

dates = []
for s in l:
    t=time.strptime(s, "%Y%m%d")
    for i in range(0,240):
        stamp = int(time.mktime(t))+360*i
        res = datetime.datetime.fromtimestamp(stamp)
        res = res.strftime("%Y-%m-%d %H:%M:%S")
        dates.append(res)

data = {'has_rainfall':np.zeros(len(dates))}        
df = pd.DataFrame(data,index=dates,columns=['has_rainfall'])    
df.to_pickle('new_rainy_{}.pkl'.format(mode))

a = pd.read_pickle('new_rainy_{}.pkl'.format(mode))

f = open('rainy_{}.txt'.format(mode),'w')
for i in l1:
    f.writelines(i)
f.close()
