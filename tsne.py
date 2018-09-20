from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
#import input_data
import heapq as hpq
import pickle as pk

iris = load_iris()
with open(r'tsnedata.pkl','rb') as f:
    data_dic = pk.load(f)
#print(data_dic)
a = data_dic['h_conv1_rsp']
#wa = data_dic['wf']
#wafa = data_dic['h_conv3_rsp']
#print(a)
data = []
hfc = 0
wfc = 1
data_tmp = []

if(hfc == 0):
    for i in range(len(a)):
        data_tmp = []
        for j in range(len(a[0])):
            for k in range(len(a[0][0])):
                data_tmp.append(a[i][j][k])
        data.append(list(data_tmp))
        '''
else:
    for i in range(len(a)):
        data_tmp = []
        for j in range(len(a[0])):
            data_tmp.append(a[i][j])
        data.append(list(data_tmp))
if(wfc == 1):
    wl1 = list(wa[0:116,0])
    wl2 = list(wa[0:116,1])
    wafa = list(wafa[0,:])
    #wl1 = [abs(i) for i in wl1]
    #wl2 = [abs(i) for i in wl2]
    large_index = hpq.nlargest(25, wl1)
    aaa = [wl1.index(large_index[i])+1 for i in range(25)]
    print(hpq.nlargest(25,aaa))


    large_index = hpq.nlargest(25, wafa)
    aaa = [wafa.index(large_index[i])+1 for i in range(25)]
    print(hpq.nlargest(25,aaa))
#print(type(data_tmp[0]))
#print(type(data))
data = np.array(data)
#(type(data))
#print(data.shape)
'''
label = data_dic['label']
#print(list(label[:,0]))
X_tsne = TSNE(learning_rate=100).fit_transform(data_dic['x'])
X_pca = PCA().fit_transform(data)

'''
print(len(wl1))
plt.scatter([i for i in range(116)],wl1)
plt.show()
'''
TFlist = list(label[:,0]==1)
print(TFlist)
cmark = []
for k in range(len(TFlist)):
    if(TFlist[k]):
        cmark.append('line1')
    else:
        cmark.append('line2')
plt.figure(figsize=(10, 5))
axes = plt.subplot(121)
type1 = plt.scatter(X_tsne[0:2, 0], X_tsne[0:2, 1], c='#FDE725', label='Patient')
type2 = plt.scatter(X_tsne[4:5, 0], X_tsne[4:5, 1], c='k', label='Control')
plt.legend()
plt.show()
