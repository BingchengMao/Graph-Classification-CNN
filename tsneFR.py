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
with open(r'tsnedatawf.pkl','rb') as f:
    data_dic = pk.load(f)
#print(data_dic)
a = data_dic['h_conv1_rsp']
a2 = data_dic['h_conv2_rsp']
a3 = data_dic['h_conv3_rsp']
wa = data_dic['wf']
label = data_dic['label']
pos_idx = label[:,0]==1
print(len(wa[0]))
neg_idx = label[:,0]==0
print(len(wa))
#print(a)

pos_data3 = []
neg_data3 = []
hfc = 0
wfc = 1
data_tmp = []
www = []
ww = []
for j in range(len(wa[0])):
    www = []
    for i in range(len(wa)):
        data_tmp.append(wa[i][j])
        if((i+1)%60 == 0):
            www.append(data_tmp)
            #print(data_tmp)
            data_tmp = []
    ww.append(www)

for i in range(len(a3)):
    data_tmp = []
    for j in range(len(a3[0])):
        for k in range(len(a3[0][0])):
            data_tmp.append(a3[i][j][k])
    if(pos_idx[i]):
        pos_data3.append(list(data_tmp))
    else:
        neg_data3.append(list(data_tmp))
pos_data3 = np.array(pos_data3)
neg_data3 = np.array(neg_data3)
ww = np.array(ww)
#print(len(a3[0]))
#print(len(pos_data3))
sio.savemat('save_w', {'w':ww})

'''
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

#print(list(label[:,0]))
X_tsne = TSNE(learning_rate=100).fit_transform(data_dic['x'])
X_pca = PCA().fit_transform(data)
print(len(wl1))
plt.scatter([i for i in range(116)],wl1)
plt.show()
'''
'''
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=list(label[:,0]))
plt.show()
'''