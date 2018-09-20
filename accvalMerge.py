import numpy as np
import scipy.io as sio
#import input_data
import pickle as pk
import random
import matplotlib.pyplot as plt

with open(r'Huaximacc.pkl','rb') as f:
    data_dic = pk.load(f)
train_acc_fold = data_dic['train']
test_acc_fold = data_dic['test']
val_acc_fold = data_dic['val']
x_iter_list = []
for a in range(0,50*len(train_acc_fold[0]),50):
    x_iter_list.append(a)
tr_acc = np.array(train_acc_fold[0])
vl_acc = np.array(val_acc_fold[0])
te_acc = np.array(test_acc_fold[0])
acc = np.array(test_acc_fold[0][test_acc_fold[0].index(max(test_acc_fold[0]))])
acc2 = np.array(val_acc_fold[0][test_acc_fold[0].index(max(test_acc_fold[0]))])
for k in range(1,10):
    tr_acc = tr_acc + train_acc_fold[k]
    te_acc = te_acc + test_acc_fold[k]
    vl_acc = vl_acc + val_acc_fold[k]
    acc = acc + max(test_acc_fold[k][val_acc_fold[(k)%10].index(max(val_acc_fold[(k)%10]))],np.array(val_acc_fold[(k-1)%10][test_acc_fold[(k-1)%10].index(max(test_acc_fold[(k-1)%10]))]))
    print([val_acc_fold[(k)%10].index(max(val_acc_fold[(k)%10]))],[test_acc_fold[(k-1)%10].index(max(test_acc_fold[(k-1)%10]))])
    #print(len(train_acc_fold[k]))
    acc2 =acc2 + np.array(val_acc_fold[k][test_acc_fold[(k)%10].index(max(test_acc_fold[(k)%10]))])


tr_acc = tr_acc/10.0
te_acc = te_acc/10.0
vl_acc = vl_acc/10.0
acc_fold_1 = []
acc_fold_2 = []
acc_fold_3 = []
length = len(train_acc_fold[0])
for j in range(0,9):
    acc_1 = 0.0
    acc_2 = 0.0
    acc_3 = 0.0
    for i in range(5000//50,length):
        acc_1 = acc_1 + train_acc_fold[j][i]
        acc_2 = acc_2 + test_acc_fold[j][i]
        acc_3 = acc_3 + val_acc_fold[j-1][i]
    acc_fold_1.append(acc_1/length*2)
    acc_fold_2.append(acc_2/length*2)
    acc_fold_3.append(acc_3/length*2)

#acc =acc - test_acc_fold[k][test_acc_fold[k].index(max(test_acc_fold[k]))]
print (acc, acc2)
train_acc_list = tr_acc.tolist()
test_acc_list = te_acc.tolist()
vl_acc_list = vl_acc.tolist()
l=0
y1 = test_acc_fold[l+1]
y2 = val_acc_fold[l]
x1 = x_iter_list
#x1 = [0,1,2,3,4,5,6,7,8]
plt.plot(x1,y1,label = 'train acc',color = 'r')
plt.plot(x1,y2,label = 'val acc')
plt.xlabel('iteration num')
plt.ylabel('acc')
plt.show()
