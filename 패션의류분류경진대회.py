# python 3.7.13, cuda 11.3, torch 1.12.1 
# https://dacon.io/competitions/open/235594/overview/description
# Data Download -> https://dacon.io/competitions/open/235594/data
# 참고 코드 -> https://www.kaggle.com/code/nohrud/fashion-mnist-with-pytorch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('./data/train.csv', index_col = 'index')
test_df = pd.read_csv('./data/test.csv', index_col = 'index') # 10000개

x_train = train_df.iloc[:,1:].values/255  # /255를 함으로써 스케일링? -> 각 픽셀이 255값이 최대값이여서 0~1사이의 값으로 조정(일종의 Min Max Scaler)
y_train = train_df.iloc[:,0].values
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=0.2) #, random_state=2022

x_train.shape
x_val.shape
y_train.shape
y_val.shape

#Change them into Pytorch's Float Tensors.
train_x_torch = torch.from_numpy(x_train).type(torch.FloatTensor) # 파이토치에서 연산하기 위해 Tensor형태로 변환
valid_x_torch = torch.from_numpy(x_val).type(torch.FloatTensor)
train_y_torch = torch.from_numpy(y_train).type(torch.LongTensor)
valid_y_torch = torch.from_numpy(y_val).type(torch.LongTensor)


train_x_torch = train_x_torch.view(-1, 1,28,28).float() # view와 reshape은 같다, 48000,1,28,28 형태
valid_x_torch = valid_x_torch.view(-1, 1,28,28).float() # 12000,1,28,28 형태

train_set = torch.utils.data.TensorDataset(train_x_torch, train_y_torch)
valid_set = torch.utils.data.TensorDataset(valid_x_torch, valid_y_torch)

train_loader = torch.utils.data.DataLoader(train_set, shuffle = True, batch_size = 64)
valid_loader = torch.utils.data.DataLoader(valid_set, shuffle = True, batch_size = 64)

plt.figure(figsize=(30,40))
for i in range(100):
    plt.subplot(15, 15, i+1)
    plt.title("No." + str(i))
    plt.imshow(train_df.iloc[:,1:].iloc[i].values.reshape(28,28),cmap='Greys')
    
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else 'cpu')
DEVICE

list_process=[]
loss_process = []
def train(model, epoch):
    model.train()
    train_loss = 0
    correct = 0  # 정답 수
    for data, label in train_loader:
        data, label = data.to(DEVICE), label.to(DEVICE)  
        optimizer.zero_grad() #Pytorch에서는 gradients값들을 추후에 backward를 해줄때 계속 더해주기 때문에 0으로 초기화
        output = model(data) 
        loss = criterion(output, label)   #criterion = nn.CrossEntropyLoss(), lossfunction에 따라 계산해줌
        loss.backward() 
        optimizer.step()  
        train_loss += loss.item()  # 손실이 갖고 있는 스칼라 값을 더해줌
        #get argmax values in outputs
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(label.view_as(pred)).sum().item()
    print('epoch for train: {}, accuracy: ({:.2f}%), train_loss: ({:.4f})'.format(epoch,correct*100 / len(train_loader.dataset),train_loss/len(train_loader.dataset)))
    list_process.append(correct*100 / len(train_loader.dataset))
    loss_process.append(train_loss/len(train_loader.dataset))

val_list_process = []
val_loss_process = []
def valid(model, epoch):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, label in valid_loader:
            data, label = data.to(DEVICE), label.to(DEVICE)
            output = model(data)
            val_loss += criterion(output, label).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()
        print('epoch for test: {}, accuracy: ({:.2f}%) val_loss: ({:.4f})'.format(epoch,correct*100 / len(valid_loader.dataset),val_loss/len(valid_loader.dataset)))
    val_list_process.append(correct*100 / len(valid_loader.dataset))
    val_loss_process.append(val_loss/len(valid_loader.dataset))
class Net(nn.Module):
    def __init__(self, dropout_p=0.2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout_p = dropout_p
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        
        # 드롭아웃 추가
        x = F.dropout(x, training=self.training, p=self.dropout_p)
        x = F.relu(self.fc2(x))
        
        # 드롭아웃 추가
        x = F.dropout(x, training=self.training, p=self.dropout_p)
        x = self.fc3(x)
        return x
    
model = Net(dropout_p=0.2).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss() 

print(model)
for epoch in range(100):
    train(model,epoch)
    valid(model,epoch)
    
#for epoch in range(20):
#    valid(model,epoch)
    
import matplotlib.pyplot as plt
plt.plot(list_process, label='train_acc')
plt.plot(val_list_process, label='val_acc')
plt.xlabel("number of epochs")
plt.ylabel("accuracy(%)")
plt.title("DNN with Pytorch Model_acc")
plt.legend()
plt.show()

import matplotlib.pyplot as plt
plt.plot(loss_process, label='train_loss')
plt.plot(val_loss_process, label='val_loss')
plt.xlabel("number of epochs")
plt.ylabel("loss")
plt.title("DNN with Pytorch Model_loss")
plt.legend()
plt.show()

x_test = test_df.values/255
x_test_torch = torch.from_numpy(x_test).type(torch.FloatTensor)
d_labels = np.zeros(x_test.shape) # test데이터에는 label값이 없으므로 0으로 일단 채우기
d_labels = torch.from_numpy(d_labels) # tensor형태로 변환
#Think about dimentions of data. Without this "an shapes doesn't fit error", will occur.
x_test_torch = x_test_torch.view(-1, 1, 28, 28)
#Make a tensordataset and a testloader
testset = torch.utils.data.TensorDataset(x_test_torch, d_labels)
testloader = torch.utils.data.DataLoader(testset, batch_size = 1, shuffle = False)


y_pred = []
with torch.no_grad():
    model.eval()
    for images,label in testloader:
        images,label = images.to(DEVICE), label.to(DEVICE)
        outputs = model(images)
        probs = torch.exp(outputs) # outputs의 지수형태로 tensor화
        top_p, top_class = probs.topk(1, dim = 1) # 각각 10000개, top_p는 가장 큰 값, top_class는 가장 큰 값의 indice 즉 label값?
        for preds in top_class:
            y_pred.append(preds.item()) #item함수는 텐서를 요소화(숫자 형태로 꺼내옴)
            
# 제출
submission = pd.read_csv('./data/sample_submission.csv', encoding = 'utf-8') 
submission['label'] = y_pred
submission.to_csv('fashion_submission.csv', index = False)


df = pd.read_csv('fashion_submission.csv')

plt.figure(figsize=(30,30))
for i in range(10):
    plt.subplot(20, 20, i+1)
    plt.imshow(test_df.iloc[i].values.reshape(28,28),cmap='Greys')

t_label=["T-shirt or top","Pants","Pullover","Dress","Coat","Sandal","Shirt","Shoes","Bag","Ankle boot"]
check_answer=df.iloc[:10,[1]].T
tolist_check=np.array(check_answer).tolist()
import itertools
tlc=list(itertools.chain.from_iterable(tolist_check))
count=0
for k in tlc:
    count=count+1
    print("No",count,":",t_label[int(k)])