from torch.autograd import Variable
from model.Face_Fake_Net import Face_Fake_Net
from dataset.MyDataset import MyDataset
from torchmetrics.classification import BinaryAccuracy
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import albumentations as A
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df = pd.read_csv('/media/quocviet/SSD Data/Viet_FAS/CelebA_Spoof/image_preprocessed/annotation/train_annotation.csv')
df = df.iloc[:10000]
num_samples = df.shape[0]
num_samples_train_set = int(num_samples * 0.8)

random_indices = np.random.permutation(num_samples)

train_set_indices = random_indices[:num_samples_train_set]
train_set = df.iloc[train_set_indices]

val_set_indices = random_indices[num_samples_train_set:]
val_set = df.iloc[val_set_indices]


train_transform = A.Compose([
    A.Resize(width=240, height=240),
    A.RandomSizedCrop([200, 220], height=224, width=224, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=(-10,10), p=0.5),
    A.RandomBrightnessContrast(contrast_limit=0.05, brightness_limit=0.05, p = 0.6),
    # A.RandomBrightness(limit=0.05, p=0.75),
    A.CenterCrop(width=224, height=224),
])

test_transform = A.Compose([
    A.Resize(width=240, height=240),  # Thay đổi kích thước ảnh
    A.CenterCrop(width=224, height=224),
])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Face_Fake_Net()
model.to(device)
# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.00005)
criterion = nn.BCELoss()


# empty list to store training losses
train_losses = []
# empty list to store validation losses
val_losses = []
best_val_loss = float(99)
print(device)
train_dataset = MyDataset(train_set, trans = train_transform)
val_dataset = MyDataset(val_set, trans = test_transform)

train_dataloader = data.DataLoader(train_dataset, batch_size = 64, shuffle=True)
valid_dataloader = data.DataLoader(val_dataset, batch_size = 64, shuffle=False)

def train(epoch):
    global train_losses, val_losses, best_val_loss
    model.train()
    loss = 0
    val_loss = 0
    for x_batch,y_batch in train_dataloader:
        x_train, y_train = Variable(x_batch), Variable(y_batch)
        # getting the validation set
        #xóa gradients
        optimizer.zero_grad()
        # Cho dữ liệu qua model và trả về output cần tìm
        pred = model(x_train.to(device))
        # Tính toán giá trị lỗi và backpropagation
        loss = criterion(pred, y_train.to(device))
        loss.backward()
        # Cập nhật trọng số
        optimizer.step()
    train_losses.append(loss)
    #Thiết lập trạng thái đánh giá cho mô hình, ở bước này thì mô hình không backward và cập nhật trọng số
    model.eval()
    for x_batch,y_batch in valid_dataloader:
        x_val, y_val = Variable(x_batch), Variable(y_batch)
        pred = model(x_val.to(device))
        val_loss = criterion(pred, y_val.to(device))
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # torch.save(model.state_dict(), '/content/drive/MyDrive/Colab Notebooks/Hoc_Sau/Data/Best_model/ResNet_101_miml_best_check_point')
        print('Best val_loss: ', val_loss)
    val_losses.append(val_loss)
    print('Epoch : ',epoch+1, '\t', 'loss :', loss, '\t', 'Valloss :', val_loss)
    
for epoch in range(10):
    train(epoch)