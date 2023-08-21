import torchvision.transforms as transforms
import torch.utils.data as data
import albumentations as A
import cv2
import numpy as np
import torch

train_transform = A.Compose([
    A.Resize(width=240, height=240),
    A.RandomSizedCrop([200, 220], height=224, width=224, p=0.4),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=(-10,10), p=0.6),
    A.RandomBrightnessContrast(contrast_limit=0.05, brightness_limit=0.05, p=0.75),
    # A.RandomBrightness(limit=0.05, p=0.75),
    A.CenterCrop(width=224, height=224),
])

test_transform = A.Compose([
    A.Resize(width=240, height=240),  # Thay đổi kích thước ảnh
    A.CenterCrop(width=224, height=224),
])

class MyDataset(data.Dataset):
    def __init__(self, data_frame, trans = None):
        super().__init__()
        self.data_frame = data_frame
        self.trans = trans

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self,idx):
        if self.trans:
            pass
        path, label = self.data_frame.iloc[idx][0], self.data_frame.iloc[idx][1]
        x = cv2.imread(path)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = x.astype(np.float32)/255
        y = [0] * 10
        y[label] = 1
        y = np.array(y)
        if self.trans:
            sample = {
                "image": x
            }
            sample = self.trans(**sample)
            x = sample["image"]

        x = np.transpose(x, (2, 0, 1))
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y