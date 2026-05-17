import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

import psutil
import os
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
EPOCH_SIZE = 5
TRAIN_BATCH_SIZE = 128
TRAINED_DATA_PATH = "./Datas/1.pth"
ANSWER_FILE_PATH = "./Answers/1.csv"

class Train_DigitDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.labels = torch.tensor(df["label"].to_numpy(), dtype=torch.long)
        
        pixels = df.drop(columns=["label"]).to_numpy() / 255.0
        # (num_samples, x, y)
        self.imgs = torch.tensor(pixels.reshape(-1, 28, 28), 
                                 dtype=torch.float32) 
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.labels[idx], self.imgs[idx]
    
class Test_DigitDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        pixels = df.to_numpy() / 255.0
        self.imgs = torch.tensor(pixels.reshape(-1, 28, 28),
                                 dtype=torch.float32) # (num_samples, x, y)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx]
#-------------------------------MODELS-------------------------------

# TODO : Dropout 넣을까?
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, num_classes, 
                               kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_classes)
        self.conv2 = nn.Conv2d(num_classes, num_classes, 
                                kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_classes)
        self.shortcut = nn.Identity()
        self.shortcut_bn = nn.Identity()
        if in_channels != num_classes:
            self.shortcut = nn.Conv2d(in_channels, num_classes, kernel_size=1)
            self.shortcut_bn = nn.BatchNorm2d(num_classes)

    def forward(self, x):
        identity = self.shortcut_bn(self.shortcut(x)) # outchannels * 32 * 32
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + identity)

class DigitRecongnizerModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, pool_size = 2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = num_classes
        self.pool_size = pool_size
        self.pool = nn.MaxPool2d(self.pool_size, self.pool_size) # 32->16
        self.res1 = ResidualBlock(in_channels, num_classes) 
        self.res2 = ResidualBlock(num_classes, num_classes) 
        self.fc1 = nn.Linear(num_classes*(28//self.pool_size**2)**2, 120) # if outchannel = 8 && poolsize = 2, Image size is 32x32, res has 2 pools(2,2).
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.dropout_fc = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.pool(self.res1(x))
        x = self.pool(self.res2(x))
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout_fc(x)
        
        x = self.fc3(x)
        return x
    

if __name__ == "__main__":
    model = DigitRecongnizerModel()
    model.to(DEVICE)

    # train_dataset = Train_DigitDataset("./Datas/train.csv")
    # train_dataloader = DataLoader(train_dataset, batch_size = TRAIN_BATCH_SIZE, shuffle=True)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # train_transform = transforms.RandomApply([
    #     transforms.RandomAffine(
    #         degrees=10,
    #         translate=(0.1, 0.1),
    #         scale=(0.95, 1.05),
    #         shear=5,
    #     ),
    # ], p=0.5)

    # process = psutil.Process(os.getpid())
    # peak_ram = 0.0
    # plt.title("LOSS PER EPOCH")
    # plt_losses = []

    # model.train()
    # for epoch in range(EPOCH_SIZE):
    #     if torch.cuda.is_available():
    #         torch.cuda.reset_peak_memory_stats()
    #     total_loss = 0.0
    #     train_cases = 0

    #     for labels, imgs in train_dataloader:
    #         labels = labels.to(DEVICE) # [B]
    #         imgs = imgs.unsqueeze(1).to(DEVICE) # [B,1,28,28] model에 BCHW 순서 필요 
    #         imgs = train_transform(imgs)
    #         optimizer.zero_grad()

    #         output = model(imgs)
    #         loss = criterion(output, labels)
    #         loss.backward()
    #         optimizer.step()

    #         total_loss += loss.item() * labels.size(0)
    #         train_cases += labels.size(0)
    #         ram = process.memory_info().rss
    #         peak_ram = max(ram, peak_ram)

    #     print(f"EPOCH[{epoch}] LOSS : {total_loss/train_cases:.3f}")
    #     plt_losses.append(total_loss/train_cases)
    #     print(f"  PEAK RAM : [{ram / 1024**2:.3f}]MB")
    #     if torch.cuda.is_available():
    #         peak_vram = torch.cuda.max_memory_allocated() / 1024**2
    #         print(f"  PEAK VRAM : [{peak_vram:.3f}]MB")
   
    # torch.save(model.state_dict(), TRAINED_DATA_PATH)

    # plt.plot(range(1,len(plt_losses)+1), plt_losses)
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.show()

    model.load_state_dict(torch.load(TRAINED_DATA_PATH, map_location=DEVICE))
    model.eval()
    test_dataset = Test_DigitDataset("./Datas/test.csv")
    test_dataloader = DataLoader(test_dataset, batch_size=32,
                                 shuffle=False)
    submission_list = []
    with torch.no_grad():
        for imgs in test_dataloader:
            imgs = imgs.unsqueeze(1).to(DEVICE) # [B,C,H,W]
            pred = model(imgs).argmax(dim=-1) # [B,rows] 
            submission_list.extend(pred.reshape(-1).tolist())

    submission_df = pd.DataFrame({
        "ImageId": list(range(1,len(submission_list)+1)),
        "Label": submission_list
    })

    submission_df.to_csv(ANSWER_FILE_PATH, index=False)