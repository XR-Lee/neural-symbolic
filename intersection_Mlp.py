import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
import os

# 设置设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class DualPositionDataset(Dataset):
    def __init__(self, annotation_path, img_dir1, img_dir2, transform=None):
        self.img_labels = {}
        self.img_dir1 = img_dir1
        self.img_dir2 = img_dir2
        self.transform = transform
        with open(annotation_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) >= 3:
                    folder = parts[0]
                    image_name_jpg = parts[1] + '.jpg'
                    image_name_png = parts[1] + '.png'
                    image_path1 = os.path.join(self.img_dir1, folder, image_name_jpg)
                    image_path2 = os.path.join(self.img_dir2, folder, image_name_png)
                    if os.path.exists(image_path1) and os.path.exists(image_path2):
                        self.img_labels[(image_path1, image_path2)] = int(parts[2])
        self.img_paths = list(self.img_labels.keys())
        print(f"Dual Dataset initialized with {len(self.img_paths)} image pairs.")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path1, img_path2 = self.img_paths[idx]
        image1 = Image.open(img_path1).convert('RGB')
        image2 = Image.open(img_path2).convert('RGB')
        label = self.img_labels[(img_path1, img_path2)]
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, label

# 初始化数据集
annotation_path = '/DATA_EDS2/zhangzz2401/zhangzz2401/OpenLane-V2-master/mapless/area_annotation_new.txt'
img_dir1 = '/DATA_EDS2/wanggl/datasets/Opentest_mini_batch_area'  # JPG格式
img_dir2 = '/DATA_EDS2/wanggl/datasets/BEV_pairwise_polygon4'     # PNG格式

full_dataset = DualPositionDataset(annotation_path, img_dir1, img_dir2, transform)
print(len(full_dataset))
train_size = 323
test_size = 100
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
print("---------- Data Loaded and Split Successfully ----------")

# 定义双编码器模型
class DualEncoderClassifier(nn.Module):
    def __init__(self):
        super(DualEncoderClassifier, self).__init__()
        self.encoder1 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder2 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.feature_size = self.encoder1.fc.in_features
        self.encoder1.fc = nn.Identity()
        self.encoder2.fc = nn.Identity()

        # 分类器，将两个编码器的特征结合后进行分类
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # 假设是二分类
        )

    def forward(self, x1, x2):
        # 分别通过两个编码器
        features1 = self.encoder1(x1)
        features2 = self.encoder2(x2)

        # 将两个特征拼接在一起
        combined_features = torch.cat((features1, features2), dim=1)

        # 通过分类器
        output = self.classifier(combined_features)
        return output

# 初始化模型
model = DualEncoderClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def save_model(epoch, model):
    torch.save(model.state_dict(), f'zzz_model_epoch_{epoch}.pth')
    print(f"---------- Model Saved: Epoch {epoch} ----------")


# 训练函数
def train_model(model, train_loader, criterion, optimizer, num_epochs=20):
    model.train()
    print("---------- Start Training ----------")
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        for images1, images2, labels in train_loader:
            images1, images2, labels = images1.to(device), images2.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images1, images2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}, Accuracy: {accuracy:.2f}%')
        save_model(epoch + 1, model)
    print("---------- Training Completed ----------")

# 测试函数
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    print("---------- Start Testing ----------")
    with torch.no_grad():
        for images1, images2, labels in test_loader:
            images1, images2, labels = images1.to(device), images2.to(device), labels.to(device)
            outputs = model(images1, images2)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    print("---------- Testing Completed ----------")

# 训练和测试模型
train_model(model, train_loader, criterion, optimizer, num_epochs=20)
test_model(model, test_loader)
