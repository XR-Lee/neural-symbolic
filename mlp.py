import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import os
from tqdm import tqdm


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class PositionDataset(Dataset):
    def __init__(self, annotation_path, img_dir, transform=None):
        self.img_labels = {}
        self.img_dir = img_dir
        self.transform = transform
        with open(annotation_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) >= 3:
                    folder = parts[0]
                    image_name = parts[1] + '.png'
                    image_path = os.path.join(self.img_dir, folder, image_name)
                    if os.path.exists(image_path):
                        self.img_labels[image_path] = int(parts[3])
        self.img_paths = list(self.img_labels.keys())
        print(f"Dataset initialized with {len(self.img_paths)} images.")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels[img_path]
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

full_dataset = PositionDataset('/DATA_EDS2/zhangzz2401/zhangzz2401/OpenLane-V2-master/mapless/lr_annotation_part1.txt', '/DATA_EDS2/wanggl/datasets/BEV_pairwise_polygon_arrow2', transform)
train_size = 300
test_size = 102
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
print("---------- Data Loaded and Split Successfully ----------")

class SimplifiedImageClassifier(nn.Module):
    def __init__(self):
        super(SimplifiedImageClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)  # multi-classification 3
        print("---------- Model Initialized ----------")

    def forward(self, x):
        return self.resnet(x)

model = SimplifiedImageClassifier().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def save_model(epoch, model):
    torch.save(model.state_dict(), f'yyy_model_epoch_{epoch}.pth')
    print(f"---------- Model Saved: Epoch {epoch} ----------")

def train_model(model, train_loader, criterion, optimizer, num_epochs=20):
    model.train()
    print("---------- Start Training ----------")
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
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

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    print("---------- Start Testing ----------")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    print("---------- Testing Completed ----------")

train_model(model, train_loader, criterion, optimizer, num_epochs=20)
test_model(model, test_loader)  # test
