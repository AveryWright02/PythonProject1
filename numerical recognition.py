import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),          # 转换为张量 (范围 [0,1])
    transforms.Normalize((0.5,), (0.5,))  # 归一化到 [-1, 1]
])

# 下载并加载数据集
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# 创建数据加载器
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),  # 卷积层
            nn.ReLU(),
            nn.MaxPool2d(2),  # 池化层
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 10)
        )

    def forward(self, x):
        return self.net(x)


# 初始化模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNNClassifier().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()          # 交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

epochs = 5

for epoch in range(epochs):
    model.train()
    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)

        # 前向传播
        pred = model(X)
        loss = loss_fn(pred, y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 每100个批次打印进度
        if batch % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Batch {batch}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "mnist_digit_classifier.pth")

# 加载测试集中的一张图片
sample_image, sample_label = test_dataset[0]
sample_image = sample_image.unsqueeze(0).to(device)  # 增加批次维度

# 预测
model.eval()
with torch.no_grad():
    pred = model(sample_image)
    predicted_label = pred.argmax(1).item()

# 显示结果
plt.imshow(sample_image.cpu().squeeze(), cmap='gray')
plt.title(f"True: {sample_label}, Predicted: {predicted_label}")
plt.show()