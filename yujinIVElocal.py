import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import toml

# 設置訓練參數
batch_size = 8
num_epochs = 10
learning_rate = 0.001

# 定義 LoRA 模型
class LoRAModel(torch.nn.Module):
    def __init__(self):
        super(LoRAModel, self).__init__()
        # 在這裡定義你的模型結構

    def forward(self, x):
        # 定義前向傳播邏輯
        pass

# 處理自定義數據集配置
custom_dataset = """
[[datasets]]

[[datasets.subsets]]
image_dir = "//Users/tszsanwu/Desktop/Loras/Image/100_yujin_ikuyo"
num_repeats = 10

[[datasets.subsets]]
image_dir = "/Users/tszsanwu/Desktop/Loras/reg/1_woman/woman_v1-5_mse_vae_ddim50_cfg7_n4420"
is_reg = true
num_repeats = 1
"""

# 解析自定義數據集配置
dataset_config = toml.loads(custom_dataset)
datasets = dataset_config.get("datasets", [])
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # 根據需要調整大小
    transforms.ToTensor(),  # 轉換為張量
])

# 創建數據集和數據讀取器
train_datasets = []
for dataset in datasets:
    subsets = dataset.get("subsets", [])
    for subset in subsets:
        image_dir = subset.get("image_dir")
        num_repeats = subset.get("num_repeats", 1)
        is_reg = subset.get("is_reg", False)

        dataset = torchvision.datasets.ImageFolder(root=image_dir, transform=transform)
        train_datasets.extend([dataset] * num_repeats)

# 合併數據集
train_dataset = torch.utils.data.ConcatDataset(train_datasets)

# 創建數據讀取器
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 初始化模型、損失函數和優化器
model = LoRAModel()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 訓練模型
total_step = len(dataloader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(dataloader):
        # 前向傳播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向傳播和優化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印訓練信息
        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item()}")

# 保存模型
save_path = "/Users/tszsanwu/Desktop/Loras/Model/Yujin_V2/model.pth"
torch.save(model.state_dict(), save_path)
