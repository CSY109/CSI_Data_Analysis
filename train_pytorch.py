import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ================= Configuration =================
DATA_DIR = r"F:\CSI数据集\data3"
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 1D CNN Architecture =================
class CSI1DCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(CSI1DCNN, self).__init__()
        # 输入形状: (Batch, Channels=6, Time=25)
        self.features = nn.Sequential(
            # 第一卷积层：提取粗粒度的时空变化特征
            nn.Conv1d(in_channels=6, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            
            # 第二卷积层：深层抽象特征提取
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            
            # 第三卷积层：自适应时序汇聚
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # 最终池化到 1 维时序，提取全局高层特征
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5), # Dropout 防止过拟合
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # 传入的 x 形状是 (Batch, Time=25, Subcarriers=6)
        # Conv1d 期待 shape 是 (Batch, Channels=6, Time=25)
        x = x.transpose(1, 2)
        x = self.features(x)
        x = torch.flatten(x, 1) # 展平为 (Batch, 256)
        x = self.classifier(x)
        return x

def main():
    print("=" * 60)
    print("         CSI PyTorch 1D CNN 深度学习模型训练")
    print("=" * 60)
    print(f"当前运行设备: {DEVICE}")
    
    # 1. 加载数据集
    print("正在加载数据集...")
    try:
        X_train = np.load(os.path.join(DATA_DIR, "X_custom.npy"))
        y_train = np.load(os.path.join(DATA_DIR, "y_custom.npy"))
        X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
        y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
        X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
        y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    except FileNotFoundError as e:
        print(f"[错误] 未找到数据集文件，请先运行 generate_dataset.py。详情: {e}")
        return

    print("数据加载成功，转为 PyTorch Tensors...")
    
    # 2. 构建 DataLoader
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. 初始化模型、损失函数与优化器
    model = CSI1DCNN(num_classes=3).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # 4. 训练循环
    best_val_acc = 0.0
    model_save_path = os.path.join(DATA_DIR, "best_csi_model.pth")
    
    train_losses = []
    val_losses = []
    
    print("\n开始模型训练...")
    for epoch in range(1, EPOCHS + 1):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()
            
        epoch_loss = running_loss / total_train
        epoch_acc = correct_train / total_train
        train_losses.append(epoch_loss)
        
        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total_val += targets.size(0)
                correct_val += predicted.eq(targets).sum().item()
                
        val_loss = val_running_loss / total_val
        val_acc = correct_val / total_val
        val_losses.append(val_loss)
        
        print(f"Epoch [{epoch:02d}/{EPOCHS}] | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"  --> 保存了最佳验证模型 checkpoint (Acc: {val_acc*100:.2f}%)")
            
    print("\n训练完成！")
    print(f"最佳验证集准确率: {best_val_acc*100:.2f}%")
    
    # 5. 加载最佳模型并评估测试集
    print(f"正在加载最佳模型评估测试集: {model_save_path}")
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    
    correct_test = 0
    total_test = 0
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total_test += targets.size(0)
            correct_test += predicted.eq(targets).sum().item()
            test_preds.extend(predicted.cpu().numpy())
            test_targets.extend(targets.cpu().numpy())
            
    test_acc = correct_test / total_test
    print(f"[测试集最终评估] 准确率: {test_acc*100:.2f}%")
    
    # 6. 绘制并保存 Loss 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", color="blue", linewidth=2)
    plt.plot(val_losses, label="Val Loss", color="red", linewidth=2)
    plt.title("CSI 1D CNN 训练与验证 Loss 曲线", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    loss_curve_path = os.path.join(DATA_DIR, "pytorch_loss_curve.png")
    plt.savefig(loss_curve_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存至: {loss_curve_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
