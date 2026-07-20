import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ================= Configuration =================
DATA_DIR = r"F:\CSI数据集\data"

def main():
    print("=" * 60)
    print("      CSI 机器学习分类基线训练 (Random Forest + PCA)")
    print("=" * 60)
    
    # 1. 加载生成的 .npy 数据集
    print("正在加载数据集...")
    try:
        X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
        y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
        X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
        y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
        X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
        y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    except FileNotFoundError as e:
        print(f"[错误] 未找到数据集文件，请先运行 generate_dataset.py 生成数据。详情: {e}")
        return

    print(f"数据加载成功:")
    print(f"  - 训练集: {X_train.shape} | 标签: {y_train.shape}")
    print(f"  - 验证集: {X_val.shape} | 标签: {y_val.shape}")
    print(f"  - 测试集: {X_test.shape} | 标签: {y_test.shape}")
    
    # 2. 数据展平 (Flatten): 将 (N, 250, 192) 转为 (N, 48000)
    print("\n正在展平数据...")
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # 3. PCA 降维: 48000 维特征太高，使用 PCA 降维到 50 维以提高训练效率并防过拟合
    n_components = 50
    print(f"正在进行 PCA 降维 (从 {X_train_flat.shape[1]} 维降至 {n_components} 维)...")
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_flat)
    X_val_pca = pca.transform(X_val_flat)
    X_test_pca = pca.transform(X_test_flat)
    print(f"  - 降维后训练集形状: {X_train_pca.shape}")
    print(f"  - 累计解释方差比例: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # 4. 训练随机森林分类器
    print("\n正在训练随机森林模型 (n_estimators=100)...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train_pca, y_train)
    print("模型训练完成！")
    
    # 5. 模型评估
    # 在验证集上评估
    y_val_pred = clf.predict(X_val_pca)
    val_acc = np.mean(y_val_pred == y_val)
    print(f"\n[验证集评估] 准确率 (Accuracy): {val_acc:.4f}")
    
    # 在测试集上评估
    y_test_pred = clf.predict(X_test_pca)
    test_acc = np.mean(y_test_pred == y_test)
    print(f"[测试集评估] 准确率 (Accuracy): {test_acc:.4f}")
    
    print("\n[测试集分类报告]:")
    target_names = ["Walking/Normal (日常/行走)", "Lying Down (躺下)", "Falling (跌倒)"]
    print(classification_report(y_test, y_test_pred, target_names=target_names))
    
    # 6. 生成混淆矩阵并保存图表
    print("\n正在生成并保存混淆矩阵...")
    cm = confusion_matrix(y_test, y_test_pred)
    
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("CSI 随机森林分类混淆矩阵 (测试集)", fontsize=14, fontweight='bold')
    
    cm_path = os.path.join(DATA_DIR, "sklearn_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"混淆矩阵图已成功保存至: {cm_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
