import os
import glob
import json
import numpy as np
import pandas as pd
from scipy import signal

# ================= Configuration =================
SHUJU_DIR = r"F:\CSI数据集\shuju"
OUTPUT_DIR = r"F:\CSI数据集\data"
WINDOW_SIZE = 250   # 滑动窗口大小 (约 3 秒数据)
STEP_SIZE = 10      # 滑动窗口步长 (Stride)
HIGHPASS_CUTOFF = 0.5  # 高通截止频率 (Hz)
ACTIVE_SEARCH_RANGE = 40  # 在最强活动起点前后多少个数据点内进行滑动切片（用于数据增强）

# ================= Helper Functions =================

def get_label_from_dirname(dirname):
    """
    根据文件夹名称映射动作类别：
    - 包含 'fall' -> 2: 跌倒 (Falling)
    - 包含 'lie'  -> 1: 躺下 (Lying Down)
    - 其他        -> 0: 行走/日常 (Walking/Normal)
    """
    name_lower = dirname.lower()
    if 'fall' in name_lower:
        return 2
    elif 'lie' in name_lower:
        return 1
    else:
        return 0

def apply_filtering(data, window_size=3):
    """对每个子载波应用中值滤波以消除突发冲击噪声。"""
    filtered = np.zeros_like(data)
    for i in range(data.shape[1]):
        filtered[:, i] = signal.medfilt(data[:, i], kernel_size=window_size)
    return filtered

def apply_highpass_filter(data, fs, cutoff=HIGHPASS_CUTOFF, order=4):
    """Butterworth 高通滤波器，消除各子载波的直流漂移和静态环境基线。"""
    nyq = 0.5 * fs
    wn = cutoff / nyq
    # 如果采样率太低或数据量太少，直接返回原数据
    if wn >= 1.0 or len(data) < 3 * order:
        return data.astype(np.float32)
    b, a = signal.butter(order, wn, btype='high', analog=False)
    filtered = np.zeros_like(data, dtype=np.float32)
    for i in range(data.shape[1]):
        filtered[:, i] = signal.filtfilt(b, a, data[:, i].astype(float))
    return filtered

def process_single_file(file_path):
    """
    读取单个 CSV 文件，解析时间戳和幅度，执行滤波和预处理。
    """
    df = pd.read_csv(file_path)
    
    # 动态计算实际采样率
    t_us = df['local_timestamp_us'].values
    t_sec = (t_us - t_us[0]) / 1e6
    duration = t_sec[-1]
    fs = len(df) / duration if duration > 0 else 84.0
    
    # 解析 JSON 格式的 amplitude
    amplitudes = []
    for val in df['amplitude']:
        amplitudes.append(json.loads(val))
    amplitudes = np.array(amplitudes, dtype=np.float32) # shape: (packets, 192)
    
    # 1. 中值滤波
    filtered = apply_filtering(amplitudes)
    # 2. 高通滤波
    preprocessed = apply_highpass_filter(filtered, fs=fs)
    
    return preprocessed

def main():
    print("=" * 60)
    print("                CSI 神经网络训练数据集生成程序")
    print("=" * 60)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建输出目录: {OUTPUT_DIR}")
        
    # 1. 扫描所有文件夹和 CSV 文件
    subdirs = [d for d in os.listdir(SHUJU_DIR) if os.path.isdir(os.path.join(SHUJU_DIR, d))]
    
    print(f"在 {SHUJU_DIR} 中扫描到 {len(subdirs)} 个子目录:")
    class_files = {0: [], 1: [], 2: []}
    class_names = {0: "Walking/Normal (行走/日常)", 1: "Lying Down (躺下)", 2: "Falling (跌倒)"}
    
    for subdir in subdirs:
        label = get_label_from_dirname(subdir)
        path = os.path.join(SHUJU_DIR, subdir)
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        class_files[label].extend(csv_files)
        print(f"  - 目录: {subdir:<20} | 类别: {class_names[label]:<25} | 文件数: {len(csv_files)}")
        
    print("\n全局类别文件统计:")
    for label, files in class_files.items():
        print(f"  - 类别 {label} ({class_names[label]}): 共 {len(files)} 个文件")
        
    # 2. 对每个类别的文件进行分层划分 (Stratified Split)，防止数据泄露
    # 训练集:验证集:测试集 = 70% : 15% : 15%
    train_files = []
    val_files = []
    test_files = []
    
    np.random.seed(42) # 固定随机种子保证可重复性
    
    for label, files in class_files.items():
        if len(files) == 0:
            continue
        # 随机打乱文件列表
        shuffled = np.random.permutation(files).tolist()
        n = len(shuffled)
        
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        # 余下的给测试集
        
        train_files.extend([(f, label) for f in shuffled[:n_train]])
        val_files.extend([(f, label) for f in shuffled[n_train:n_train+n_val]])
        test_files.extend([(f, label) for f in shuffled[n_train+n_val:]])
        
    print(f"\n文件级划分结果:")
    print(f"  - 训练集文件数: {len(train_files)}")
    print(f"  - 验证集文件数: {len(val_files)}")
    print(f"  - 测试集文件数: {len(test_files)}")
    
    # 3. 逐个文件处理并切分滑动窗口
    def extract_windows_from_set(file_label_list, set_name):
        X_set = []
        y_set = []
        total_files = len(file_label_list)
        
        print(f"\n正在处理 {set_name}...")
        for idx, (f_path, label) in enumerate(file_label_list):
            try:
                # 提取滤波后的 CSI 幅度矩阵 (num_packets, 192)
                data = process_single_file(f_path)
                
                # 如果文件太短，跳过
                if len(data) < WINDOW_SIZE:
                    print(f"  [警告] 跳过过短文件: {os.path.basename(f_path)} (长度 {len(data)})")
                    continue
                    
                # --- 活动区域定位与有限切片 ---
                # 1. 计算一阶差分的均值，作为每个时间点动作强度的指标
                diff = np.abs(np.diff(data, axis=0)) # shape: (N-1, 192)
                score = np.mean(diff, axis=1)        # shape: (N-1,)
                
                # 2. 卷积计算滑动窗口内的累积动作能量
                rolling_activity = np.convolve(score, np.ones(WINDOW_SIZE), mode='valid')
                
                # 3. 找到能量最大（动作最剧烈）的窗口起点
                best_start = np.argmax(rolling_activity)
                
                # 4. 在最佳起点前后限制滑动范围，既做了数据增强，又避免切到平静期
                search_start = max(0, best_start - ACTIVE_SEARCH_RANGE)
                search_end = min(len(data) - WINDOW_SIZE, best_start + ACTIVE_SEARCH_RANGE)
                
                count = 0
                for start in range(search_start, search_end + 1, STEP_SIZE):
                    end = start + WINDOW_SIZE
                    window = data[start:end] # shape: (250, 192)
                    X_set.append(window)
                    y_set.append(label)
                    count += 1
                
                if (idx + 1) % 10 == 0 or (idx + 1) == total_files:
                    print(f"  已处理: {idx+1}/{total_files} 文件")
                    
            except Exception as e:
                print(f"  [错误] 处理文件 {os.path.basename(f_path)} 失败: {e}")
                
        return np.array(X_set, dtype=np.float32), np.array(y_set, dtype=np.int64)

    X_train, y_train = extract_windows_from_set(train_files, "训练集 (Train Set)")
    X_val, y_val = extract_windows_from_set(val_files, "验证集 (Validation Set)")
    X_test, y_test = extract_windows_from_set(test_files, "测试集 (Test Set)")
    
    # 4. 打印生成的数据集形状与类别分布
    print("\n" + "=" * 60)
    print("                    数据集处理完成统计报告")
    print("=" * 60)
    print(f"训练集 (Train) 形状: X = {str(X_train.shape):<18} | y = {y_train.shape}")
    print(f"验证集 (Val)   形状: X = {str(X_val.shape):<18} | y = {y_val.shape}")
    print(f"测试集 (Test)  形状: X = {str(X_test.shape):<18} | y = {y_test.shape}")
    
    def print_class_dist(y, name):
        unique, counts = np.unique(y, return_counts=True)
        dist = dict(zip(unique, counts))
        print(f"  - {name} 类别分布: " + ", ".join([f"类 {k} ({class_names[k]}): {dist.get(k, 0)}样本" for k in [0, 1, 2]]))
        
    print_class_dist(y_train, "训练集")
    print_class_dist(y_val, "验证集")
    print_class_dist(y_test, "测试集")
    
    # 5. 保存为标准 .npy 文件
    print("\n正在保存数据集到 F:\\CSI数据集\\data 目录...")
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)
    
    print("\n数据集生成成功！所有数据均以 float32 保存，完美适用于神经网络训练。")
    print("保存的文件包括:")
    print(f"  - {os.path.join(OUTPUT_DIR, 'X_train.npy')}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'y_train.npy')}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'X_val.npy')}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'y_val.npy')}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'X_test.npy')}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'y_test.npy')}")
    print("=" * 60)

if __name__ == "__main__":
    main()
