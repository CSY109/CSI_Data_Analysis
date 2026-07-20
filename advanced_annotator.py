import os
import glob
import json
import numpy as np
import pandas as pd
from scipy import signal
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn.decomposition import PCA

# 防止中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# ================= 数据预处理与特征提取 =================
def apply_filtering(data, window_size=3):
    filtered = np.zeros_like(data)
    for i in range(data.shape[1]):
        filtered[:, i] = signal.medfilt(data[:, i], kernel_size=window_size)
    return filtered

def apply_highpass_filter(data, fs=80.0, cutoff=0.5, order=4):
    nyq = 0.5 * fs
    wn = cutoff / nyq
    if wn >= 1.0 or len(data) < 3 * order:
        return data.astype(np.float32)
    b, a = signal.butter(order, wn, btype='high', analog=False)
    filtered = np.zeros_like(data, dtype=np.float32)
    for i in range(data.shape[1]):
        filtered[:, i] = signal.filtfilt(b, a, data[:, i].astype(float))
    return filtered

def compute_stft_spectrogram(signal_1d, fs=80.0, nperseg=100, noverlap=50, n_freq_bins=25):
    n = len(signal_1d)
    nperseg = min(nperseg, n)
    if nperseg < 4: return None
    noverlap = min(noverlap, nperseg - 1)
    _, _, Zxx = signal.stft(signal_1d.astype(float), fs=fs, nperseg=nperseg, noverlap=noverlap)
    return np.abs(Zxx[:n_freq_bins, :])

def extract_stft_features(chunk_2d):
    if len(chunk_2d) < 10: return None
    # PCA 降维提取第一主成分，消除子载波间的冗余
    pc1 = PCA(n_components=1).fit_transform(chunk_2d).flatten()
    return compute_stft_spectrogram(pc1)
# =========================================================

class AdvancedAnnotatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CSI 专家级标注与滑动窗口调节工具 (批量版)")
        self.root.geometry("1400x900")
        
        # State variables
        self.csv_files = []
        self.current_file_idx = -1
        self.current_csv = None
        self.raw_data = None
        self.fs = 80.0
        
        # Dataset accumulation
        self.X_collected = []
        self.y_collected = []
        
        # UI Variables
        self.window_size_var = tk.IntVar(value=250)
        self.step_size_var = tk.IntVar(value=10)
        self.search_range_var = tk.IntVar(value=40)
        self.label_var = tk.IntVar(value=0)
        self.manual_start_var = tk.IntVar(value=-1) # -1 means auto
        self.use_pca_stft_var = tk.BooleanVar(value=True)
        
        self.setup_ui()
        
    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # 放大字体和按钮
        LARGE_FONT = ("Microsoft YaHei", 12)
        BOLD_FONT = ("Microsoft YaHei", 12, "bold")
        
        style.configure(".", font=LARGE_FONT)
        style.configure("TLabel", font=LARGE_FONT, padding=2)
        style.configure("TButton", font=BOLD_FONT, padding=8)
        style.configure("TRadiobutton", font=LARGE_FONT)
        style.configure("TLabelframe.Label", font=BOLD_FONT, foreground="#0052cc")
        
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)
        
        # --- Left Panel (Controls) ---
        left_panel = ttk.Frame(main_pane, width=420, padding=10)
        left_panel.pack_propagate(False)
        main_pane.add(left_panel, weight=0)
        
        # 1. File loading
        file_frame = ttk.LabelFrame(left_panel, text=" 1. 批量加载 CSV ", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        btn_frame = ttk.Frame(file_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        btn_open_files = ttk.Button(btn_frame, text="选择多个文件", command=self.load_multiple_files)
        btn_open_files.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        btn_open_folder = ttk.Button(btn_frame, text="选择文件夹", command=self.load_folder)
        btn_open_folder.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))
        
        self.lbl_file_count = ttk.Label(file_frame, text="待处理文件池: 0 个", font=("Microsoft YaHei", 11, "italic"))
        self.lbl_file_count.pack(anchor=tk.W, pady=2)
        
        self.lbl_filename = ttk.Label(file_frame, text="当前文件: 无", wraplength=380, foreground="red")
        self.lbl_filename.pack(anchor=tk.W, pady=5)
        
        nav_frame = ttk.Frame(file_frame)
        nav_frame.pack(fill=tk.X, pady=5)
        ttk.Button(nav_frame, text="<< 上一个", command=self.prev_file).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(nav_frame, text="下一个 >>", command=self.next_file).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))
        
        # 2. Parameters
        param_frame = ttk.LabelFrame(left_panel, text=" 2. 调节切片参数 ", padding=10)
        param_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(param_frame, text="滑动窗口大小 (Window Size):").pack(anchor=tk.W)
        ttk.Scale(param_frame, from_=50, to=1000, variable=self.window_size_var, command=self.update_plot).pack(fill=tk.X)
        self.lbl_ws = ttk.Label(param_frame, text="250", foreground="blue")
        self.lbl_ws.pack(anchor=tk.E)
        
        ttk.Label(param_frame, text="滑动步长 (Step Size):").pack(anchor=tk.W)
        ttk.Scale(param_frame, from_=1, to=200, variable=self.step_size_var, command=self.update_plot).pack(fill=tk.X)
        self.lbl_ss = ttk.Label(param_frame, text="10", foreground="blue")
        self.lbl_ss.pack(anchor=tk.E)
        
        ttk.Label(param_frame, text="搜索宽度/范围 (Search Range):").pack(anchor=tk.W)
        ttk.Scale(param_frame, from_=0, to=500, variable=self.search_range_var, command=self.update_plot).pack(fill=tk.X)
        self.lbl_sr = ttk.Label(param_frame, text="40", foreground="blue")
        self.lbl_sr.pack(anchor=tk.E)
        
        ttk.Label(param_frame, text="手动最强点起点 (设为-1为自动):").pack(anchor=tk.W, pady=(10,0))
        ttk.Entry(param_frame, textvariable=self.manual_start_var, font=LARGE_FONT).pack(fill=tk.X, pady=2)
        ttk.Button(param_frame, text="应用手动起点 (或右键点击图表)", command=self.update_plot).pack(fill=tk.X, pady=5)
        
        # 3. Annotation
        anno_frame = ttk.LabelFrame(left_panel, text=" 3. 提取与标注 ", padding=10)
        anno_frame.pack(fill=tk.X, pady=10)
        
        ttk.Radiobutton(anno_frame, text="0: Normal (静止)", variable=self.label_var, value=0).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(anno_frame, text="1: Lying Down (躺下)", variable=self.label_var, value=1).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(anno_frame, text="2: Falling (跌倒)", variable=self.label_var, value=2).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(anno_frame, text="3: Walking (行走)", variable=self.label_var, value=3).pack(anchor=tk.W, pady=2)
        ttk.Separator(anno_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        ttk.Checkbutton(anno_frame, text="使用 PCA 降维 + STFT 特征提取\n(取消勾选则保存全部192子载波)", variable=self.use_pca_stft_var).pack(anchor=tk.W, pady=2)
        
        # Style specially for the main action button
        style.configure("Accent.TButton", font=("Microsoft YaHei", 12, "bold"), foreground="green")
        btn_add = ttk.Button(anno_frame, text="✅ 将当前框选片段加入数据集", style="Accent.TButton", command=self.add_to_dataset)
        btn_add.pack(fill=tk.X, pady=10)
        
        # 4. Export
        export_frame = ttk.LabelFrame(left_panel, text=" 4. 导出数据集 ", padding=10)
        export_frame.pack(fill=tk.X, pady=10)
        
        self.lbl_stats = ttk.Label(export_frame, text="📦 已收集样本数: 0", foreground="purple", font=BOLD_FONT)
        self.lbl_stats.pack(anchor=tk.W, pady=5)
        
        btn_export = ttk.Button(export_frame, text="💾 导出为 .npy 文件", command=self.export_dataset)
        btn_export.pack(fill=tk.X, pady=5)
        
        # --- Right Panel (Plot) ---
        right_panel = ttk.Frame(main_pane, padding=10)
        main_pane.add(right_panel, weight=1)
        
        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(self.canvas, right_panel)
        toolbar.update()
        
        self.canvas.mpl_connect("button_press_event", self.on_click)
        
    def load_multiple_files(self):
        filepaths = filedialog.askopenfilenames(
            title="选择多个 CSV 文件",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not filepaths: return
        self.csv_files = list(filepaths)
        self.lbl_file_count.config(text=f"待处理文件池: {len(self.csv_files)} 个")
        self.current_file_idx = 0
        self.load_current_index()

    def load_folder(self):
        folderpath = filedialog.askdirectory(title="选择包含 CSV 的文件夹")
        if not folderpath: return
        
        csvs = glob.glob(os.path.join(folderpath, "*.csv"))
        if not csvs:
            messagebox.showwarning("提示", "该文件夹下没有找到 CSV 文件。")
            return
            
        self.csv_files = csvs
        self.lbl_file_count.config(text=f"待处理文件池: {len(self.csv_files)} 个")
        self.current_file_idx = 0
        self.load_current_index()
        
    def prev_file(self):
        if not self.csv_files: return
        if self.current_file_idx > 0:
            self.current_file_idx -= 1
            self.load_current_index()
        else:
            messagebox.showinfo("提示", "已经是第一个文件了。")
            
    def next_file(self):
        if not self.csv_files: return
        if self.current_file_idx < len(self.csv_files) - 1:
            self.current_file_idx += 1
            self.load_current_index()
        else:
            messagebox.showinfo("提示", "已经是最后一个文件了。")
            
    def load_current_index(self):
        if self.current_file_idx < 0 or self.current_file_idx >= len(self.csv_files):
            return
            
        filepath = self.csv_files[self.current_file_idx]
        try:
            self.current_csv = filepath
            filename = os.path.basename(filepath)
            self.lbl_filename.config(text=f"({self.current_file_idx+1}/{len(self.csv_files)}) {filename}", foreground="#d32f2f")
            
            df = pd.read_csv(filepath)
            amplitudes = [json.loads(val) for val in df['amplitude']]
            self.raw_data = np.array(amplitudes, dtype=np.float32)
            
            self.mean_amp = np.mean(self.raw_data, axis=1)
            
            # Automatically try to guess the label from filename or directory name
            name_lower = filepath.lower()
            if 'fall' in name_lower: self.label_var.set(2)
            elif 'lie' in name_lower: self.label_var.set(1)
            else: self.label_var.set(0)
            
            self.manual_start_var.set(-1) 
            self.update_plot()
            
        except Exception as e:
            messagebox.showerror("加载失败", f"文件: {filepath}\n{str(e)}")
            
    def update_plot(self, *args):
        if self.raw_data is None: return
        
        ws = int(self.window_size_var.get())
        ss = int(self.step_size_var.get())
        sr = int(self.search_range_var.get())
        
        self.lbl_ws.config(text=str(ws))
        self.lbl_ss.config(text=str(ss))
        self.lbl_sr.config(text=str(sr))
        
        self.ax.clear()
        
        # 均匀挑选 6 根代表性的子载波进行绘制，避免线条过多导致杂乱
        num_subcarriers = self.raw_data.shape[1]
        selected_idx = np.linspace(0, num_subcarriers - 1, 6, dtype=int)
        for idx in selected_idx:
            self.ax.plot(self.raw_data[:, idx], alpha=0.7, linewidth=1.5, label=f"子载波 #{idx}")
        
        diff = np.abs(np.diff(self.raw_data, axis=0))
        score = np.mean(diff, axis=1)
        rolling_activity = np.convolve(score, np.ones(ws), mode='valid')
        
        manual_start = self.manual_start_var.get()
        if manual_start >= 0 and manual_start < len(self.raw_data) - ws:
            best_start = manual_start
            auto = False
        else:
            if len(rolling_activity) > 0:
                best_start = np.argmax(rolling_activity)
            else:
                best_start = 0
            auto = True
            
        search_start = max(0, best_start - sr)
        search_end = min(len(self.raw_data) - ws, best_start + sr)
        
        if len(rolling_activity) > 0:
            max_val = np.max(self.raw_data)
            norm_activity = rolling_activity / np.max(rolling_activity) * (max_val * 0.95)
            padded_activity = np.pad(norm_activity, (0, len(self.raw_data) - len(norm_activity)), 'constant')
            self.ax.plot(padded_activity, color='black', label="动作能量曲线 (Activity)", linestyle='--', linewidth=2.5)
        
        self.ax.axvspan(search_start, search_end + ws, color='#4CAF50', alpha=0.15, label="搜索范围 (Search Area)")
        
        count = 0
        for start in range(search_start, search_end + 1, ss):
            end = start + ws
            if count == 0:
                self.ax.axvspan(start, end, color='#D32F2F', alpha=0.2, ymin=0.1, ymax=0.9, label="提取的窗口 (Extracted Windows)")
            else:
                self.ax.axvspan(start, end, color='#D32F2F', alpha=0.2, ymin=0.1, ymax=0.9)
            count += 1
            
        self.ax.axvline(best_start, color='#D32F2F', linestyle='-', linewidth=2.5, label="最强动作起点 (Best Start)")
        
        title_text = f"提取了 {count} 个窗口 (窗口:{ws}, 步长:{ss}, 宽度:{sr})"
        if not auto: title_text += " [手动选点]"
        self.ax.set_title(title_text, fontsize=14, fontweight='bold')
        self.ax.legend(loc="upper right", fontsize=10, ncol=2)
        self.ax.grid(True, linestyle=':', alpha=0.7)
        
        self.canvas.draw()
        
    def on_click(self, event):
        if event.inaxes != self.ax: return
        if event.button == 3 or event.button == 1:
            x = int(event.xdata)
            ws = int(self.window_size_var.get())
            if 0 <= x < len(self.raw_data) - ws:
                self.manual_start_var.set(x)
                self.update_plot()
            
    def add_to_dataset(self):
        if self.raw_data is None: return
        
        ws = int(self.window_size_var.get())
        ss = int(self.step_size_var.get())
        sr = int(self.search_range_var.get())
        label = self.label_var.get()
        
        manual_start = self.manual_start_var.get()
        if manual_start >= 0 and manual_start < len(self.raw_data) - ws:
            best_start = manual_start
        else:
            diff = np.abs(np.diff(self.raw_data, axis=0))
            score = np.mean(diff, axis=1)
            rolling_activity = np.convolve(score, np.ones(ws), mode='valid')
            best_start = np.argmax(rolling_activity) if len(rolling_activity) > 0 else 0
            
        search_start = max(0, best_start - sr)
        search_end = min(len(self.raw_data) - ws, best_start + sr)
        
        count = 0
        skipped = 0
        for start in range(search_start, search_end + 1, ss):
            end = start + ws
            window = self.raw_data[start:end]
            
            # --- 滤波与特征提取 ---
            filtered_window = apply_filtering(window)
            hp_window = apply_highpass_filter(filtered_window, fs=self.fs)
            
            if self.use_pca_stft_var.get():
                feat = extract_stft_features(hp_window)
            else:
                feat = hp_window
            
            if feat is not None:
                self.X_collected.append(feat)
                self.y_collected.append(label)
                count += 1
            else:
                skipped += 1
            
        self.lbl_stats.config(text=f"📦 已收集样本数: {len(self.X_collected)}")
        
        # 自动跳转到下一个文件
        if self.current_file_idx < len(self.csv_files) - 1:
            self.next_file()
        else:
            messagebox.showinfo("成功", f"已提取 {count} 个样本。\n\n当前已经是文件列表的最后一个了，你可以随时导出数据集！")
        
    def export_dataset(self):
        if len(self.X_collected) == 0:
            messagebox.showwarning("警告", "没有收集到任何样本！")
            return
            
        save_path_X = filedialog.asksaveasfilename(defaultextension=".npy", initialfile="X_custom.npy", title="保存 X 数据集")
        if not save_path_X: return
        save_path_y = save_path_X.replace("X_", "y_")
        
        X_arr = np.array(self.X_collected, dtype=np.float32)
        y_arr = np.array(self.y_collected, dtype=np.int64)
        
        np.save(save_path_X, X_arr)
        np.save(save_path_y, y_arr)
        
        messagebox.showinfo("成功", f"数据集导出成功！\n保存至:\n{save_path_X}\n{save_path_y}\n总样本数: {len(X_arr)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedAnnotatorApp(root)
    root.mainloop()
