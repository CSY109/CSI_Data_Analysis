import os
import numpy as np
import pandas as pd
import csiread
from scipy import signal
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ================= Processing Parameters =================
CSI_SAMPLE_RATE = 1000   # Intel CSI 典型包率 (Hz)
HIGHPASS_CUTOFF = 0.5    # 高通截止频率 (Hz)，去除静态分量
STFT_NPERSEG = 100
STFT_NOVERLAP = 50
STFT_FREQ_BINS = 25      # 保留前 25 个频段（与 transform.py 一致）

# ================= 1. CSI Data Parsing Utilities =================
class Bfee:
    @staticmethod
    def from_file(filename):
        csidata = csiread.Intel(filename)
        csidata.read()
        return csidata

def get_scaled_csi_amplitude(csidata):
    """Extracts scaled amplitude from Intel CSI data."""
    csi_matrix = csidata.get_scaled_csi()[:, :, 0:3, 0] 
    true_amplitude = np.abs(csi_matrix)
    data_90 = np.transpose(true_amplitude, (0, 2, 1)).reshape(true_amplitude.shape[0], 90)
    return data_90

def apply_filtering(data, window_size=3):
    """Applies median filter to each subcarrier."""
    filtered = np.zeros_like(data)
    for i in range(data.shape[1]):
        filtered[:, i] = signal.medfilt(data[:, i], kernel_size=window_size)
    return filtered

def apply_highpass_filter(data, fs=CSI_SAMPLE_RATE, cutoff=HIGHPASS_CUTOFF, order=4):
    """Butterworth 高通滤波，逐子载波去除低频漂移。"""
    nyq = 0.5 * fs
    wn = cutoff / nyq
    if wn >= 1.0 or len(data) < 3 * order:
        return data.astype(float)
    b, a = signal.butter(order, wn, btype='high', analog=False)
    filtered = np.zeros_like(data, dtype=float)
    for i in range(data.shape[1]):
        filtered[:, i] = signal.filtfilt(b, a, data[:, i].astype(float))
    return filtered

def compute_stft_spectrogram(signal_1d, fs=CSI_SAMPLE_RATE,
                             nperseg=STFT_NPERSEG, noverlap=STFT_NOVERLAP,
                             n_freq_bins=STFT_FREQ_BINS):
    """对一维信号做 STFT，返回幅度谱 (freq_bins, time_bins)。"""
    n = len(signal_1d)
    nperseg = min(nperseg, n)
    if nperseg < 4:
        return None
    noverlap = min(noverlap, nperseg - 1)
    _, _, Zxx = signal.stft(signal_1d.astype(float), fs=fs,
                            nperseg=nperseg, noverlap=noverlap)
    return np.abs(Zxx[:n_freq_bins, :])

def compute_stft_waveform(signal_1d, fs=CSI_SAMPLE_RATE,
                          nperseg=STFT_NPERSEG, noverlap=STFT_NOVERLAP,
                          n_freq_bins=STFT_FREQ_BINS):
    """将 STFT 前 n 频段幅度均值插值到包序号，得到一维波形。"""
    spec = compute_stft_spectrogram(signal_1d, fs, nperseg, noverlap, n_freq_bins)
    if spec is None:
        return None
    energy = np.mean(spec, axis=0)
    n_packets = len(signal_1d)
    x_in = np.linspace(0, n_packets - 1, len(energy))
    return np.interp(np.arange(n_packets), x_in, energy)

def extract_stft_features(chunk_2d):
    """对片段 (packets, subcarriers) 做 PCA + STFT，得到时频特征。"""
    if len(chunk_2d) < 10:
        return None
    pc1 = PCA(n_components=1).fit_transform(chunk_2d).flatten()
    return compute_stft_spectrogram(pc1)

# ================= 2. UI Application =================
class CSILabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("CSI 多文件自由标注工具")
        self.root.geometry("1400x900")
        
        # State
        self.files = []           # List of { 'name', 'path', 'data', 'pca' }
        self.current_file_idx = -1
        self.labeled_segments = [] # List of { 'file_idx', 'start', 'end', 'label_id' }
        self.current_selection = None # (start, end)
        self.view_mode = tk.StringVar(value="PCA")
        self.fixed_length_var = tk.StringVar(value="") # Empty means free-dragging
        
        self.label_config = [
            (0, "Walking (走)", "#4caf50"),
            (1, "Sitting (坐下)", "#2196f3"),
            (2, "Standing (站起)", "#ff9800"),
            (3, "Falling (跌倒)", "#f44336"),
            (4, "Other (其他)", "#9e9e9e")
        ]
        
        self.setup_ui()

    def setup_ui(self):
        # Left Panel: File List & Segment List
        left_panel = ttk.Frame(self.root, width=300, padding=10)
        left_panel.pack(side=tk.LEFT, fill=tk.Y)
        
        ttk.Label(left_panel, text="已加载文件:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.file_listbox = tk.Listbox(left_panel, height=10)
        self.file_listbox.pack(fill=tk.X, pady=5)
        self.file_listbox.bind('<<ListboxSelect>>', self.on_file_select)
        
        ttk.Button(left_panel, text="添加 .dat 文件", command=self.open_files).pack(fill=tk.X, pady=2)
        
        ttk.Separator(left_panel, orient='horizontal').pack(fill=tk.X, pady=10)
        
        ttk.Label(left_panel, text="已标注片段:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.segment_listbox = tk.Listbox(left_panel, height=15)
        self.segment_listbox.pack(fill=tk.X, pady=5)
        
        ttk.Button(left_panel, text="删除选中片段", command=self.delete_segment).pack(fill=tk.X)

        # Right Panel: Plot and Controls
        right_panel = ttk.Frame(self.root, padding=10)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Top Controls
        top_ctrl = ttk.Frame(right_panel)
        top_ctrl.pack(fill=tk.X)
        
        ttk.Label(top_ctrl, text="可视化模式:").pack(side=tk.LEFT, padx=5)
        for m, label in [("Raw", "原始"), ("Highpass", "高通"), ("PCA", "PCA"), ("STFT", "STFT")]:
            ttk.Radiobutton(top_ctrl, text=label, variable=self.view_mode, value=m, command=self.refresh_plot).pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(top_ctrl, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        ttk.Label(top_ctrl, text="固定选取长度:").pack(side=tk.LEFT, padx=5)
        self.len_entry = ttk.Entry(top_ctrl, textvariable=self.fixed_length_var, width=8)
        self.len_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(top_ctrl, text="(空为自由拖动)").pack(side=tk.LEFT, padx=2)

        self.info_label = ttk.Label(top_ctrl, text="提示: 在图上拖动鼠标选择区域", foreground="blue")
        self.info_label.pack(side=tk.RIGHT, padx=10)

        # Plot Area
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, right_panel)
        self.toolbar.update()
        
        # Rectangle Selector (Draggable & Resizable)
        self.selector = RectangleSelector(self.ax, self.on_select,
                                          useblit=True,
                                          interactive=True,
                                          props=dict(facecolor='red', edgecolor='black', alpha=0.4, fill=True))
        
        # Follower Ghost Span (For MATLAB-style stamping)
        self.ghost_span = None
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('button_press_event', self.on_mouse_click)

        # Bottom Controls: Labels
        bottom_ctrl = ttk.Frame(right_panel, padding=10)
        bottom_ctrl.pack(fill=tk.X)
        
        ttk.Label(bottom_ctrl, text="标注选区为:", font=("Arial", 11, "bold")).pack(side=tk.LEFT, padx=10)
        for lid, name, color in self.label_config:
            btn = tk.Button(bottom_ctrl, text=name, bg=color, fg="white", 
                           command=lambda l=lid: self.add_label(l))
            btn.pack(side=tk.LEFT, padx=5)
            
        ttk.Button(bottom_ctrl, text="导出全量数据集 (.npy)", command=self.save_dataset).pack(side=tk.RIGHT, padx=10)

    def open_files(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("CSI Data", "*.dat")])
        if not file_paths:
            return
        
        for path in file_paths:
            try:
                # Avoid duplicate loading
                if any(f['path'] == path for f in self.files):
                    continue
                    
                csidata = Bfee.from_file(path)
                # Warm-up call for csiread 1.4.0+
                if hasattr(csidata, 'get_total_rss'):
                    csidata.get_total_rss()
                    
                raw_amp = get_scaled_csi_amplitude(csidata)
                filtered_amp = apply_filtering(raw_amp)
                hp_amp = apply_highpass_filter(filtered_amp)
                
                pca = PCA(n_components=1)
                pca_result = pca.fit_transform(hp_amp).flatten()
                stft_spec = compute_stft_spectrogram(pca_result)
                stft_wave = compute_stft_waveform(pca_result)
                
                self.files.append({
                    'name': os.path.basename(path),
                    'path': path,
                    'raw': filtered_amp,
                    'data': hp_amp,
                    'pca': pca_result,
                    'stft': stft_spec,
                    'stft_wave': stft_wave,
                })
                self.file_listbox.insert(tk.END, os.path.basename(path))
                
            except Exception as e:
                messagebox.showerror("加载错误", f"文件 {os.path.basename(path)} 加载失败: {e}")

        if self.current_file_idx == -1 and self.files:
            self.file_listbox.selection_set(0)
            self.on_file_select(None)

    def on_file_select(self, event):
        selection = self.file_listbox.curselection()
        if not selection:
            return
        
        self.current_file_idx = selection[0]
        self.refresh_plot()
        self.update_segment_listbox()

    def on_select(self, eclick, erelease):
        xmin, xmax = eclick.xdata, erelease.xdata
        if xmin > xmax: xmin, xmax = xmax, xmin
        
        # Check if fixed length is set
        fixed_len_str = self.fixed_length_var.get().strip()
        if fixed_len_str:
            try:
                flen = int(fixed_len_str)
                xmax = xmin + flen
            except ValueError:
                pass
                
        self.current_selection = (int(xmin), int(xmax))
        self.info_label.config(text=f"当前选区: {self.current_selection[0]} 到 {self.current_selection[1]} (长度: {self.current_selection[1]-self.current_selection[0]})")

    def add_label(self, label_id):
        if self.current_file_idx == -1 or self.current_selection is None:
            messagebox.showwarning("警告", "请先选择一个文件并在图上拖动选择区域！")
            return
        
        start, end = self.current_selection
        if end - start < 10:
            messagebox.showwarning("警告", "选区太短！")
            return
            
        self.labeled_segments.append({
            'file_idx': self.current_file_idx,
            'start': start,
            'end': end,
            'label_id': label_id
        })
        
        self.current_selection = None
        self.refresh_plot()
        self.update_segment_listbox()

    def update_segment_listbox(self):
        self.segment_listbox.delete(0, tk.END)
        for i, seg in enumerate(self.labeled_segments):
            file_name = self.files[seg['file_idx']]['name']
            label_name = next(n for lid, n, c in self.label_config if lid == seg['label_id'])
            self.segment_listbox.insert(tk.END, f"[{i}] {file_name}: {seg['start']}-{seg['end']} ({label_name})")

    def delete_segment(self):
        selection = self.segment_listbox.curselection()
        if not selection:
            return
        
        idx = selection[0]
        del self.labeled_segments[idx]
        self.update_segment_listbox()
        self.refresh_plot()

    def refresh_plot(self):
        if self.current_file_idx == -1:
            return
        
        file_data = self.files[self.current_file_idx]
        self.ax.clear()
        
        # Re-initialize RectangleSelector safely
        self.selector = RectangleSelector(self.ax, self.on_select,
                                          useblit=True,
                                          interactive=True,
                                          props=dict(facecolor='red', edgecolor='black', alpha=0.4, fill=True))
        
        mode = self.view_mode.get()
        n_packets = len(file_data['pca'])
        if mode == "Raw":
            self.ax.plot(file_data['raw'][:, :10], alpha=0.5, lw=0.5)
            self.ax.set_title(f"原始振幅 (中值滤波, 前10子载波) - {file_data['name']}")
            self.ax.set_ylabel("振幅")
        elif mode == "Highpass":
            self.ax.plot(file_data['data'][:, :10], alpha=0.5, lw=0.5)
            self.ax.set_title(f"高通滤波波形 (前10子载波) - {file_data['name']}")
            self.ax.set_ylabel("振幅")
        elif mode == "PCA":
            self.ax.plot(file_data['pca'], color='blue', lw=1)
            self.ax.set_title(f"PCA 主成分 (高通后) - {file_data['name']}")
            self.ax.set_ylabel("投影值")
        else:
            stft_wave = file_data.get('stft_wave')
            if stft_wave is not None:
                self.ax.plot(stft_wave, color='purple', lw=1)
                self.ax.set_title(f"STFT 能量波形 (前{STFT_FREQ_BINS}频段均值) - {file_data['name']}")
                self.ax.set_ylabel("STFT 幅度")
            else:
                self.ax.text(0.5, 0.5, "数据过短，无法计算 STFT",
                             ha='center', va='center', transform=self.ax.transAxes)
        
        # Draw existing segments for THIS file
        for seg in self.labeled_segments:
            if seg['file_idx'] == self.current_file_idx:
                _, _, color = next(c for c in self.label_config if c[0] == seg['label_id'])
                self.ax.axvspan(seg['start'], seg['end'], color=color, alpha=0.3)
        
        self.ax.set_xlabel("Packets")
        self.canvas.draw()

    def save_dataset(self):
        if not self.labeled_segments:
            messagebox.showwarning("警告", "没有标注任何数据！")
            return
            
        # Group by length? Or fixed length?
        # The user wants "one dataset". If lengths are different, we might need to pad/truncate
        # or just save as a list of arrays.
        # But usually .npy wants a fixed shape.
        
        # Let's ask if they want to force a fixed length (e.g. 250)
        # For now, I'll extract them as they are. If shapes vary, I'll warn.
        
        X = []
        y = []
        
        lengths = [seg['end'] - seg['start'] for seg in self.labeled_segments]
        unique_lengths = set(lengths)
        
        target_len = 250 # Default from original script
        if len(unique_lengths) > 1:
            msg = (f"检测到不同长度的片段 ({unique_lengths})。是否统一强制切分为 {target_len} 包再算 STFT？\n"
                   f"取消则尝试保存原始 STFT 形状（若不一致会报错）。")
            res = messagebox.askyesnocancel("长度不一致", msg)
            if res is True:
                force_fixed = True
            elif res is False:
                force_fixed = False
            else:
                return
        else:
            force_fixed = False

        skipped = 0
        for seg in self.labeled_segments:
            data = self.files[seg['file_idx']]['data']
            s, e = seg['start'], seg['end']
            
            s = max(0, s)
            e = min(len(data), e)
            
            chunk = data[s:e]
            
            if force_fixed:
                if len(chunk) >= target_len:
                    chunk = chunk[:target_len]
                else:
                    pad_width = ((0, target_len - len(chunk)), (0, 0))
                    chunk = np.pad(chunk, pad_width, mode='constant')
            
            stft_feat = extract_stft_features(chunk)
            if stft_feat is None:
                skipped += 1
                continue
            
            X.append(stft_feat)
            y.append(seg['label_id'])
        
        if not X:
            messagebox.showwarning("警告", "没有可导出的 STFT 特征（片段可能过短）！")
            return
        
        try:
            X = np.array(X)
            y = np.array(y)
            
            save_path_X = filedialog.asksaveasfilename(defaultextension=".npy", initialfile="X_dataset.npy", title="保存 STFT 特征数据")
            if not save_path_X: return
            save_path_y = save_path_X.replace("X_dataset", "y_dataset")
            
            np.save(save_path_X, X)
            np.save(save_path_y, y)
            
            skip_msg = f"\n跳过过短片段: {skipped}" if skipped else ""
            messagebox.showinfo("成功",
                f"STFT 数据集已导出！\n文件数: {len(self.files)}\n样本数: {len(X)}\n形状: {X.shape} (频段×时间){skip_msg}")
        except Exception as e:
            messagebox.showerror("保存失败", f"数据形状可能不一致，无法转换为 Numpy 数组。\n错误: {e}")

    def on_mouse_move(self, event):
        if event.inaxes != self.ax:
            if self.ghost_span:
                self.ghost_span.remove()
                self.ghost_span = None
                self.canvas.draw_idle()
            return
            
        fixed_len_str = self.fixed_length_var.get().strip()
        if not fixed_len_str:
            if self.ghost_span:
                self.ghost_span.remove()
                self.ghost_span = None
                self.canvas.draw_idle()
            return
            
        try:
            flen = int(fixed_len_str)
            xmin = event.xdata
            xmax = xmin + flen
            
            if self.ghost_span:
                self.ghost_span.remove()
            
            self.ghost_span = self.ax.axvspan(xmin, xmax, color='red', alpha=0.2)
            self.canvas.draw_idle()
        except:
            pass

    def on_mouse_click(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
            
        # If toolbar is in a mode (zoom/pan), don't stamp
        if self.toolbar.mode != "":
            return
            
        fixed_len_str = self.fixed_length_var.get().strip()
        if fixed_len_str:
            try:
                flen = int(fixed_len_str)
                xmin = int(event.xdata)
                xmax = xmin + flen
                self.current_selection = (xmin, xmax)
                self.info_label.config(text=f"已选定区域: {xmin} 到 {xmax} (长度: {flen})")
            except:
                pass

if __name__ == "__main__":
    root = tk.Tk()
    app = CSILabeler(root)
    root.mainloop()
