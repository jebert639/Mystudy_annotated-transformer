"""
model_gui.py - Transformer 模型推理小界面 (Tkinter, Python 自带, 无需安装任何依赖)

功能:
    - 自动扫描项目目录下所有 output*/ 里的 .pt 模型, 列出 epoch / val_loss / 配置
    - 选中模型 -> 加载 (后台线程, 不卡界面), 显示参数量 / 设备 / 训练信息
    - 输入前缀 -> 生成补全 (温度 0~1.5 可调, 0 为纯贪心; 最大长度可调)
    - 「查看训练曲线」用系统看图工具打开该模型目录下的 loss_curve.png
    - 也可以「浏览...」选择任意位置的 .pt 文件

启动:
    python model_gui.py
    或双击 launch_gui.bat
    或 python transformer_api.py gui
"""

import os
import sys
import time
import queue
import threading

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, font as tkfont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformer_api import PROJECT_ROOT, list_models, load_model, generate


class InferenceGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("Transformer 模型推理工具")
        self.root.geometry("1120x700")
        self.root.minsize(940, 580)

        self.q = queue.Queue()      # 后台线程 -> UI 线程的消息队列
        self.models = []            # 扫描到的模型列表
        self.pack = None            # 当前已加载的模型包 (load_model 返回值)
        self._loading = False
        self._generating = False

        self._setup_fonts()
        self._build_ui()
        self._poll_queue()
        self.refresh_models()

    # ------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------

    def _setup_fonts(self):
        try:
            for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont"):
                tkfont.nametofont(name).configure(family="Microsoft YaHei UI", size=10)
        except tk.TclError:
            pass

    def _build_ui(self):
        # ---------- 顶部工具栏 ----------
        top = ttk.Frame(self.root, padding=(8, 8, 8, 4))
        top.pack(fill="x")

        self.btn_refresh = ttk.Button(top, text="🔄 刷新模型列表", command=self.refresh_models)
        self.btn_refresh.pack(side="left")

        self.btn_browse = ttk.Button(top, text="📂 浏览 .pt 文件...", command=self.browse_model)
        self.btn_browse.pack(side="left", padx=(6, 0))

        self.btn_load = ttk.Button(top, text="⚡ 加载选中模型", command=self.load_selected, state="disabled")
        self.btn_load.pack(side="left", padx=(6, 0))

        self.btn_curve = ttk.Button(top, text="📈 查看训练曲线", command=self.show_loss_curve, state="disabled")
        self.btn_curve.pack(side="left", padx=(6, 0))

        # ---------- 底部状态栏 (先占位, 再让主体填剩余空间) ----------
        self.status_var = tk.StringVar(value="就绪")
        status = ttk.Label(self.root, textvariable=self.status_var, relief="sunken",
                           anchor="w", padding=(8, 3))
        status.pack(fill="x", side="bottom")

        # ---------- 主体: 左右分栏 ----------
        body = ttk.Frame(self.root, padding=(8, 4))
        body.pack(fill="both", expand=True)

        # 左侧: 模型列表
        left = ttk.LabelFrame(body, text=" 可用模型 (双击加载) ", padding=4)
        left.pack(side="left", fill="both", expand=False, padx=(0, 8))

        columns = ("path", "kind", "epoch", "val", "cfg", "size", "mtime")
        self.tree = ttk.Treeview(left, columns=columns, show="headings", height=14)
        headings = {
            "path": ("模型文件", 170), "kind": ("类型", 44), "epoch": ("Epoch", 46),
            "val": ("Val Loss", 58), "cfg": ("配置 (d_model/N/h/d_ff)", 190),
            "size": ("大小", 52), "mtime": ("修改时间", 92),
        }
        for col, (text, width) in headings.items():
            anchor = "e" if col in ("epoch", "val", "size") else "w"
            self.tree.heading(col, text=text)
            self.tree.column(col, width=width, anchor=anchor, stretch=False)
        self.tree.column("cfg", stretch=True)

        vsb = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Double-1>", lambda e: self.load_selected())

        # 右侧: 详情 + 生成
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)

        info_frame = ttk.LabelFrame(right, text=" 模型详情 ", padding=6)
        info_frame.pack(fill="x")
        self.info_text = tk.Text(info_frame, height=5, width=42, wrap="word", relief="flat",
                                 background="#f5f5f5", state="disabled")
        self.info_text.pack(fill="x")
        self.info_text.config(state="normal")
        self.info_text.insert("1.0", "在左侧选择一个模型查看详情；双击模型行即可加载。")
        self.info_text.config(state="disabled")

        # 提示词输入
        prompt_frame = ttk.LabelFrame(right, text=" 输入前缀 (模型会续写后面的内容) ", padding=6)
        prompt_frame.pack(fill="x", pady=(8, 0))
        self.prompt_box = tk.Text(prompt_frame, height=4, width=42, wrap="word",
                                  font=("Microsoft YaHei UI", 10))
        self.prompt_box.pack(fill="x")
        self.prompt_box.insert("1.0", "I had always thought")

        # 生成参数行 (第一行: 参数; 第二行: 按钮, 避免窄窗口时按钮被挤出)
        param_frame = ttk.Frame(right, padding=(0, 8, 0, 0))
        param_frame.pack(fill="x")

        row1 = ttk.Frame(param_frame)
        row1.pack(fill="x")
        ttk.Label(row1, text="温度:").pack(side="left")
        self.temp_var = tk.DoubleVar(value=0.7)
        self.temp_scale = ttk.Scale(row1, from_=0.0, to=1.5, variable=self.temp_var,
                                    length=130, command=self._on_temp_change)
        self.temp_scale.pack(side="left", padx=(4, 2))
        self.temp_label = ttk.Label(row1, text="0.70", width=4)
        self.temp_label.pack(side="left")
        ttk.Label(row1, text="(0=贪心)").pack(side="left", padx=(2, 16))
        ttk.Label(row1, text="长度:").pack(side="left")
        self.maxlen_var = tk.IntVar(value=100)
        ttk.Spinbox(row1, from_=10, to=500, increment=10, width=6,
                    textvariable=self.maxlen_var).pack(side="left", padx=(4, 0))

        row2 = ttk.Frame(param_frame)
        row2.pack(fill="x", pady=(6, 0))
        self.btn_generate = ttk.Button(row2, text="▶ 生成补全", command=self.start_generate,
                                       state="disabled")
        self.btn_generate.pack(side="right")
        self.btn_clear = ttk.Button(row2, text="清空结果", command=self._clear_results)
        self.btn_clear.pack(side="right", padx=(0, 8))

        # 结果区
        result_frame = ttk.LabelFrame(right, text=" 生成结果 ", padding=4)
        result_frame.pack(fill="both", expand=True, pady=(8, 0))
        self.result_box = scrolledtext.ScrolledText(result_frame, wrap="word", state="disabled",
                                                    width=42, font=("Microsoft YaHei UI", 10))
        self.result_box.pack(fill="both", expand=True)

    # ------------------------------------------------------------
    # 模型扫描 / 列表
    # ------------------------------------------------------------

    def refresh_models(self):
        self.btn_refresh.config(state="disabled")
        self.set_status("正在扫描模型 (读取 checkpoint 信息, 可能需要几秒)...")
        threading.Thread(target=self._scan_worker, daemon=True).start()

    def _scan_worker(self):
        try:
            models = list_models(PROJECT_ROOT, verbose=False)
            self.q.put(("models", models))
        except Exception as e:
            self.q.put(("scan_error", e))

    def _fill_tree(self, models):
        self.models = models
        self.tree.delete(*self.tree.get_children())
        for i, m in enumerate(models):
            c = m["config"]
            cfg = (f"d_model={c.get('d_model', '?')} N={c.get('N', '?')} h={c.get('h', '?')} "
                   f"d_ff={c.get('d_ff', '?')}") if c else "-"
            val = f"{m['val_loss']:.3f}" if m["val_loss"] is not None else "-"
            kind_label = {"best": "best", "final": "final"}.get(m["kind"], "其他")
            self.tree.insert("", "end", iid=str(i), values=(
                m["path"], kind_label, m["epoch"], val, cfg,
                f"{m['size_mb']:.0f}M", m["mtime"],
            ))
        self.set_status(f"找到 {len(models)} 个模型。双击或选中后点「加载选中模型」。")

    def _selected_path(self):
        sel = self.tree.selection()
        if not sel:
            return None
        return self.models[int(sel[0])]["abs_path"]

    def _on_select(self, _event=None):
        has_sel = bool(self.tree.selection())
        self.btn_load.config(state="normal" if has_sel and not self._loading else "disabled")
        self.btn_curve.config(state="normal" if has_sel else "disabled")
        if has_sel:
            self._show_info(self.models[int(self.tree.selection()[0])])

    # ------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------

    def load_selected(self):
        if self._loading:
            return
        path = self._selected_path()
        if path is None:
            messagebox.showinfo("提示", "请先在左侧列表中选择一个模型")
            return
        self._load_path(path)

    def browse_model(self):
        path = filedialog.askopenfilename(
            title="选择模型 checkpoint",
            initialdir=PROJECT_ROOT,
            filetypes=[("PyTorch 模型", "*.pt"), ("所有文件", "*.*")],
        )
        if path:
            self._load_path(path)

    def _load_path(self, path):
        self._loading = True
        self.btn_load.config(state="disabled")
        self.btn_generate.config(state="disabled")
        rel = os.path.relpath(path, PROJECT_ROOT)
        self.set_status(f"正在加载 {rel} (首次需构建模型, 可能需要十几秒)...")
        threading.Thread(target=self._load_worker, args=(path,), daemon=True).start()

    def _load_worker(self, path):
        try:
            pack = load_model(path)
            self.q.put(("loaded", pack))
        except Exception as e:
            self.q.put(("load_error", path, e))

    def _after_loaded(self, pack):
        self.pack = pack
        self._loading = False
        self.btn_generate.config(state="normal")
        if self.tree.selection():
            self.btn_load.config(state="normal")
        self._show_info(pack["info"], loaded=True)
        self.set_status(f"已加载 {pack['info']['path']}，可以点击「生成补全」了。")

    # ------------------------------------------------------------
    # 信息展示
    # ------------------------------------------------------------

    def _show_info(self, m, loaded=False):
        lines = []
        if loaded:
            c = m["config"]
            lines.append(f"已加载: {m['path']}   (device: {'GPU' if self.pack['device'] == 'cuda' else 'CPU'})")
            lines.append(f"结构: d_model={c.get('d_model')}  N={c.get('N')}  h={c.get('h')}  "
                         f"d_ff={c.get('d_ff')}  dropout={c.get('dropout')}  vocab={c.get('vocab_size')}")
            lines.append(f"参数量: {m['total_params']:,}")
            ep = m["epoch"]
            ep_txt = f"第 {ep} epoch" if isinstance(ep, int) else str(ep)
            lines.append(f"训练信息: {ep_txt}  |  val_loss={m['val_loss'] if m['val_loss'] is not None else '-'}"
                         f"  train_loss={m['train_loss'] if m['train_loss'] is not None else '-'}")
        else:
            c = m["config"]
            lines.append(f"选中: {m['path']}   [{m['kind']}]   epoch={m['epoch']}")
            lines.append(f"结构: d_model={c.get('d_model', '?')}  N={c.get('N', '?')}  h={c.get('h', '?')}  "
                         f"d_ff={c.get('d_ff', '?')}  vocab={c.get('vocab_size', '?')}")
            lines.append(f"val_loss={m['val_loss'] if m['val_loss'] is not None else '-'}   "
                         f"大小={m['size_mb']:.1f}M   修改时间={m['mtime']}")
            lines.append("双击该行或点「加载选中模型」开始加载。")
        self.info_text.config(state="normal")
        self.info_text.delete("1.0", "end")
        self.info_text.insert("1.0", "\n".join(lines))
        self.info_text.config(state="disabled")

    def show_loss_curve(self):
        path = self._selected_path()
        if path is None:
            messagebox.showinfo("提示", "请先在左侧列表中选择一个模型")
            return
        curve = os.path.join(os.path.dirname(path), "loss_curve.png")
        if os.path.exists(curve):
            os.startfile(curve)  # Windows 用默认看图工具打开
        else:
            messagebox.showwarning("未找到", f"该模型目录下没有 loss_curve.png:\n{curve}")

    # ------------------------------------------------------------
    # 生成
    # ------------------------------------------------------------

    def _on_temp_change(self, _value=None):
        self.temp_label.config(text=f"{self.temp_var.get():.2f}")

    def start_generate(self):
        if self.pack is None or self._generating:
            return
        prompt = self.prompt_box.get("1.0", "end").strip()
        if not prompt:
            messagebox.showinfo("提示", "请先输入前缀文本")
            return
        temperature = round(self.temp_var.get(), 2)
        max_len = int(self.maxlen_var.get())

        self._generating = True
        self.btn_generate.config(state="disabled")
        self.set_status(f"正在生成 (温度={temperature}, 最大长度={max_len})...")
        threading.Thread(target=self._generate_worker,
                         args=(prompt, max_len, temperature), daemon=True).start()

    def _generate_worker(self, prompt, max_len, temperature):
        try:
            t0 = time.time()
            text = generate(self.pack, prompt, max_len=max_len, temperature=temperature)
            self.q.put(("result", prompt, text, time.time() - t0,
                        self.pack["info"]["path"], temperature))
        except Exception as e:
            self.q.put(("gen_error", e))

    def _append_result(self, prompt, text, elapsed, model_name, temperature):
        stamp = time.strftime("%H:%M:%S")
        block = (
            f"{'─' * 64}\n"
            f"[{stamp}] 模型: {model_name}   温度: {temperature}\n"
            f"输入: {prompt}\n"
            f"输出: {text}\n"
            f"(耗时 {elapsed:.1f} 秒)\n"
        )
        self.result_box.config(state="normal")
        self.result_box.insert("end", block)
        self.result_box.see("end")
        self.result_box.config(state="disabled")

    def _clear_results(self):
        self.result_box.config(state="normal")
        self.result_box.delete("1.0", "end")
        self.result_box.config(state="disabled")

    # ------------------------------------------------------------
    # 消息队列轮询 (后台线程 -> UI)
    # ------------------------------------------------------------

    def _poll_queue(self):
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]

                if kind == "models":
                    self._fill_tree(msg[1])
                    self.btn_refresh.config(state="normal")
                elif kind == "scan_error":
                    self.btn_refresh.config(state="normal")
                    self.set_status("扫描失败")
                    messagebox.showerror("扫描失败", str(msg[1]))

                elif kind == "loaded":
                    self._after_loaded(msg[1])
                elif kind == "load_error":
                    self._loading = False
                    if self.tree.selection():
                        self.btn_load.config(state="normal")
                    self.set_status("模型加载失败")
                    messagebox.showerror("加载失败", f"{msg[1]}\n\n{msg[2]}")

                elif kind == "result":
                    self._generating = False
                    self.btn_generate.config(state="normal")
                    self._append_result(*msg[1:])
                    self.set_status("生成完成。")
                elif kind == "gen_error":
                    self._generating = False
                    self.btn_generate.config(state="normal")
                    self.set_status("生成失败")
                    messagebox.showerror("生成失败", str(msg[1]))

        except queue.Empty:
            pass
        self.root.after(120, self._poll_queue)

    def set_status(self, text):
        self.status_var.set(text)


def main():
    root = tk.Tk()
    InferenceGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
