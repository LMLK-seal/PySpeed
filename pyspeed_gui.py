# In pyspeed_project/pyspeed_gui.py

import os
import sys
import tempfile
import time
import traceback
import threading
import ast
import difflib

try:
    import customtkinter as ctk
    from tkinter import messagebox
except Exception as e:
    print("customtkinter is required. Install with: pip install customtkinter"); raise

from pyspeed import analyzer

def show_diff_preview(original: str, modified: str, parent):
    if not original or not modified: return
    diff_window = ctk.CTkToplevel(parent); diff_window.title("Optimization Preview"); diff_window.geometry("1000x600"); diff_window.transient(parent)
    diff_window.grid_columnconfigure(0, weight=1); diff_window.grid_columnconfigure(1, weight=1); diff_window.grid_rowconfigure(1, weight=1)
    ctk.CTkLabel(diff_window, text="Original Function", font=("", 14, "bold")).grid(row=0, column=0, pady=5)
    ctk.CTkLabel(diff_window, text="Optimized Function", font=("", 14, "bold")).grid(row=0, column=1, pady=5)
    original_box = ctk.CTkTextbox(diff_window, wrap="none", font=("monospace", 12)); original_box.grid(row=1, column=0, sticky="nsew", padx=(10,5), pady=10); original_box.insert("1.0", original)
    modified_box = ctk.CTkTextbox(diff_window, wrap="none", font=("monospace", 12)); modified_box.grid(row=1, column=1, sticky="nsew", padx=(5,10), pady=10); modified_box.insert("1.0", modified)
    original_box.tag_config("del", background="#552222"); modified_box.tag_config("add", background="#225522")
    d = difflib.SequenceMatcher(None, original.splitlines(), modified.splitlines())
    for tag, i1, i2, j1, j2 in d.get_opcodes():
        if tag in ('replace', 'delete'): original_box.tag_add("del", f"{i1 + 1}.0", f"{i2}.end")
        if tag in ('replace', 'insert'): modified_box.tag_add("add", f"{j1 + 1}.0", f"{j2}.end")

def show_suggestion_popup(template: str, parent):
    suggestion_window = ctk.CTkToplevel(parent)
    suggestion_window.title("Actionable Suggestion"); suggestion_window.geometry("800x600"); suggestion_window.transient(parent)
    suggestion_window.grid_rowconfigure(1, weight=1); suggestion_window.grid_columnconfigure(0, weight=1)
    ctk.CTkLabel(suggestion_window, text="PySpeed Refactoring Template", font=("", 14, "bold")).grid(row=0, column=0, pady=10)
    suggestion_box = ctk.CTkTextbox(suggestion_window, wrap="word", font=("monospace", 12)); suggestion_box.grid(row=1, column=0, sticky="nsew", padx=10, pady=5); suggestion_box.insert("1.0", template)
    ctk.CTkLabel(suggestion_window, text="Copy this template and adapt it to your code.", justify="left").grid(row=2, column=0, pady=10, padx=10, sticky="w")

class PySpeedApp:
    def __init__(self, root):
        self.root = root; ctk.set_appearance_mode("System"); ctk.set_default_color_theme("blue")
        self.root.title("PySpeed Accelerator v2.2"); self.root.geometry("1200x800")
        self.script_path, self.script_source, self.optimized_source, self.optimized_path = None, "", None, None
        self.prof_entries = []; self._build_ui()
        self.log("Welcome to PySpeed! Open a script and click 'Profile' to begin.")

    def _build_ui(self):
        top_frame = ctk.CTkFrame(self.root); top_frame.pack(side="top", fill="x", padx=10, pady=10)
        ctk.CTkButton(top_frame, text="1. Open Script", command=self.open_script).pack(side="left", padx=5)
        ctk.CTkButton(top_frame, text="2. Profile", command=self.action_profile).pack(side="left", padx=5)
        ctk.CTkLabel(top_frame, text="Optimization Mode:").pack(side="left", padx=(15, 5))
        self.opt_mode = ctk.StringVar(value="Auto (Recommended)")
        modes = ["Auto (Recommended)", "Numba JIT", "NumPy Vectorize", "Memoization", "Multiprocessing Suggest"]
        ctk.CTkOptionMenu(top_frame, variable=self.opt_mode, values=modes).pack(side="left")
        ctk.CTkButton(top_frame, text="3. Optimize", command=self.action_optimize).pack(side="left", padx=(15, 5))
        ctk.CTkButton(top_frame, text="Run Original", command=lambda: self._run_script_task(self.script_path, "original")).pack(side="left", padx=5)
        ctk.CTkButton(top_frame, text="Run Optimized", command=lambda: self._run_script_task(self.optimized_path, "optimized")).pack(side="left", padx=5)
        ctk.CTkButton(top_frame, text="4. Compare", command=self.compare_timings).pack(side="left", padx=5)
        ctk.CTkButton(top_frame, text="About", command=self.show_about).pack(side="right", padx=5)
        main_paned_window = ctk.CTkFrame(self.root, fg_color="transparent"); main_paned_window.pack(fill="both", expand=True, padx=10, pady=5)
        main_paned_window.grid_columnconfigure(0, weight=1); main_paned_window.grid_rowconfigure(0, weight=1)
        left_right_splitter = ctk.CTkFrame(main_paned_window, fg_color="transparent"); left_right_splitter.pack(fill="both", expand=True)
        left_right_splitter.grid_columnconfigure(0, weight=3); left_right_splitter.grid_columnconfigure(1, weight=2); left_right_splitter.grid_rowconfigure(0, weight=1)
        left_frame = ctk.CTkFrame(left_right_splitter); left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5)); left_frame.grid_rowconfigure(1, weight=1); left_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(left_frame, text="Original Script Source").pack(anchor="w", padx=10, pady=(5,0))
        self.txt_source = ctk.CTkTextbox(left_frame, wrap="none", font=("monospace", 12)); self.txt_source.pack(fill="both", expand=True, padx=5, pady=5)
        right_frame = ctk.CTkFrame(left_right_splitter); right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0)); right_frame.grid_rowconfigure(1, weight=1); right_frame.grid_columnconfigure(0, weight=1)
        tab_view = ctk.CTkTabview(right_frame); tab_view.pack(fill="both", expand=True, padx=5, pady=5)
        tab_profile = tab_view.add("Hotspots"); tab_optimize = tab_view.add("Optimized Source")
        ctk.CTkLabel(tab_profile, text="Hotspots (sorted by Impact Score, * = leaf function)").pack(anchor="w", padx=5)
        self.txt_hotspots = ctk.CTkTextbox(tab_profile, wrap="none", font=("monospace", 11)); self.txt_hotspots.pack(fill="both", expand=True, padx=5, pady=5)
        ctk.CTkLabel(tab_optimize, text="Optimized Source Preview").pack(anchor="w", padx=5)
        self.txt_opt = ctk.CTkTextbox(tab_optimize, wrap="none", font=("monospace", 12)); self.txt_opt.pack(fill="both", expand=True, padx=5, pady=5)
        bottom_frame = ctk.CTkFrame(self.root, height=165); bottom_frame.pack(side="bottom", fill="both", expand=False, padx=10, pady=(5, 10))
        bottom_frame.grid_propagate(False); bottom_frame.grid_columnconfigure(0, weight=1); bottom_frame.grid_rowconfigure(2, weight=1) 
        self.lbl_status = ctk.CTkLabel(bottom_frame, text="Ready", anchor="w"); self.lbl_status.grid(row=0, column=0, sticky="ew", padx=10, pady=(5,0))
        self.progress_bar = ctk.CTkProgressBar(bottom_frame, mode='indeterminate')
        self.txt_log = ctk.CTkTextbox(bottom_frame, font=("monospace", 11)); self.txt_log.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)

    def log(self, msg: str): self.root.after(0, lambda: self._log_threadsafe(msg))
    def _log_threadsafe(self, msg: str): ts = time.strftime("%H:%M:%S"); self.txt_log.insert("end", f"[{ts}] {msg}\n"); self.txt_log.see("end")
    def set_status(self, text: str): self.root.after(0, lambda: self._set_status_threadsafe(text))
    def _set_status_threadsafe(self, text: str): self.lbl_status.configure(text=text)
    
    def open_script(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(filetypes=[("Python files", "*.py")]);
        if not path: return
        self.script_path = path;
        with open(path, "r", encoding="utf-8") as f: self.script_source = f.read()
        self.txt_source.delete("1.0", "end"); self.txt_source.insert("1.0", self.script_source); self.log(f"Opened script: {path}"); self.set_status(f"Loaded {os.path.basename(path)}")
        self.txt_hotspots.delete("1.0", "end"); self.txt_opt.delete("1.0", "end"); self.optimized_source = None; self.optimized_path = None
    
    def action_profile(self):
        if not self.script_path: return self.log("No script loaded.")
        self.set_status("Profiling..."); self.log("Starting profiling...")
        self.progress_bar.grid(row=1, column=0, sticky="ew", padx=5, pady=5); self.progress_bar.start()
        def worker():
            prof_file, err = analyzer.run_profile_on_script(self.script_path)
            def ui_cleanup(error_message=None):
                self.progress_bar.stop(); self.progress_bar.grid_remove()
                if error_message: self.log(f"Profiling error: {error_message}"); self.set_status("Profiling failed")
                else:
                    self.prof_entries = analyzer.parse_profile(prof_file)
                    self.update_hotspots_ui(); self.log("Profiling complete."); self.set_status("Profiling complete")
            self.root.after(0, ui_cleanup, err)
        threading.Thread(target=worker, daemon=True).start()

    def update_hotspots_ui(self):
        self.txt_hotspots.delete("1.0", "end")
        if not self.prof_entries: self.txt_hotspots.insert("1.0", "No hotspots found."); return
        self.txt_hotspots.tag_config("leaf", foreground="#33FF57")
        header = f"{'Function':<30} {'File:Line':<40} {'Impact Score':<18} {'Total Time (s)':<18}\n"; separator = "-" * 106 + "\n"
        self.txt_hotspots.insert("end", header); self.txt_hotspots.insert("end", separator)
        for e in self.prof_entries:
            func_display = f"{e.func_name}{' *' if e.is_leaf else ''}"; loc = f"{os.path.basename(e.file)}:{e.line}"
            line_text = f"{func_display:<30} {loc:<40} {e.impact_score:<18.2e} {e.tottime:<18.6f}\n"
            start_index = self.txt_hotspots.index("end-1c")
            self.txt_hotspots.insert("end", line_text)
            if e.is_leaf: self.txt_hotspots.tag_add("leaf", start_index, self.txt_hotspots.index("end-1c"))
    
    def action_optimize(self):
        if not self.script_source: return self.log("No script loaded.")
        mode = self.opt_mode.get()
        self.set_status(f"Analyzing with mode: {mode}..."); self.log(f"Starting optimization pipeline (Mode: {mode})...")
        def worker():
            try:
                hotspot_names = [e.func_name for e in self.prof_entries]
                result = analyzer.run_optimization_pipeline(self.script_source, hotspot_names, mode, self.prof_entries)
                def ui_update():
                    self.log("--- Analysis Report ---"); [self.log(f"[{s.severity.upper()}] {s.func_name} (L{s.line_no}): {s.message}") for s in result.suggestions]; self.log("-----------------------")
                    
                    if result.rejection_reason:
                        self.log(f"INFO: {result.rejection_reason}")
                        self.set_status("Optimization rejected.")
                    elif result.actionable_suggestion:
                        self.log("Generated an actionable suggestion template.")
                        self.set_status("Suggestion provided.")
                        show_suggestion_popup(result.actionable_suggestion, self.root)
                    elif result.transformed_funcs:
                        func_name, opt_type = list(result.transformed_funcs.items())[0]
                        self.log(f"Transformation applied to '{func_name}' using: {opt_type.upper()}")
                        self.optimized_source = result.modified_source
                        base_dir = os.path.dirname(self.script_path) if self.script_path else tempfile.gettempdir()
                        opt_name = f"pyspeed_opt_{int(time.time())}_{os.path.basename(self.script_path or 'script.py')}"
                        self.optimized_path = os.path.join(base_dir, opt_name)
                        with open(self.optimized_path, "w", encoding="utf-8") as f: f.write(self.optimized_source)
                        self.txt_opt.delete("1.0", "end"); self.txt_opt.insert("1.0", self.optimized_source)
                        self.log(f"Optimized file written to: {self.optimized_path}"); self.set_status("Optimization complete. Showing preview...")
                        show_diff_preview(result.original_func_source, result.modified_func_source, self.root)
                    else:
                        self.log("No functions were modified based on current mode and heuristics."); self.set_status("Analysis complete (no changes)")
                self.root.after(0, ui_update)
            except Exception as e: self.log(f"Optimization failed: {e}\n{traceback.format_exc()}"); self.set_status("Optimization error")
        threading.Thread(target=worker, daemon=True).start()

    def _run_script_task(self, path, name):
        if not path: return self.log(f"No {name} script available.")
        self.set_status(f"Running {name}..."); self.log(f"Running {name} script...")
        def worker():
            t, err = analyzer.run_script_time(path)
            if err: self.log(f"{name.capitalize()} run error: {err}"); self.set_status(f"{name.capitalize()} run failed"); return
            self.log(f"✅ {name.capitalize()} run time: {t:.6f}s"); self.set_status(f"{name.capitalize()} run complete")
        threading.Thread(target=worker, daemon=True).start()

    def compare_timings(self):
        if not self.script_path or not self.optimized_path: return self.log("Run Profile and Optimize first.")
        self.set_status("Comparing timings..."); self.log("Comparing (median of 3 runs)...")
        def worker():
            try:
                def median_time(path, name):
                    times = []
                    for i in range(3):
                        self.set_status(f"Running {name} (run {i+1}/3)..."); t, err = analyzer.run_script_time(path)
                        if err: return None, err
                        times.append(t)
                    return sorted(t for t in times if t is not None)[1], None
                orig_t, err0 = median_time(self.script_path, "original")
                if err0: return self.log("Original run error: " + err0), self.set_status("Compare failed")
                opt_t, err1 = median_time(self.optimized_path, "optimized")
                if err1: return self.log("Optimized run error: " + err1), self.set_status("Compare failed")
                speedup = orig_t / opt_t if opt_t > 0 else float('inf')
                self.log(f"--- Comparison ---"); self.log(f"Median Original:   {orig_t:.6f}s"); self.log(f"Median Optimized:  {opt_t:.6f}s"); self.log(f"Speedup:           {speedup:.2f}x"); self.set_status("Comparison complete")
            except Exception as e: self.log(f"Compare failed: {e}\n{traceback.format_exc()}"), self.set_status("Compare error")
        threading.Thread(target=worker, daemon=True).start()

    def show_about(self): messagebox.showinfo("About PySpeed", "PySpeed Accelerator v2.2\n\nA professional-grade, multi-mode Python optimization tool. Autor: LMLK-seal")

def main():
    root = ctk.CTk(); app = PySpeedApp(root); root.mainloop()

if __name__ == "__main__":
    main()
