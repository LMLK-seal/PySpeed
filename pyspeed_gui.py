# In pyspeed_project/pyspeed_gui.py

import os
import sys
import tempfile
import time
import traceback
import threading
import ast

# GUI toolkit
try:
    import customtkinter as ctk
    from tkinter import messagebox
except Exception as e:
    print("customtkinter is required. Install with: pip install customtkinter")
    raise

# Import the refactored non-GUI components from our package
from pyspeed import analyzer

def check_numba_compatibility():
    """Checks if Numba can be imported and is compatible with the current NumPy."""
    try:
        import numba
        return (True, f"Numba v{numba.__version__} is available.")
    except ImportError as e:
        return (False, f"Numba is not available or incompatible: {e}")

class PySpeedApp:
    def __init__(self, root):
        self.root = root
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")
        self.root.title("PySpeed — The Intelligent Python Accelerator")
        self.root.geometry("1200x800")
        self.script_path, self.script_source, self.optimized_source, self.optimized_path = None, "", None, None
        self.prof_entries = []
        
        self.numba_is_compatible, self.numba_status_msg = check_numba_compatibility()

        self._build_ui()
        self.log("Welcome to PySpeed! Open a script and click 'Profile' to begin.")
        if not self.numba_is_compatible:
            self.log(f"WARNING: {self.numba_status_msg}")
            self.log("WARNING: Numba decorators will not be applied until the environment issue is resolved.")
        else:
            self.log(f"INFO: {self.numba_status_msg}")

    def _build_ui(self):
        # Top frame is unchanged
        top_frame = ctk.CTkFrame(self.root); top_frame.pack(side="top", fill="x", padx=10, pady=10)
        ctk.CTkButton(top_frame, text="Open Script", command=self.open_script).pack(side="left", padx=5)
        ctk.CTkButton(top_frame, text="Profile", command=self.action_profile).pack(side="left", padx=5)
        ctk.CTkButton(top_frame, text="Analyze & Optimize", command=self.action_optimize).pack(side="left", padx=5)
        ctk.CTkButton(top_frame, text="Run Original", command=self.run_original).pack(side="left", padx=5)
        ctk.CTkButton(top_frame, text="Run Optimized", command=self.run_optimized).pack(side="left", padx=5)
        ctk.CTkButton(top_frame, text="Compare", command=self.compare_timings).pack(side="left", padx=5)
        ctk.CTkButton(top_frame, text="About", command=self.show_about).pack(side="right", padx=5)
        
        # Main content area is unchanged
        main_paned_window = ctk.CTkFrame(self.root, fg_color="transparent"); main_paned_window.pack(fill="both", expand=True, padx=10, pady=5)
        main_paned_window.grid_columnconfigure(0, weight=3); main_paned_window.grid_columnconfigure(1, weight=2); main_paned_window.grid_rowconfigure(0, weight=1)
        left_frame = ctk.CTkFrame(main_paned_window); left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5)); left_frame.grid_rowconfigure(1, weight=1); left_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(left_frame, text="Original Script Source").pack(anchor="w", padx=10, pady=(5,0))
        self.txt_source = ctk.CTkTextbox(left_frame, wrap="none", font=("monospace", 12)); self.txt_source.pack(fill="both", expand=True, padx=5, pady=5)
        right_frame = ctk.CTkFrame(main_paned_window); right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0)); right_frame.grid_rowconfigure(1, weight=1); right_frame.grid_columnconfigure(0, weight=1)
        tab_view = ctk.CTkTabview(right_frame); tab_view.pack(fill="both", expand=True, padx=5, pady=5)
        tab_profile = tab_view.add("Profile"); tab_optimize = tab_view.add("Optimized")
        ctk.CTkLabel(tab_profile, text="Hotspots (from cProfile)").pack(anchor="w", padx=5)
        self.txt_hotspots = ctk.CTkTextbox(tab_profile, wrap="none", font=("monospace", 11)); self.txt_hotspots.pack(fill="both", expand=True, padx=5, pady=5)
        ctk.CTkLabel(tab_optimize, text="Optimized Source Preview").pack(anchor="w", padx=5)
        self.txt_opt = ctk.CTkTextbox(tab_optimize, wrap="none", font=("monospace", 12)); self.txt_opt.pack(fill="both", expand=True, padx=5, pady=5)

        # --- MODIFIED: Bottom frame with Progress Bar ---
        bottom_frame = ctk.CTkFrame(self.root, height=165); bottom_frame.pack(side="bottom", fill="both", expand=False, padx=10, pady=(5, 10))
        bottom_frame.grid_propagate(False)
        bottom_frame.grid_columnconfigure(0, weight=1)
        # Configure rows: status, progressbar, log
        bottom_frame.grid_rowconfigure(2, weight=1) 

        self.lbl_status = ctk.CTkLabel(bottom_frame, text="Ready", anchor="w")
        self.lbl_status.grid(row=0, column=0, sticky="ew", padx=10, pady=(5,0))

        # Create the progress bar but don't show it yet
        self.progress_bar = ctk.CTkProgressBar(bottom_frame, mode='indeterminate')
        
        self.txt_log = ctk.CTkTextbox(bottom_frame, font=("monospace", 11))
        self.txt_log.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)

    def log(self, msg: str): self.root.after(0, lambda: self._log_threadsafe(msg))
    def _log_threadsafe(self, msg: str): ts = time.strftime("%H:%M:%S"); self.txt_log.insert("end", f"[{ts}] {msg}\n"); self.txt_log.see("end")
    def set_status(self, text: str): self.root.after(0, lambda: self._set_status_threadsafe(text))
    def _set_status_threadsafe(self, text: str): self.lbl_status.configure(text=text)
    
    def open_script(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(filetypes=[("Python files", "*.py")])
        if not path: return
        self.script_path = path
        with open(path, "r", encoding="utf-8") as f: self.script_source = f.read()
        self.txt_source.delete("1.0", "end"); self.txt_source.insert("1.0", self.script_source); self.log(f"Opened script: {path}"); self.set_status(f"Loaded {os.path.basename(path)}")
        self.txt_hotspots.delete("1.0", "end"); self.txt_opt.delete("1.0", "end"); self.optimized_source = None; self.optimized_path = None
    
    # --- MODIFIED: action_profile to control the progress bar ---
    def action_profile(self):
        if not self.script_path: return self.log("No script loaded.")
        
        # Show and start the progress bar on the main UI thread
        self.set_status("Profiling..."); self.log("Starting profiling...")
        self.progress_bar.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        self.progress_bar.start()

        def worker():
            prof_file, err = analyzer.run_profile_on_script(self.script_path, timeout=300) # Increased timeout for long scripts
            
            # This function will be scheduled to run on the main thread when the worker is done
            def ui_cleanup_and_update(error_message=None):
                self.progress_bar.stop()
                self.progress_bar.grid_remove()
                if error_message:
                    self.log(f"Profiling error: {error_message}")
                    self.set_status("Profiling failed")
                else:
                    self.prof_entries = analyzer.parse_profile(prof_file, top_n=20)
                    self.update_hotspots_ui()
                    self.log("Profiling complete.")
                    self.set_status("Profiling complete")
            
            self.root.after(0, ui_cleanup_and_update, err)

        # Start the background task
        threading.Thread(target=worker, daemon=True).start()

    def update_hotspots_ui(self):
        self.txt_hotspots.delete("1.0", "end")
        if not self.prof_entries: self.txt_hotspots.insert("1.0", "No hotspots found."); return
        header = f"{'Function':<30} {'File:Line':<40} {'Total Time (s)':<18} {'Cum. Time (s)':<18}\n"; separator = "-" * 106 + "\n"
        self.txt_hotspots.insert("end", header); self.txt_hotspots.insert("end", separator)
        for e in self.prof_entries:
            file_display = os.path.basename(e.file)
            if len(file_display) > 30: file_display = "..." + file_display[-27:]
            loc = f"{file_display}:{e.line}"
            line_text = f"{e.func_name:<30} {loc:<40} {e.tottime:<18.6f} {e.cumtime:<18.6f}\n"
            self.txt_hotspots.insert("end", line_text)
    
    def action_optimize(self):
        if not self.script_source: return self.log("No script loaded.")
        self.set_status("Analyzing..."); self.log("Starting advanced optimization pipeline...")
        def worker():
            try:
                hotspot_names = [e.func_name for e in self.prof_entries[:5]] if self.prof_entries else [n.name for n in ast.walk(ast.parse(self.script_source)) if isinstance(n, ast.FunctionDef)]
                result = analyzer.run_optimization_pipeline(self.script_source, hotspot_names)
                self.log("--- Analysis Report ---"); [self.log(f"[{s.severity.upper()}] {s.func_name} (L{s.line_no}): {s.message}") for s in result.suggestions]; self.log("-----------------------")
                if result.transformed_funcs:
                    self.log("Applied transformations:")
                    for func, opt_type in result.transformed_funcs.items(): self.log(f"  - Function '{func}' was transformed using: {opt_type.upper()}")
                    if not self.numba_is_compatible and 'numba' in result.needed_imports:
                        self.log("WARNING: Numba optimization was prepared but may fail to run due to environment incompatibility.")
                        self.log(f"  - Reason: {self.numba_status_msg}")
                    self.optimized_source = result.modified_source
                    base_dir = os.path.dirname(self.script_path) if self.script_path else tempfile.gettempdir()
                    opt_name = f"pyspeed_opt_{int(time.time())}_{os.path.basename(self.script_path or 'script.py')}"
                    self.optimized_path = os.path.join(base_dir, opt_name)
                    with open(self.optimized_path, "w", encoding="utf-8") as f: f.write(self.optimized_source)
                    self.root.after(0, lambda: self.txt_opt.delete("1.0", "end") or self.txt_opt.insert("1.0", self.optimized_source))
                    self.log(f"Optimized file written to: {self.optimized_path}"); self.set_status("Optimization complete")
                else: self.log("No functions were modified based on current hotspots and heuristics."); self.set_status("Analysis complete")
            except Exception as e: self.log(f"Optimization failed: {e}\n{traceback.format_exc()}"); self.set_status("Optimization error")
        threading.Thread(target=worker, daemon=True).start()

    def _run_script_task(self, path, name):
        if not path: return self.log(f"No {name} script available.")
        self.set_status(f"Running {name}..."); self.log(f"Running {name} script...")
        def worker():
            t, err = analyzer.run_script_time(path, timeout=300)
            if err: self.log(f"{name.capitalize()} run error: {err}"); self.set_status(f"{name.capitalize()} run failed"); return
            self.log(f"✅ {name.capitalize()} run time: {t:.6f}s"); self.set_status(f"{name.capitalize()} run complete")
        threading.Thread(target=worker, daemon=True).start()
    
    def run_original(self): self._run_script_task(self.script_path, "original")
    def run_optimized(self): self._run_script_task(self.optimized_path, "optimized")

    def compare_timings(self):
        if not self.script_path or not self.optimized_path: return self.log("Run Profile and Optimize first.")
        self.set_status("Comparing timings..."); self.log("Comparing (median of 3 runs)...")
        def worker():
            try:
                def median_time(path, name):
                    times = []
                    for i in range(3):
                        self.set_status(f"Running {name} (run {i+1}/3)...")
                        t, err = analyzer.run_script_time(path, timeout=300)
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

    def show_about(self): messagebox.showinfo("About PySpeed", "PySpeed Optimizer v1.0\n\nAn intelligent, multi-backend Python accelerator with a real-time profiler.")

def main():
    root = ctk.CTk(); app = PySpeedApp(root); root.mainloop()

if __name__ == "__main__":
    main()