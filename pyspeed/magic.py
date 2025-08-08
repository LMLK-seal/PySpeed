# In pyspeed_project/pyspeed/magic.py

import tempfile
import os
import time

from IPython.core.magic import register_cell_magic
from IPython.display import display, Markdown

from . import analyzer
from . import telemetry

def check_numba_compatibility():
    """Checks if Numba can be imported and is compatible with the current NumPy."""
    try:
        import numba
        return (True, f"Numba v{numba.__version__} is available.")
    except ImportError as e:
        return (False, f"Numba is not available or incompatible: {e}")

@register_cell_magic
def pyspeed(line, cell):
    """
    Profiles and suggests optimizations for the code in a Jupyter cell.
    """
    telemetry.ask_for_consent()
    
    # Check environment compatibility
    numba_is_compatible, numba_status_msg = check_numba_compatibility()
    if not numba_is_compatible:
        print(f"⚠️ PySpeed Warning: {numba_status_msg}")
        print("Numba decorators will not be applied, but analysis will proceed.")

    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.py', encoding='utf-8') as f:
        original_script_path = f.name
        f.write(cell)
        
    try:
        print("1. Profiling to find hotspots...")
        prof_file, err = analyzer.run_profile_on_script(original_script_path)
        if err:
            print(f"  -> Profiling failed: {err}")
            return

        prof_entries = analyzer.parse_profile(prof_file)
        hotspot_names = [e.func_name for e in prof_entries[:5]]
        
        if hotspot_names: print(f"  -> Top hotspots: {', '.join(hotspot_names)}")
        else: print("  -> No significant hotspots found during profiling.")
        
        print("\n2. Analyzing code for optimizations...")
        result = analyzer.run_optimization_pipeline(cell, hotspot_names)

        md_output = "### PySpeed Analysis Report\n\n"
        if result.suggestions:
            md_output += "| Severity | Function | Line | Suggestion |\n"
            md_output += "|----------|----------|------|------------|\n"
            for s in result.suggestions:
                md_output += f"| {s.severity} | `{s.func_name}` | {s.line_no} | {s.message} |\n"
        else: md_output += "No specific optimization suggestions found.\n"
        display(Markdown(md_output))
        
        if result.decorated_funcs:
            # *** MODIFIED LOGIC HERE ***
            if not numba_is_compatible:
                print("\n3. ⚠️ Numba decorator was NOT applied due to environment incompatibility.")
            else:
                print("\n3. Generated Optimized Code:")
                display(Markdown(f"```python\n{result.modified_source}\n```"))
            
            if "--compare" in line and numba_is_compatible:
                print("\n4. Running timing comparison (median of 3 runs)...")
                # ... the rest of the comparison logic is unchanged
                with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.py', encoding='utf-8') as opt_f:
                    optimized_script_path = opt_f.name
                    opt_f.write(result.modified_source)
                try:
                    def median_time(path):
                        times = [analyzer.run_script_time(path)[0] for _ in range(3)]
                        return sorted(t for t in times if t is not None)[1] if times else None
                    orig_t = median_time(original_script_path)
                    opt_t = median_time(optimized_script_path)
                    if orig_t and opt_t:
                        speedup = (orig_t / opt_t) if opt_t > 0 else float('inf')
                        timing_data = {'original_s': orig_t, 'optimized_s': opt_t, 'speedup_x': speedup}
                        comp_md = "### Performance Comparison\n\n"
                        comp_md += f"- **Original:** `{orig_t:.6f}` seconds\n"
                        comp_md += f"- **Optimized:** `{opt_t:.6f}` seconds\n"
                        comp_md += f"- **Speedup:** `{speedup:.2f}x`"
                        display(Markdown(comp_md))
                        payload = telemetry.build_payload(cell, result, timing_data)
                        telemetry.start_telemetry_upload_thread(payload)
                finally:
                    os.remove(optimized_script_path)
            elif "--compare" in line:
                 print("\n4. Skipping comparison because optimized code could not be generated.")

        else:
            print("\n3. No code modifications were applied.")
    finally:
        os.remove(original_script_path)

def load_ipython_extension(ipython):
    ipython.register_magic_function(pyspeed, 'cell')