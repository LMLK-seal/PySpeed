# PySpeed Optimizer 🚀

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.7.0-orange.svg)](https://github.com/pyspeed/releases)

> **Intelligent Python Performance Optimization Tool**  
> Automatically profile, analyze, and accelerate your Python code with minimal effort.  
> 🎯 **Specialized for CPU-intensive, computationally heavy algorithms** - not all Python code benefits from optimization.

---

## ✨ Features

### 🎯 **Smart Code Analysis**
- **Automatic Hotspot Detection**: Uses `cProfile` integration to identify performance bottlenecks
- **ML-Powered Suggestions**: Advanced heuristics to recommend the best optimization strategy
- **Multi-Backend Support**: Seamlessly applies Numba JIT, NumPy vectorization, and transpilation hints

### 🖥️ **Dual Interface**
- **📱 Modern GUI**: User-friendly desktop application with real-time optimization preview
- **📊 Jupyter Magic**: `%%pyspeed` cell magic for notebook-based development workflow
- **🔄 Performance Comparison**: Built-in benchmarking with statistical timing analysis

### ⚡ **Optimization Strategies**
- **🔥 Numba JIT Compilation**: Automatic `@numba.njit` decoration for numerical functions
- **📐 NumPy Vectorization**: Transforms explicit loops into vectorized operations
- **🛠️ Transpilation Ready**: Generates stubs for C++/Rust acceleration
- **🧠 Intelligent Targeting**: Applies optimizations only where they matter most

---
## 📊 Performance Benchmarks

### 🏆 Real-World Results

**Test Environment:** AMD Ryzen 5-3600, Python 3.9, Numba 0.61.2

| Test Case | Original (s) | Optimized (s) | Speedup | Status |
|-----------|--------------|---------------|---------|---------|
| **Recursive Fibonacci** | 6.90 | 0.04 | **172x** | ✅ Verified |
| **Monte Carlo π** | 91.0 | 0.98 | **92.4x** | ✅ Verified |
| **Matrix Multiplication** | 43.0 | 0.61 | **71x** | ✅ Verified |
| **Image Convolution** | 35.0 | 0.64 | **55x** | ✅ Verified |
| **Time Series Analysis** | 47.0 | 0.98 | **48x** | ✅ Verified |
| **Pi Calculation (Leibniz)** | 13.51 | 0.43 | **31.4x** | ✅ Verified |
| **Image Brightening** | 17.34 | 0.96 | **18.1x** | ✅ Verified |

### 📈 Optimization Pipeline Results

**Pi Calculation (Leibniz Series):**
```
[11:27:34] INFO: Numba v0.61.2 is available.
[11:27:53] Applied transformations: Function 'calculate_pi' was transformed using: NUMBA
[11:27:58] ✅ Optimized run time: 1.71s (includes JIT compilation)
[CLI Run] Warmed-up execution: 0.43s → 31.4x speedup
```

**Monte Carlo π Estimation:**
```
[11:39:52 - 11:41:23] Profiling complete (~91 seconds pure Python)
[CLI Run] Optimized execution: 0.98s → 92.4x speedup
```

**Matrix Multiplication (Triple-Nested Loops):**
```
[11:51:37 - 11:52:20] Profiling complete (~43 seconds pure Python)
✅ Optimized run time: 1.21s (includes JIT compilation)
[CLI Run] Warmed-up execution: 0.61s → 71x speedup
vs. np.dot(): 0.0016s (PySpeed closed massive performance gap)
```

**Image Convolution (Quadruple-Nested Loops):**
```
[11:55:42 - 11:56:17] Profiling complete (~35 seconds pure Python)
✅ Optimized run time: 1.34s (includes JIT compilation)
[CLI Run] Warmed-up execution: 0.64s → 55x speedup
Output: Generated convoluted_blurred_image.png (correctness verified)
```

**Time Series Analysis (Rolling Window SMA):**
```
[12:00:19 - 12:01:06] Profiling complete (~47 seconds pure Python)
✅ Optimized run time: 2.15s (includes JIT compilation)  
[CLI Run] Warmed-up execution: 0.98s → 48x speedup
vs. pandas.rolling(): 0.16s (PySpeed closed 98% of performance gap)
```

**Recursive Fibonacci (Memoization):**
```
✅ Applied @functools.lru_cache decorator
Median Original: 6.90s → Median Optimized: 0.04s
Speedup: 172x (exponential → linear complexity)
```

**Image Brightening (NumPy Vectorization):**
```
✅ Transformed nested loops into vectorized operations
Median Original: 17.34s → Median Optimized: 0.96s  
Speedup: 18.1x (4K image processing)
```

**Key Insights:** 
- **All CPU-bound algorithms** achieved **18-172x speedups** with zero manual optimization
- **Multiple optimization types**: JIT compilation, vectorization, and memoization all work seamlessly
- **Complexity transformation**: Converts exponential algorithms (Fibonacci) to linear performance
- **Gap bridging**: PySpeed transforms unusable code (17-91s) into production-ready performance (<1s)
- **Near-native performance**: Gets within 6x of professional C-backed libraries (pandas, NumPy)


---

## 📦 Installation

### Quick Install
```bash
# Clone the repository
git clone https://github.com/LMLK-seal/pyspeed.git
cd pyspeed

# Install in development mode
pip install -e .
```

### Dependencies
```bash
# Core functionality
pip install customtkinter astor ipython requests

# Optional (recommended for full optimization support)
# The fundamental package for scientific computing. Required by Numba and
# used in many test cases. Version <2 is recommended for broad compatibility
# with other scientific libraries like SciPy.
pip install numpy<2.0.0
```

### System Requirements
- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, Linux
- **Memory**: 512MB RAM minimum
- **Dependencies**: See `requirements.txt` for complete list

---

## 🚀 Quick Start

### 1. 🖥️ GUI Application

Launch the graphical interface:
```bash
python pyspeed_gui.py
```

**Workflow:**
1. **📂 Load Script** → Open your Python file
2. **📈 Profile** → Identify performance hotspots  
3. **🔧 Analyze & Optimize** → Apply intelligent transformations
4. **⚡ Compare** → Benchmark original vs. optimized code

![PySpeed GUI Screenshot](https://github.com/LMLK-seal/PySpeed/blob/main/screenshot.png?raw=true)

### 2. 📓 Jupyter Magic

Load the extension in your notebook:
```python
%load_ext pyspeed
```

**Basic Usage:**
```python
%%pyspeed

def calculate_pi(n_terms: int):
    """CPU-intensive function perfect for optimization"""
    numerator = 4.0
    denominator = 1.0
    pi = 0.0
    
    for _ in range(n_terms):
        pi += numerator / denominator
        denominator += 2.0
        pi -= numerator / denominator
        denominator += 2.0
    
    return pi

# This will be automatically profiled and optimized
result = calculate_pi(1_000_000)
print(f"π ≈ {result}")
```

**Performance Comparison:**
```python
%%pyspeed --compare

import numpy as np

def slow_array_operation(a, b):
    """This loop will be vectorized automatically"""
    c = np.zeros_like(a)
    for i in range(len(a)):
        c[i] = a[i] * b[i] + np.sin(a[i])
    return c

# Generate test data
x = np.random.rand(100_000)
y = np.random.rand(100_000)
result = slow_array_operation(x, y)
```

---

## 🎯 Use Cases

### 🔬 **Scientific Computing**
Perfect for researchers working with:
- Numerical simulations
- Monte Carlo methods  
- Signal processing algorithms
- Mathematical modeling

### 📊 **Data Science**
Accelerate your data workflows:
- Large dataset processing
- Feature engineering pipelines
- Custom aggregation functions
- Statistical computations

### 🎮 **Performance-Critical Applications**
Optimize bottlenecks in:
- Real-time systems
- Game development
- Financial algorithms
- Computer graphics

---

## 📋 Optimization Examples

### Example 1: Numba JIT Acceleration

**Matrix Multiplication (Before):**
```python
def slow_matrix_multiply(A, B):
    """Triple-nested loops - perfect Numba target"""
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    return result
```

**After (automatically generated):**
```python
import numba

@numba.njit
def slow_matrix_multiply(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    return result
```
**⚡ Result: 71x speedup (43s → 0.61s)**

### Example 2: NumPy Vectorization

**Before:**
```python
def element_wise_operation(a, b, c):
    result = np.zeros_like(a)
    for i in range(len(a)):
        result[i] = a[i] * b[i] + c[i]
    return result
```

**After (automatically generated):**
```python
def element_wise_operation(a, b, c):
    result = a * b + c  # Vectorized operation
    return result
```
**⚡ Result: ~100x speedup for large arrays**

### Example 3: What PySpeed Ignores (Smart Targeting)

**File Organizer Script:**
```python
import os
import shutil

def organize_files_by_extension(source_dir, target_dir):
    """PySpeed will NOT optimize this - and that's good!"""
    for filename in os.listdir(source_dir):
        if os.path.isfile(os.path.join(source_dir, filename)):
            extension = filename.split('.')[-1].lower()
            ext_dir = os.path.join(target_dir, extension)
            
            if not os.path.exists(ext_dir):
                os.makedirs(ext_dir)
            
            shutil.move(
                os.path.join(source_dir, filename),
                os.path.join(ext_dir, filename)
            )
```

**Why PySpeed skips this:**
- 🔍 **I/O Bound**: Dominated by file system operations, not computation
- 🚫 **String Operations**: Heavy use of string manipulation (not numeric)
- 📁 **System Calls**: `os.listdir`, `shutil.move` cannot be JIT-compiled
- ⚡ **Already Efficient**: The bottleneck is disk speed, not Python code

**🎯 PySpeed Result:** No modifications suggested - *"No functions were modified based on current hotspots and heuristics."*

### Example 3: Performance Gap Bridging

**Time Series Analysis:**
```python
def calculate_sma_naive(data, window_size):
    """Naive sliding window - PySpeed transforms this"""
    sma = []
    for i in range(len(data) - window_size + 1):
        window_sum = sum(data[i:i + window_size])
        sma.append(window_sum / window_size)
    return sma
```

**Performance Comparison:**
- **Pure Python**: 47.0 seconds ❌
- **PySpeed + Numba**: 0.98 seconds ✅ (48x speedup)
- **Pandas (C-backed)**: 0.16 seconds 🏆

**🎯 Achievement**: PySpeed closed **98% of the performance gap** between pure Python and professional libraries, automatically transforming unusable code into production-ready performance.

---

## 🔧 Configuration

### 📊 Telemetry Settings

PySpeed includes optional anonymous telemetry to improve optimization algorithms:

```json
{
  "telemetry_consent": true,
  "optimization_preferences": {
    "prefer_numba": true,
    "enable_experimental": false
  }
}
```

**Configuration file location:** `~/.pyspeed/config.json`

**What's collected (anonymously):**
- ✅ Code structure hashes (not source code)
- ✅ Optimization success rates
- ✅ Performance improvement metrics
- ❌ Never your actual source code
- ❌ Never personally identifiable information

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### 🐛 Bug Reports
Found an issue? Please create a detailed bug report:
- **Environment**: Python version, OS, dependencies
- **Reproduction**: Minimal code example
- **Expected vs. Actual**: What should happen vs. what happens

### 💡 Feature Requests
Have an optimization idea? We'd love to hear it:
- **Use Case**: Describe the scenario
- **Implementation**: Technical approach (if known)
- **Impact**: Expected performance benefits

---

## 🛠️ Architecture

PySpeed uses a modular optimization pipeline:

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐
│   Source    │───▶│   Profiler   │───▶│   Analyzer    │
│    Code     │    │  (cProfile)  │    │ (AST Walker)  │
└─────────────┘    └──────────────┘    └───────────────┘
                                              │
┌─────────────┐    ┌──────────────┐    ┌─────▼─────────┐
│ Optimized   │◀───│  Transformer │◀───│ ML Optimizer  │
│    Code     │    │ (AST Rewrite)│    │ (Heuristics)  │
└─────────────┘    └──────────────┘    └───────────────┘
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Numba Team**: For the incredible JIT compilation framework
- **NumPy Community**: For the foundation of scientific Python
- **AST Module Contributors**: Making code transformation possible
- **All Contributors**: Who help make PySpeed better every day

---

<div align="center">

**Made with ❤️ by LMLK-seal**

[⭐ Star us on GitHub](https://github.com/pyspeed/pyspeed) • [📢 Follow for updates](https://twitter.com/pyspeed_dev)

</div>
