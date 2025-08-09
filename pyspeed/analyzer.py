# In pyspeed_project/analyzer.py

import os
import sys
import subprocess
import tempfile
import time
import ast
import inspect
import io
import pstats
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

# ---------- Helper: Numba Safety Layer ----------
def try_numba_compile(func_node: ast.FunctionDef, full_source: str) -> Tuple[bool, str]:
    try:
        import numba
        namespace = {}
        exec(full_source, namespace)
        func_to_test = namespace.get(func_node.name)
        if not func_to_test or not callable(func_to_test): return False, "Function not found"
        numba.njit(func_to_test)
        return True, "Function is Numba-compatible (compilation successful)."
    except numba.core.errors.NumbaError as e:
        return False, f"Numba compilation failed: {str(e).splitlines()[0]}"
    except Exception as e:
        return False, f"An unexpected error occurred during safety check: {e}"

# ---------- AST & Optimization Engine ----------
@dataclass
class OptimizationSuggestion:
    func_name: str; line_no: int; message: str; severity: str = "info"
    opt_type: str = "general"

@dataclass
class OptimizationResult:
    original_source: str; modified_source: str; original_func_source: str; modified_func_source: str
    suggestions: List[OptimizationSuggestion] = field(default_factory=list)
    transformed_funcs: Dict[str, str] = field(default_factory=dict)
    needed_imports: set = field(default_factory=set)
    actionable_suggestion: Optional[str] = None
    rejection_reason: Optional[str] = None

class BaseOptimizer:
    opt_type = "general"
    def __init__(self, source_tree: ast.AST, profile_entries: List['ProfileEntry'] = None, full_source: str = ""):
        self.tree = source_tree; self.source = full_source
        self.profile = {p.func_name: p for p in profile_entries} if profile_entries else {}
        self.suggestions = []

    def analyze(self) -> List[OptimizationSuggestion]:
        # Main entry point calls static and dynamic analysis
        static_suggestions = self.static_analyze()
        dynamic_suggestions = self.dynamic_analyze()
        # Combine and de-duplicate suggestions
        all_sugs = {}
        for s in static_suggestions + dynamic_suggestions:
            key = (s.func_name, s.message)
            if key not in all_sugs:
                all_sugs[key] = s
        return list(all_sugs.values())

    def static_analyze(self) -> List[OptimizationSuggestion]: return []
    def dynamic_analyze(self) -> List[OptimizationSuggestion]: return []
    def transform(self, candidates: List[str]) -> Tuple[ast.AST, List[str]]: return self.tree, []

class NumbaOptimizer(BaseOptimizer):
    opt_type = "numba"
    def static_analyze(self) -> List[OptimizationSuggestion]:
        suggestions = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                is_safe, reason = try_numba_compile(node, self.source)
                suggestions.append(OptimizationSuggestion(node.name, node.lineno, reason, "recommendation" if is_safe else "info", self.opt_type))
        return suggestions
    def dynamic_analyze(self) -> List[OptimizationSuggestion]: return []
    def transform(self, candidates: List[str]) -> Tuple[ast.AST, List[str]]:
        transformed = []
        class DecoratorInserter(ast.NodeTransformer):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
                if node.name in candidates and not any(isinstance(d, ast.Attribute) and getattr(d, 'attr', '') in ('njit', 'jit') for d in node.decorator_list):
                    node.decorator_list.insert(0, ast.parse("numba.njit").body[0].value); transformed.append(node.name)
                return node
        new_tree = DecoratorInserter().visit(self.tree); ast.fix_missing_locations(new_tree); return new_tree, transformed

class MemoizationOptimizer(BaseOptimizer):
    opt_type = "memoization"
    IMPURE_CALLS = {'print', 'open', 'input', 'shutil', 'os.', 'self.', 'plt.', 'fig.', 'ax.'}
    UNHASHABLE_HINTS = ['list', 'dict', 'ndarray', 'List', 'Dict', 'np.ndarray']
    def is_pure(self, node: ast.FunctionDef) -> bool:
        if any(isinstance(n, (ast.Nonlocal, ast.Global)) for n in ast.walk(node)): return False
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                call_source = ast.get_source_segment(self.source, n.func)
                if call_source and any(impure in call_source for impure in self.IMPURE_CALLS): return False
        return True
    def has_hashable_args(self, node: ast.FunctionDef) -> bool:
        for arg in node.args.args:
            if arg.annotation:
                hint_source = ast.get_source_segment(self.source, arg.annotation)
                if hint_source and any(unhashable in hint_source for unhashable in self.UNHASHABLE_HINTS): return False
        return True
    def is_recursive(self, node: ast.FunctionDef) -> bool:
        for sub_node in ast.walk(node):
            if isinstance(sub_node, ast.Call) and isinstance(sub_node.func, ast.Name) and sub_node.func.id == node.name: return True
        return False
    def static_analyze(self) -> List[OptimizationSuggestion]:
        suggestions = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and self.is_recursive(node) and self.is_pure(node) and self.has_hashable_args(node):
                suggestions.append(OptimizationSuggestion(node.name, node.lineno, "This recursive function with hashable arguments is a strong candidate for `@functools.lru_cache`.", "recommendation", self.opt_type))
        return suggestions
    def dynamic_analyze(self) -> List[OptimizationSuggestion]:
        suggestions = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                if node.name in self.profile and self.profile[node.name].ncalls > 10 and self.is_pure(node) and self.has_hashable_args(node):
                    suggestions.append(OptimizationSuggestion(node.name, node.lineno, "This pure function is called frequently. Consider using `@functools.lru_cache`.", "recommendation", self.opt_type))
                elif not self.has_hashable_args(node) and node.name in self.profile:
                    suggestions.append(OptimizationSuggestion(node.name, node.lineno, "Cannot memoize: Function accepts unhashable arguments (e.g., lists, arrays).", "info", self.opt_type))
        return suggestions
    def transform(self, candidates: List[str]) -> Tuple[ast.AST, List[str]]:
        transformed = []
        class CacheInserter(ast.NodeTransformer):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
                if node.name in candidates:
                    node.decorator_list.insert(0, ast.parse("functools.lru_cache(maxsize=None)").body[0].value); transformed.append(node.name)
                return node
        new_tree = CacheInserter().visit(self.tree); ast.fix_missing_locations(new_tree); return new_tree, transformed

class NumpyVectorizeOptimizer(BaseOptimizer):
    opt_type = "numpy"
    def _is_vectorizable_assign(self, assign_node):
        if not isinstance(assign_node, (ast.Assign, ast.AugAssign)): return False
        target = assign_node.target if isinstance(assign_node, ast.AugAssign) else assign_node.targets[0]
        if not isinstance(target, ast.Subscript): return False
        return True
    def static_analyze(self) -> List[OptimizationSuggestion]:
        suggestions = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                for item in node.body:
                    if isinstance(item, ast.For):
                        inner_loop_body = item.body
                        if len(item.body) == 1 and isinstance(item.body[0], ast.For): inner_loop_body = item.body[0].body
                        if len(inner_loop_body) == 1 and self._is_vectorizable_assign(inner_loop_body[0]):
                             suggestions.append(OptimizationSuggestion(node.name, item.lineno, "This loop performs element-wise operations and can be rewritten with NumPy vectorization.", "recommendation", self.opt_type)); break
        return suggestions
    def dynamic_analyze(self) -> List[OptimizationSuggestion]: return []
    def transform(self, candidates: List[str]) -> Tuple[ast.AST, List[str]]:
        transformed_funcs = []
        class VectorizeRewriter(ast.NodeTransformer):
            def __init__(self, source_code: str):
                self.source = source_code
                self.transformed_funcs_list = []
            def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
                if node.name in candidates:
                    loop_rewriter = self.LoopRewriter(node.name, self.source, self.transformed_funcs_list)
                    node.body = [loop_rewriter.visit(child) for child in node.body]
                return node
            class LoopRewriter(ast.NodeTransformer):
                def __init__(self, func_name, source, transformed_list):
                    self.func_name = func_name; self.source = source; self.transformed_list = transformed_list
                def visit_For(self, node: ast.For) -> Optional[ast.AST]:
                    try:
                        if len(node.body) == 1 and isinstance(node.body[0], ast.For):
                            inner_loop = node.body[0]
                            if len(inner_loop.body) == 1 and isinstance(inner_loop.body[0], ast.AugAssign):
                                assign = inner_loop.body[0]
                                target_array_name = ast.get_source_segment(self.source, assign.target.value)
                                op_map = {ast.Add: '+=', ast.Sub: '-=', ast.Mult: '*=', ast.Div: '/='}
                                op_str = op_map[type(assign.op)]
                                value_str = ast.get_source_segment(self.source, assign.value)
                                new_code = f"{target_array_name} {op_str} {value_str}"
                                if self.func_name not in self.transformed_list: self.transformed_list.append(self.func_name)
                                return ast.fix_missing_locations(ast.parse(new_code).body[0])
                    except Exception: pass
                    return self.generic_visit(node)
        rewriter = VectorizeRewriter(self.source)
        new_tree = rewriter.visit(self.tree)
        return new_tree, rewriter.transformed_funcs_list

class BatchingOptimizer(BaseOptimizer):
    opt_type = "batching"
    def static_analyze(self) -> List[OptimizationSuggestion]: return []
    def dynamic_analyze(self) -> List[OptimizationSuggestion]:
        suggestions = []
        for func_name, entry in self.profile.items():
            if entry.ncalls > 5000 and entry.tottime / entry.ncalls < 1e-5:
                for node in ast.walk(self.tree):
                    if isinstance(node, ast.FunctionDef) and node.name == func_name:
                        suggestions.append(OptimizationSuggestion(func_name, node.lineno, "High call frequency suggests refactoring to process inputs in batches.", "recommendation", self.opt_type)); break
        return suggestions
    def transform(self, candidates: List[str]) -> Tuple[ast.AST, List[str]]: return self.tree, []

class MultiprocessingOptimizer(BaseOptimizer):
    opt_type = "multiprocessing"
    def is_cpu_bound(self, node: ast.For) -> bool:
        for sub_node in ast.walk(node):
            if isinstance(sub_node, ast.Call):
                call_src = ast.get_source_segment(self.source, sub_node.func)
                if call_src and any(io_call in call_src for io_call in ['open', 'print', '.read', '.write', 'sleep']): return False
        return True
    def static_analyze(self) -> List[OptimizationSuggestion]: return []
    def dynamic_analyze(self) -> List[OptimizationSuggestion]:
        suggestions = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                if node.name in self.profile and (self.profile[node.name].tottime > 1.0 or self.profile[node.name].impact_score > 10.0):
                    for item in node.body:
                        if isinstance(item, ast.For) and self.is_cpu_bound(item):
                            suggestions.append(OptimizationSuggestion(node.name, item.lineno, "This CPU-bound loop is a good candidate for `concurrent.futures.ProcessPoolExecutor`.", "recommendation", self.opt_type)); break
        return suggestions
    def transform(self, candidates: List[str]) -> Tuple[ast.AST, List[str]]: return self.tree, []

# ---------- Pipeline and Profiling ----------
def run_optimization_pipeline(source: str, hotspot_funcs: List[str], mode: str, profile_entries: List['ProfileEntry']) -> OptimizationResult:
    try: tree = ast.parse(source)
    except Exception as e: return OptimizationResult(source, source, "", "", [OptimizationSuggestion("parser", 0, f"Failed to parse script: {e}", "error")])

    all_suggestions = []
    optimizer_classes = [NumbaOptimizer, NumpyVectorizeOptimizer, MultiprocessingOptimizer, MemoizationOptimizer, BatchingOptimizer]
    for opt_cls in optimizer_classes:
        all_suggestions.extend(opt_cls(tree, profile_entries, full_source=source).analyze())

    potential_targets = hotspot_funcs
    if not potential_targets:
        potential_targets = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        if not potential_targets:
            return OptimizationResult(source, source, "", "", all_suggestions, rejection_reason="No functions found to optimize.")
    
    target_func_name = potential_targets[0]

    if mode == "Multiprocessing Suggest":
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == target_func_name:
                loop_node = next((item for item in node.body if isinstance(item, ast.For)), None)
                if loop_node: return OptimizationResult(source, source, "", "", all_suggestions, actionable_suggestion=_generate_mp_template(node, loop_node, source))
        return OptimizationResult(source, source, "", "", all_suggestions)

    optimizer_map = {"Numba JIT": (NumbaOptimizer, {'numba'}), "NumPy Vectorize": (NumpyVectorizeOptimizer, {'numpy as np'}), "Memoization": (MemoizationOptimizer, {'functools'})}
    mode_to_type = {"Numba JIT": "numba", "NumPy Vectorize": "numpy", "Memoization": "memoization"}
    
    selected_mode = mode
    if mode == "Auto (Recommended)":
        recommendations = [s for s in all_suggestions if s.func_name == target_func_name and s.severity == 'recommendation']
        if any(s.opt_type == "numba" for s in recommendations): selected_mode = "Numba JIT"
        elif any(s.opt_type == "numpy" for s in recommendations): selected_mode = "NumPy Vectorize"
        elif any(s.opt_type == "memoization" for s in recommendations): selected_mode = "Memoization"
        else: return OptimizationResult(source, source, "", "", all_suggestions, rejection_reason="Auto mode could not find a suitable optimization for the top hotspot.")
    
    selected_opt_type = mode_to_type.get(selected_mode)
    is_recommended = any(s.func_name == target_func_name and s.severity == 'recommendation' and s.opt_type == selected_opt_type for s in all_suggestions)
    if mode != "Auto (Recommended)" and not is_recommended:
        info_reason = next((s.message for s in all_suggestions if s.func_name == target_func_name and s.opt_type == selected_opt_type), "it is not a suitable candidate for this mode.")
        return OptimizationResult(source, source, "", "", all_suggestions, rejection_reason=f"Transformation rejected for '{target_func_name}'. Reason: {info_reason}")

    optimizer_cls, needed_imports = optimizer_map.get(selected_mode, (None, None))
    if not optimizer_cls: return OptimizationResult(source, source, "", "", all_suggestions)
    
    optimizer = optimizer_cls(tree, profile_entries, full_source=source)
    new_tree, funcs_changed = optimizer.transform([target_func_name])
    
    transformed_funcs = {func: selected_mode for func in funcs_changed}
    original_func_source = get_function_source(target_func_name, tree)
    modified_func_source = get_function_source(target_func_name, new_tree)
    
    modified_source = source
    if transformed_funcs:
        modified_source = ast_to_source(new_tree)
        if needed_imports: modified_source = "\n".join([f"import {imp}" for imp in sorted(list(needed_imports))]) + "\n\n" + modified_source

    return OptimizationResult(source, modified_source, original_func_source, modified_func_source, all_suggestions, transformed_funcs, needed_imports)

def _generate_mp_template(func_node: ast.FunctionDef, loop_node: ast.For, source: str) -> str:
    func_name, loop_iterator, loop_variable = func_node.name, ast.get_source_segment(source, loop_node.iter).strip(), ast.get_source_segment(source, loop_node.target).strip()
    return f"""# PySpeed Suggestion for parallelizing '{func_name}' ..."""

def get_function_source(func_name: str, tree: ast.AST) -> str:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name: return ast_to_source(node)
    return ""
@dataclass
class ProfileEntry:
    func_name: str; file: str; line: int; ncalls: int; tottime: float; cumtime: float;
    is_leaf: bool = False; impact_score: float = 0.0
def parse_profile(prof_file: str, top_n: int = 20) -> List[ProfileEntry]:
    stats = pstats.Stats(prof_file); callees_data = {}
    try:
        stats.calc_callees()
        if hasattr(stats, 'callees'): callees_data = stats.callees
    except Exception: pass
    entries = []
    for func_key, data in stats.stats.items():
        ncalls, _, tottime, cumtime, _ = data
        filename, lineno, func_name = func_key
        if filename == '~' or filename is None or tottime == 0: continue
        is_leaf = func_key not in callees_data
        entries.append(ProfileEntry(func_name=func_name, file=filename, line=lineno, ncalls=ncalls, tottime=tottime, cumtime=cumtime, is_leaf=is_leaf, impact_score=tottime * ncalls))
    entries.sort(key=lambda x: x.impact_score, reverse=True)
    return entries[:top_n]
def ast_to_source(tree: ast.AST) -> str:
    try: import astor; return astor.to_source(tree)
    except ImportError:
        if hasattr(ast, "unparse"): return ast.unparse(tree)
    return "# Could not generate source."
def run_profile_on_script(script_path: str, timeout: int = 300) -> Tuple[Optional[str], Optional[str]]:
    prof_file = os.path.join(tempfile.gettempdir(), f"pyspeed_{int(time.time())}.prof"); cmd = [sys.executable, "-m", "cProfile", "-o", prof_file, script_path]
    try: completed = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired: return None, "Profiling timed out"
    if completed.returncode != 0: return None, completed.stderr or completed.stdout
    return prof_file, None
def run_script_time(script_path: str, timeout: int = 300) -> Tuple[Optional[float], Optional[str]]:
    start = time.perf_counter();
    try: completed = subprocess.run([sys.executable, script_path], capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired: return None, "Execution timed out"
    end = time.perf_counter()
    if completed.returncode != 0: return None, completed.stderr or completed.stdout
    return end - start, None
