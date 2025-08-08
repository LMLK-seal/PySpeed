# In pyspeed_project/pyspeed/analyzer.py

import os
import sys
import subprocess
import tempfile
import time
import ast
import io
import pstats
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

# ---------- AST & Optimization Engine ----------

@dataclass
class OptimizationSuggestion:
    """A suggestion produced by an optimizer."""
    func_name: str; line_no: int; message: str; severity: str = "info"

@dataclass
class OptimizationResult:
    """The result of running the full optimization pipeline."""
    original_source: str; modified_source: str
    suggestions: List[OptimizationSuggestion] = field(default_factory=list)
    transformed_funcs: Dict[str, str] = field(default_factory=dict) # e.g., {'my_func': 'numba'}
    needed_imports: set = field(default_factory=set)

class BaseOptimizer:
    """Base class for an optimization strategy."""
    def __init__(self, source_tree: ast.AST):
        self.tree = source_tree; self.suggestions = []
    def analyze(self) -> List[OptimizationSuggestion]: raise NotImplementedError
    def transform(self, candidates: List[str]) -> Tuple[ast.AST, List[str]]: return self.tree, []

class MockMLOptimizer(BaseOptimizer):
    """Simulates a trained ML model that provides high-level guidance."""
    KNOWLEDGE_BASE = {
        'calculate_pi': ('numba', 'High confidence: Matches known CPU-bound numeric kernels.'),
        'process_image': ('numpy', 'High confidence: Matches known array-processing tasks.'),
        'estimate_pi_monte_carlo': ('numba', 'High confidence: Matches known simulation/statistical patterns.'),
    }
    def analyze(self) -> List[OptimizationSuggestion]:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name in self.KNOWLEDGE_BASE:
                opt_type, message = self.KNOWLEDGE_BASE[node.name]
                self.suggestions.append(OptimizationSuggestion(
                    func_name=node.name, line_no=node.lineno,
                    message=f"[ML SUGGESTION] {message}", severity="recommendation"
                ))
        return self.suggestions
    def get_ml_targets(self) -> Dict[str, str]:
        return {s.func_name: self.KNOWLEDGE_BASE[s.func_name][0] for s in self.suggestions}

class NumbaOptimizer(BaseOptimizer):
    """Applies @numba.njit decorators with robust, expanded heuristics."""
    IGNORE_FUNCS = {'__init__', '__str__', '__repr__'}
    UNSUPPORTED_CALL_ATTRS = {'pack', 'grid', 'place', 'configure', 'mainloop', 'title', 'geometry', 'run', 'split', 'join', 'format', 'exists', 'isfile', 'read', 'write', 'append', 'pop', 'startswith'}
    UNSUPPORTED_CALL_FUNCS = {'open', 'print', 'input', 'subprocess', 'shutil', 'os.path.exists'}
    UNSUPPORTED_KEYWORDS = (ast.Try, ast.JoinedStr, ast.With, ast.Yield, ast.Global, ast.Nonlocal, ast.AsyncFor, ast.Await)

    # Expanded list of modules that signal numeric work
    NUMERIC_MODULES = {'math', 'np', 'numpy', 'random'}

    def is_disqualified(self, node: ast.FunctionDef) -> bool:
        if node.name in self.IGNORE_FUNCS: return True
        for sub_node in ast.walk(node):
            if isinstance(sub_node, self.UNSUPPORTED_KEYWORDS): return True
            if isinstance(sub_node, ast.Call):
                func = sub_node.func
                if isinstance(func, ast.Name) and func.id in self.UNSUPPORTED_CALL_FUNCS: return True
                if isinstance(func, ast.Attribute) and func.attr in self.UNSUPPORTED_CALL_ATTRS: return True
        return False

    def is_strong_numeric_candidate(self, node: ast.FunctionDef) -> bool:
        """
        Stage 2: Score functions based on an expanded set of strong numeric signals.
        """
        score = 0
        has_loop = False
        for n in ast.walk(node):
            if isinstance(n, ast.For):
                has_loop = True
            
            # Power operations (e.g., x**2) are a good signal.
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow):
                score += 2
            # Augmented assignments (+=, -= etc.) are common in numeric kernels.
            elif isinstance(n, ast.AugAssign):
                score += 1
            # Floating point numbers are a strong signal.
            elif isinstance(n, ast.Constant) and isinstance(n.value, float):
                score += 3
            # Array/list access (subscript) inside a loop is a very strong signal.
            elif has_loop and isinstance(n, ast.Subscript):
                score += 2
            # Calls to known numeric libraries are the strongest signal.
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                if hasattr(n.func.value, 'id') and n.func.value.id in self.NUMERIC_MODULES:
                    score += 5
        
        return score >= 5 and has_loop

    def analyze(self) -> List[OptimizationSuggestion]:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and not self.is_disqualified(node) and self.is_strong_numeric_candidate(node):
                self.suggestions.append(OptimizationSuggestion(
                    func_name=node.name,
                    line_no=node.lineno,
                    message="Strong candidate for Numba JIT due to heavy numeric computation.",
                    severity="recommendation"
                ))
        return self.suggestions
        
    def transform(self, candidates: List[str]) -> Tuple[ast.AST, List[str]]:
        transformed = []
        class DecoratorInserter(ast.NodeTransformer):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
                nonlocal transformed
                if node.name in candidates:
                    is_decorated = any((isinstance(d, ast.Attribute) and getattr(d, 'attr', '') in ('njit', 'jit')) for d in node.decorator_list)
                    if not is_decorated:
                        node.decorator_list.insert(0, ast.parse("numba.njit").body[0].value); transformed.append(node.name)
                return node
        new_tree = DecoratorInserter().visit(self.tree); ast.fix_missing_locations(new_tree); return new_tree, transformed

class NumpyVectorizeOptimizer(BaseOptimizer):
    """Analyzes and transforms simple loops into vectorized NumPy operations."""
    def analyze(self) -> List[OptimizationSuggestion]:
        return self.suggestions
    def transform(self, candidates: List[str]) -> Tuple[ast.AST, List[str]]:
        transformed_funcs = []
        class VectorizeRewriter(ast.NodeTransformer):
            def __init__(self):
                self.current_func_name = None; self.has_transformed = False
            def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
                if node.name in candidates:
                    self.current_func_name = node.name; self.has_transformed = False
                    self.generic_visit(node)
                    if self.has_transformed:
                        nonlocal transformed_funcs
                        if self.current_func_name not in transformed_funcs: transformed_funcs.append(self.current_func_name)
                    self.current_func_name = None
                return node
            def visit_For(self, node: ast.For) -> Optional[ast.AST]:
                if not self.current_func_name: return node
                try:
                    if not (isinstance(node.iter, ast.Call) and node.iter.func.id == 'range' and isinstance(node.iter.args[0], ast.Call) and node.iter.args[0].func.id == 'len'): return node
                    if not (len(node.body) == 1 and isinstance(node.body[0], ast.Assign)): return node
                    assign = node.body[0]
                    if not isinstance(assign.targets[0], ast.Subscript) or not isinstance(assign.value, ast.BinOp): return node
                    target_array, left_operand, right_operand = assign.targets[0].value.id, assign.value.left.value.id, assign.value.right.value.id
                    op_map = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/'}
                    if type(assign.value.op) not in op_map: return node
                    new_code_str = f"{target_array} = {left_operand} {op_map[type(assign.value.op)]} {right_operand}"
                    new_node = ast.parse(new_code_str).body[0]
                    self.has_transformed = True
                    return ast.fix_missing_locations(new_node)
                except Exception: return node
        new_tree = VectorizeRewriter().visit(self.tree)
        return new_tree, transformed_funcs

class TranspilerOptimizer(BaseOptimizer):
    """Stubs out complex functions for transpilation to C++/Rust."""
    def analyze(self) -> List[OptimizationSuggestion]: return self.suggestions
    def transform(self, candidates: List[str]) -> Tuple[ast.AST, List[str]]:
        transformed = []
        class TranspilerStubber(ast.NodeTransformer):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
                nonlocal transformed
                if node.name in candidates:
                    arg_names = [arg.arg for arg in node.args.args]
                    new_body_str = f"return my_cpp_extension.{node.name}({', '.join(arg_names)})"
                    comment = ast.Expr(value=ast.Constant(value="This function stubbed for C++/Rust transpilation."))
                    node.body = [ast.parse(new_body_str).body[0]]
                    ast.increment_lineno(node.body[0], node.lineno)
                    node.body.insert(0, comment)
                    transformed.append(node.name)
                return node
        new_tree = TranspilerStubber().visit(self.tree); ast.fix_missing_locations(new_tree); return new_tree, transformed

def ast_to_source(tree: ast.AST) -> str:
    """Converts an AST tree back to a string, trying astor then ast.unparse."""
    try:
        import astor
        return astor.to_source(tree)
    except ImportError:
        if hasattr(ast, "unparse"):
            return ast.unparse(tree)
    return "# Could not generate source. Please install 'astor' or use Python 3.9+."

def run_optimization_pipeline(source: str, hotspot_funcs: List[str]) -> OptimizationResult:
    """Runs a robust, multi-pass optimization pipeline."""
    try: tree = ast.parse(source)
    except Exception as e: return OptimizationResult(source, source, suggestions=[OptimizationSuggestion("parser", 0, f"Failed to parse script: {e}", "error")])

    all_suggestions = []; ml_optimizer = MockMLOptimizer(tree)
    all_suggestions.extend(ml_optimizer.analyze())
    for opt_cls in [NumbaOptimizer, NumpyVectorizeOptimizer, TranspilerOptimizer]: all_suggestions.extend(opt_cls(tree).analyze())
    
    current_tree = tree; transformed_funcs, needed_imports = {}, set(); already_transformed = set()
    transformation_priority = [('numba', NumbaOptimizer), ('numpy', NumpyVectorizeOptimizer), ('transpile', TranspilerOptimizer)]
    ml_targets = ml_optimizer.get_ml_targets()

    for opt_type, optimizer_cls in transformation_priority:
        candidates_this_pass = set()
        for func_name in hotspot_funcs:
            if func_name in already_transformed: continue
            if ml_targets.get(func_name) == opt_type or (not ml_targets.get(func_name) and any(s.func_name == func_name and opt_type.upper() in s.message.upper() for s in all_suggestions)):
                 candidates_this_pass.add(func_name)
        if not candidates_this_pass: continue
        
        optimizer = optimizer_cls(current_tree)
        current_tree, funcs_changed = optimizer.transform(list(candidates_this_pass))
        
        for func_name in funcs_changed:
            transformed_funcs[func_name] = opt_type; already_transformed.add(func_name)
            if opt_type == 'numba': needed_imports.add('numba')
            if opt_type == 'numpy': needed_imports.add('numpy as np')
            if opt_type == 'transpile': needed_imports.add('my_cpp_extension')

    modified_source = source
    if transformed_funcs:
        modified_source = ast_to_source(current_tree)
        import_str = "\n".join([f"import {imp}" for imp in sorted(list(needed_imports))])
        if import_str: modified_source = import_str + "\n\n" + modified_source

    return OptimizationResult(source, modified_source, all_suggestions, transformed_funcs, needed_imports)

@dataclass
class ProfileEntry:
    func_name: str; file: str; line: int; ncalls: int
    tottime: float; percall: float; cumtime: float; percall_cum: float

def run_profile_on_script(script_path: str, timeout: int = 120) -> Tuple[Optional[str], Optional[str]]:
    prof_file = os.path.join(tempfile.gettempdir(), f"pyspeed_{int(time.time())}.prof"); cmd = [sys.executable, "-m", "cProfile", "-o", prof_file, script_path]
    try: completed = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired: return None, "Profiling timed out"
    if completed.returncode != 0: return None, completed.stderr or completed.stdout
    return prof_file, None

def parse_profile(prof_file: str, top_n: int = 20) -> List[ProfileEntry]:
    stats = pstats.Stats(prof_file); stats.sort_stats('tottime'); entries = []
    sorted_keys = stats.fcn_list
    if not sorted_keys: return []
    for func_key in sorted_keys[:top_n]:
        data = stats.stats[func_key]; filename, lineno, func_name = func_key
        ncalls, nrecursive, tottime, cumtime, callers = data
        if filename == '~' or filename is None: filename = "built-in"; lineno = 0
        elif filename.startswith('~'): filename = os.path.expanduser(filename)
        entries.append(ProfileEntry(func_name=func_name, file=filename, line=lineno, ncalls=ncalls, tottime=tottime, cumtime=cumtime, percall=tottime / ncalls if ncalls > 0 else 0, percall_cum=cumtime / ncalls if ncalls > 0 else 0))
    return entries

def run_script_time(script_path: str, timeout: int = 120) -> Tuple[Optional[float], Optional[str]]:
    start = time.perf_counter();
    try: completed = subprocess.run([sys.executable, script_path], capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired: return None, "Execution timed out"
    end = time.perf_counter()
    if completed.returncode != 0: return None, completed.stderr or completed.stdout
    return end - start, None