import sys
import ast
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
import re
import traceback


@dataclass
class PythonCall:
    """Represents a single execution unit (node) in the traced program.

    This is the core data structure that builds up the execution tree.
    Each node can represent a function call, a statement, a loop, etc.
    """

    code: str  # The original code statement
    nonexpanded: List[str] = field(default_factory=list)  # Original version
    expanded: List[str] = field(default_factory=list)  # Flattened children's code
    contains_llm_call: bool = False
    children: List["PythonCall"] = field(default_factory=list)
    function_ref: Optional[Callable] = None

    # Enhanced multi-line statement support
    line_number_start: Optional[int] = None
    line_number_end: Optional[int] = None
    line_number: Optional[int] = (
        None  # Keep for backward compatibility - maps to line_number_start
    )

    parent: Optional["PythonCall"] = None
    is_completed: bool = False
    is_function_call: bool = False  # Track if this is a function call
    function_name: Optional[str] = None  # Name of function being called

    # New fields for pending function call tracking
    original_code: str = ""  # Store original before any substitution
    pending_calls: Set[str] = field(
        default_factory=set
    )  # function names we're waiting for
    substitutions_made: Dict[str, str] = field(
        default_factory=dict
    )  # func_name -> return_var
    is_ready_for_execution: bool = True  # False if waiting for function calls

    # Loop-related attributes
    loop_var: Optional[str] = None
    loop_collection: Optional[str] = None
    loop_idx_var: Optional[str] = None
    loop_namespaced_var: Optional[str] = None

    # Loop scope attributes
    is_loop_scope: bool = False
    loop_line: Optional[int] = None
    loop_iteration_count: Dict[int, int] = field(default_factory=dict)
    loop_var_mapping: Dict[str, str] = field(
        default_factory=dict
    )  # Maps original var to namespaced var

    # Execution block attributes
    execution_type: str = "statement"  # "statement", "function_block", "expression"

    def add_child(self, child: "PythonCall") -> None:
        """Add a child node and set parent relationship."""
        self.children.append(child)
        child.parent = self

    def propagate_llm_flag(self) -> None:
        """Propagate LLM flag up the tree to all ancestors."""
        current = self
        path = []
        while current is not None:
            if not current.contains_llm_call:  # Only log when setting for first time
                path.append(current.code[:30] + "...")
            current.contains_llm_call = True
            current = current.parent
        if path:
            print(f"[TRACER]   🔥 LLM flag propagated through: {' ← '.join(path)}")

    def is_leaf(self) -> bool:
        """Check if this node is a leaf (no children)."""
        return len(self.children) == 0

    def get_final_code(self) -> List[str]:
        """Get the final code representation based on expansion rules.

        CORRECTED LOGIC:
        - Function definitions with LLM calls should NOT appear - they get inlined
        - Function calls should be converted to assignments using return variables
        - Parameters must be substituted in function bodies
        - Only non-LLM functions appear as definitions
        - Loop scopes should return their expanded code
        """
        # Special handling for loop scopes
        if self.is_loop_scope:
            # Loop scopes always return their expanded code
            return self.expanded

        # Special handling for function call nodes
        if self.is_function_call and self.contains_llm_call:
            # This is a function call to a function that contains LLM calls
            # Return the expanded code (inlined function body)
            return self.expanded

        if self.contains_llm_call and self.children:
            # This node contains LLM calls - return ONLY the inlined body, NO function definition
            return self.expanded
        else:
            # No LLM calls - return original code
            return self.nonexpanded

    def finalize(self) -> None:
        """Clean finalize with no special case hacks."""
        if self.is_completed:
            return

        print(
            f"[TRACER]  Finalizing node: {self.code[:30]}... (children: {len(self.children)})"
        )

        # First, ensure all children are finalized
        for child in self.children:
            if not child.is_completed:
                child.finalize()

        # Build expanded code from children
        if self.children:
            for child in self.children:
                # Get the appropriate code from each child
                child_code = child.get_final_code()

                print(
                    f"[TRACER]    Child {child.code[:20]}... contributes: {child_code}"
                )

                # Apply variable substitution if we're in a loop scope
                if self.is_loop_scope and hasattr(self, "loop_var") and self.loop_var:
                    # Create substitution mapping
                    namespaced_var = f"loop_line_{self.loop_line}_{self.loop_var}"
                    substituted_code = []
                    for line in child_code:
                        # Apply substitution with word boundaries to avoid partial matches
                        import re

                        pattern = r"\b" + re.escape(self.loop_var) + r"\b"
                        new_line = re.sub(pattern, namespaced_var, line)
                        substituted_code.append(new_line)
                    self.expanded.extend(substituted_code)
                else:
                    # For function calls, the substitution already happened proactively
                    self.expanded.extend(child_code)

        # REMOVED: All the assignment detection garbage!
        # The iterative substitution handles everything elegantly

        print(f"[TRACER]    Final expanded: {self.expanded}")
        self.is_completed = True


@dataclass
class ExpressionNode(PythonCall):
    """Enhanced node for expressions with nested function calls."""

    original_expression: str = ""
    current_expression: str = ""
    final_expression: str = ""
    execution_schedule: List[Dict] = field(default_factory=list)
    completed_executions: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.execution_type = "expression"

    def get_final_code(self) -> List[str]:
        """For expressions, return expanded code if LLM calls, otherwise original code."""
        if self.contains_llm_call:
            # Contains LLM calls - return the properly built expanded content
            return self.expanded if self.expanded else self.nonexpanded
        else:
            # No LLM calls - return original code unchanged
            return [self.code.strip()]

    def add_execution_block(
        self, func_name: str, execution_block: "PythonCall"
    ) -> None:
        """Add an execution block for a function."""
        execution_block.execution_type = "function_block"
        execution_block.function_name = func_name
        self.add_child(execution_block)
        self.completed_executions.append(func_name)

    def update_expression_after_completion(
        self, completed_func: str, return_var: str
    ) -> None:
        """Update expression state after a function completes using AST."""
        try:
            # Clean up whitespace and indentation for parsing
            current_expr = self.current_expression.strip()

            # Remove any leading whitespace and handle multi-line expressions
            lines = current_expr.split("\n")
            cleaned_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped:
                    cleaned_lines.append(stripped)

            # Join back into single line if possible
            if len(cleaned_lines) == 1:
                clean_expr = cleaned_lines[0]
            else:
                # For multi-line, try to join or take first meaningful line
                clean_expr = " ".join(cleaned_lines)

            print(f"[EXPRESSION]  Parsing clean expression: '{clean_expr}'")

            # Parse current expression
            tree = ast.parse(clean_expr)

            # Create substitutor for only the completed function
            substitutor = SingleFunctionSubstitutor(completed_func, return_var)
            new_tree = substitutor.visit(tree)

            # Update current expression
            old_expression = self.current_expression
            self.current_expression = ast.unparse(new_tree)

            # Remove from pending
            self.pending_calls.discard(completed_func)
            self.substitutions_made[completed_func] = return_var

            print(
                f"[EXPRESSION] Updated after {completed_func}: {old_expression.strip()} -> {self.current_expression}"
            )

        except Exception as e:
            print(f"[EXPRESSION] Error updating expression: {e}")
            print(f"[EXPRESSION] Problem expression: '{self.current_expression}'")
            # Fallback: mark as ready
            self.pending_calls.discard(completed_func)

    def is_expression_complete(self) -> bool:
        """Check if all scheduled executions are complete."""
        return len(self.completed_executions) == len(self.execution_schedule)

    def finalize(self) -> None:
        """Custom finalize for expression nodes - build complete expanded content."""
        if self.is_completed:
            return

        print(f"[TRACER]  Finalizing expression node: {self.code[:30]}...")

        # First, finalize execution block children (function bodies)
        for child in self.children:
            if (
                not child.is_completed
                and hasattr(child, "execution_type")
                and child.execution_type == "function_block"
            ):
                child.finalize()
                print(f"[TRACER]    Finalized execution block: {child.code}")

        # Build complete expanded content: execution blocks + final expression
        self.expanded = []

        # Add all execution block content (function bodies)
        for child in self.children:
            if (
                hasattr(child, "execution_type")
                and child.execution_type == "function_block"
            ):
                child_code = child.get_final_code()
                self.expanded.extend(child_code)
                print(f"[TRACER]    Added execution block content: {child_code}")

        # Add the final transformed expression (if not a simple variable name)
        if hasattr(self, "current_expression") and self.current_expression:
            final_expr = self.current_expression.strip()
            if final_expr and final_expr not in self.expanded:
                # Only add if it's an assignment or multi-word expression (not just a variable name)
                if "=" in final_expr or len(final_expr.split()) > 1:
                    self.expanded.append(final_expr)
                    print(f"[TRACER]    Added final expression: {final_expr}")
                else:
                    print(f"[TRACER]    Skipping simple variable result: {final_expr}")

        # Mark as completed
        self.is_completed = True
        print("[TRACER]  Expression finalization complete")


class FunctionCallExtractor(ast.NodeVisitor):
    """Extract all function calls with their AST nodes and argument details."""

    def __init__(self):
        self.function_calls = []

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            # Skip built-ins
            if func_name not in [
                "print",
                "len",
                "str",
                "int",
                "float",
                "bool",
                "list",
                "dict",
                "set",
                "tuple",
                "range",
            ]:
                self.function_calls.append(
                    {
                        "name": func_name,
                        "call_ast": node,
                        "arguments": [ast.unparse(arg) for arg in node.args],
                        "call_text": ast.unparse(node),
                    }
                )
        elif isinstance(node.func, ast.Attribute):
            # Method calls like obj.method()
            method_name = node.func.attr
            self.function_calls.append(
                {
                    "name": method_name,
                    "call_ast": node,
                    "arguments": [ast.unparse(arg) for arg in node.args],
                    "call_text": ast.unparse(node),
                    "is_method": True,
                    "object": ast.unparse(node.func.value),
                }
            )

        self.generic_visit(node)


class FunctionCallSubstitutor(ast.NodeTransformer):
    """Substitute function calls with return variables using AST comparison."""

    def __init__(self, substitution_map: Dict[ast.Call, str]):
        self.substitution_map = substitution_map

    def visit_Call(self, node: ast.Call):
        # Check if this call node should be substituted
        for call_ast, return_var in self.substitution_map.items():
            if self._ast_nodes_equal(node, call_ast):
                # Replace with return variable
                return ast.Name(id=return_var, ctx=ast.Load())

        # Continue traversing if no substitution
        return self.generic_visit(node)

    def _ast_nodes_equal(self, node1: ast.Call, node2: ast.Call) -> bool:
        """Compare two AST Call nodes for equality."""
        # Compare function names
        if type(node1.func) != type(node2.func):
            return False

        if isinstance(node1.func, ast.Name) and isinstance(node2.func, ast.Name):
            if node1.func.id != node2.func.id:
                return False
        elif isinstance(node1.func, ast.Attribute) and isinstance(
            node2.func, ast.Attribute
        ):
            if ast.unparse(node1.func) != ast.unparse(node2.func):
                return False
        else:
            return False

        # Compare arguments
        if len(node1.args) != len(node2.args):
            return False

        for arg1, arg2 in zip(node1.args, node2.args):
            if ast.unparse(arg1) != ast.unparse(arg2):
                return False

        return True


class SingleFunctionSubstitutor(ast.NodeTransformer):
    """Substitute calls to a specific function with a return variable."""

    def __init__(self, target_func_name: str, return_var: str):
        self.target_func_name = target_func_name
        self.return_var = return_var

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == self.target_func_name:
            return ast.Name(id=self.return_var, ctx=ast.Load())
        elif (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == self.target_func_name
        ):
            return ast.Name(id=self.return_var, ctx=ast.Load())

        return self.generic_visit(node)


class SpecificFunctionCallFinder(ast.NodeVisitor):
    """Find calls to a specific function name."""

    def __init__(self, target_func_name: str):
        self.target_func_name = target_func_name
        self.matching_calls = []

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == self.target_func_name:
            self.matching_calls.append(node)
        elif (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == self.target_func_name
        ):
            self.matching_calls.append(node)

        self.generic_visit(node)


class ParameterSubstitutor(ast.NodeTransformer):
    """AST transformer that substitutes parameter names in function bodies."""

    def __init__(self, param_mapping: Dict[str, str]):
        self.param_mapping = param_mapping

    def visit_Name(self, node: ast.Name) -> ast.AST:
        """Replace parameter names with their substituted versions."""
        if node.id in self.param_mapping:
            return ast.Name(id=self.param_mapping[node.id], ctx=node.ctx)
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        """Replace parameter names in function arguments."""
        if node.arg in self.param_mapping:
            node.arg = self.param_mapping[node.arg]
        return node


class ScopeInfo:
    """Information about a scope in the code."""

    def __init__(self, scope_type: str, name: str, start_line: int, end_line: int):
        self.scope_type = scope_type  # "function", "for_loop", "while_loop", etc.
        self.name = name
        self.start_line = start_line
        self.end_line = end_line

    def __repr__(self):
        return f"{self.scope_type}_{self.name}"

    def __eq__(self, other):
        if not isinstance(other, ScopeInfo):
            return False
        return (
            self.scope_type == other.scope_type
            and self.name == other.name
            and self.start_line == other.start_line
        )


class ScopeAnalyzer(ast.NodeVisitor):
    """Analyzes Python source code to build a comprehensive scope map.

    This pre-processes the code to understand which lines belong to which scopes.
    """

    def __init__(self, source_code: str):
        self.source_code = source_code
        self.lines = source_code.splitlines()
        self.scope_map: Dict[int, List[ScopeInfo]] = {}
        self.current_scope_stack: List[ScopeInfo] = []

    def analyze(self) -> Dict[int, List[ScopeInfo]]:
        """Analyze the source code and return the scope map."""
        # Initialize all lines with empty scope
        for i in range(1, len(self.lines) + 2):  # Extra line for safety
            self.scope_map[i] = []

        try:
            tree = ast.parse(self.source_code)
            self.visit(tree)
        except SyntaxError:
            # If parsing fails, return empty scope map
            pass

        return self.scope_map

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_scope_node(node, "function", node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scope_node(node, "function", node.name)

    def visit_For(self, node: ast.For) -> None:
        self._visit_scope_node(node, "for_loop", f"line_{node.lineno}")

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._visit_scope_node(node, "for_loop", f"line_{node.lineno}")

    def visit_While(self, node: ast.While) -> None:
        self._visit_scope_node(node, "while_loop", f"line_{node.lineno}")

    def visit_If(self, node: ast.If) -> None:
        self._visit_scope_node(node, "if", f"line_{node.lineno}")

    def visit_With(self, node: ast.With) -> None:
        self._visit_scope_node(node, "with", f"line_{node.lineno}")

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._visit_scope_node(node, "with", f"line_{node.lineno}")

    def _visit_scope_node(self, node: ast.AST, scope_type: str, name: str) -> None:
        """Visit a node that creates a new scope."""
        # Check if node has line number information
        try:
            start_line = node.lineno  # type: ignore
            end_line = getattr(node, "end_lineno", start_line)
        except AttributeError:
            return

        # Create scope info
        scope_info = ScopeInfo(scope_type, name, start_line, end_line)

        # Enter scope
        self.current_scope_stack.append(scope_info)

        # Record scope for each line in this node
        for line_num in range(start_line, end_line + 1):
            if line_num in self.scope_map:
                self.scope_map[line_num] = self.current_scope_stack.copy()

        # Visit children
        self.generic_visit(node)

        # Exit scope
        self.current_scope_stack.pop()


class CodeTracer:
    """Main tracer class that monitors Python execution and builds the execution tree.

    This implements the real-time execution monitoring described in the design document.
    """

    def __init__(self, llm_call_references: Optional[Set[Callable]] = None, repo_path: Optional[str] = None):
        self.llm_call_references = llm_call_references or set()
        self.repo_path = repo_path  # Only trace code within this path
        self.root: Optional[PythonCall] = None  # Will be initialized in trace_code
        self.parent_stack: List[PythonCall] = []
        self.current_parent: Optional[PythonCall] = (
            None  # Will always be valid after initialization
        )
        self.scope_map: Dict[int, List[ScopeInfo]] = {}
        self.source_lines: List[str] = []
        self.last_line: Optional[int] = None
        self.last_scope: List[ScopeInfo] = []

        # Enhanced multi-line statement support
        self.statement_map: Dict[int, Dict[str, Any]] = {}
        self.processed_statements: Set[Tuple[int, int, str]] = set()

        # Track execution state
        self.active_frames: Dict[int, PythonCall] = {}  # frame_id -> node
        self.frame_stack: List[Any] = []  # Stack of frames
        self.is_tracing: bool = False
        self.target_filename: str = "<traced>"  # Only trace this file

        # Track lines that are part of unrolled loops
        self.unrolled_loop_lines: Set[int] = set()

        # Track function calls
        self.pending_function_calls: Dict[
            int, PythonCall
        ] = {}  # frame_id -> function call node
        self.function_call_counter: Dict[str, int] = {}  # Track call count per function

        # Loop scope tracking
        self.active_loop_scopes: Dict[
            int, PythonCall
        ] = {}  # line_no -> loop scope node

        # Variable namespacing
        self.function_params: Dict[str, Dict[str, Any]] = {}
        self.loop_counters: Dict[str, int] = {}
        self.return_vars: Dict[str, str] = {}

        # Code substitution tracking
        self.substituted_code: Dict[
            int, Dict[int, str]
        ] = {}  # frame_id -> {line_no: substituted_code}
        self.active_substitutions: Dict[
            int, Dict[str, str]
        ] = {}  # frame_id -> param mappings

        # Clean tracking for pending expressions
        self.pending_expressions: Dict[
            str, List[PythonCall]
        ] = {}  # func_name -> [waiting_nodes]

        # New tracking for execution blocks
        self.pending_execution_blocks: Dict[
            str, List[ExpressionNode]
        ] = {}  # func_name -> [expression_nodes]

        # Logging
        self.debug_log: List[str] = []
        self.node_counter = 0

    def log(self, message: str) -> None:
        """Add debug log message."""
        self.debug_log.append(message)
        print(f"[TRACER] {message}")

    def log_parent_stack_operation(self, operation: str, context: str = "") -> None:
        """Log parent stack operations for debugging."""
        stack_names = [node.code[:20] + "..." for node in self.parent_stack]
        current_name = (
            self.current_parent.code[:20] + "..." if self.current_parent else "None"
        )
        print(
            f"[STACK] {operation} | Current: {current_name} | Stack: {stack_names} | Context: {context}"
        )

    def push_parent(self, new_parent: "PythonCall", context: str = "") -> None:
        """Push current parent to stack and set new current parent."""
        if self.current_parent:
            self.parent_stack.append(self.current_parent)
        self.current_parent = new_parent
        self.log_parent_stack_operation("PUSH", context)

    def pop_parent(self, context: str = "") -> Optional["PythonCall"]:
        """Pop parent from stack and restore as current parent."""
        if self.parent_stack:
            old_current = self.current_parent
            self.current_parent = self.parent_stack.pop()
            self.log_parent_stack_operation("POP", context)
            return old_current
        return None

    def _parse_expression_with_preemptive_substitution(self, code_line: str) -> Dict:
        """Parse expression and immediately substitute function calls with return variables using AST."""

        # Don't parse loops as expressions - they need special handling
        if any(code_line.strip().startswith(kw) for kw in ["for ", "while "]):
            return {
                "original": code_line,
                "final_expression": code_line,
                "execution_schedule": [],
                "substitution_map": {},
                "original_ast": None,
                "final_ast": None,
            }

        try:
            tree = ast.parse(code_line.strip())
        except SyntaxError:
            # Fallback for unparseable lines
            return {
                "original": code_line,
                "final_expression": code_line,
                "execution_schedule": [],
                "substitution_map": {},
                "original_ast": None,
                "final_ast": None,
            }

        # Extract all function calls using AST visitor
        call_extractor = FunctionCallExtractor()
        call_extractor.visit(tree)

        execution_schedule = []
        substitution_map = {}

        # Create execution schedule and substitution map
        for call_info in call_extractor.function_calls:
            func_name = call_info["name"]
            return_var = f"return_{func_name}"

            substitution_map[call_info["call_ast"]] = return_var
            execution_schedule.append(
                {
                    "function_name": func_name,
                    "return_var": return_var,
                    "call_ast": call_info["call_ast"],
                    "arguments": call_info["arguments"],
                    "call_text": call_info["call_text"],
                }
            )

        # Apply substitutions using AST transformer
        if substitution_map:
            substitutor = FunctionCallSubstitutor(substitution_map)
            new_tree = substitutor.visit(tree)
            final_expression = ast.unparse(new_tree)
        else:
            final_expression = code_line
            new_tree = tree

        return {
            "original": code_line,
            "final_expression": final_expression,
            "execution_schedule": execution_schedule,
            "substitution_map": substitution_map,
            "original_ast": tree,
            "final_ast": new_tree,
        }

    def _create_parameter_assignments_from_expression(
        self, func_name: str, params: Dict, current_expr: str, frame: Any
    ) -> List[str]:
        """Create parameter assignments using AST to parse the current expression."""
        assignments = []

        try:
            # Clean up expression for parsing
            clean_expr = current_expr.strip()
            tree = ast.parse(clean_expr)
        except SyntaxError:
            self.log(f"    Could not parse expression: {current_expr}")
            return assignments

        # Find function calls for this specific function
        call_finder = SpecificFunctionCallFinder(func_name)
        call_finder.visit(tree)

        if not call_finder.matching_calls:
            self.log(f"    No matching calls found for {func_name} in: {clean_expr}")
            return assignments

        # Use the first matching call (there should only be one for each function)
        call_node = call_finder.matching_calls[0]

        # Extract arguments using AST
        arguments = [ast.unparse(arg) for arg in call_node.args]

        self.log(f"    Found {func_name} call with args: {arguments}")

        # Create parameter assignments
        param_names = list(params.keys())
        for i, (param_name, arg_expr) in enumerate(zip(param_names, arguments)):
            namespaced_param = f"{param_name}_{func_name}_0"
            assignments.append(f"{namespaced_param} = {arg_expr}")

            self.log(f"    Parameter assignment: {namespaced_param} = {arg_expr}")

        return assignments

    def print_tree(self, title: str) -> None:
        """Print the current tree structure with enhanced visualization."""
        print(f"\n{'=' * 60}")
        print(f"🌳 {title}")
        print(f"{'=' * 60}")
        if self.root:
            self._print_node_enhanced(self.root, 0, is_last=True, prefix="")
        print(f"{'=' * 60}\n")

    def _print_node_enhanced(
        self, node: PythonCall, depth: int, is_last: bool = True, prefix: str = ""
    ) -> None:
        """Enhanced tree visualization with better symbols and structure."""

        # Determine node type and symbols
        if hasattr(node, "execution_type"):
            if node.execution_type == "expression":
                node_symbol = "📝"  # Expression with function calls
                type_info = (
                    f"[EXPR: {len(node.execution_schedule)} calls]"
                    if hasattr(node, "execution_schedule")
                    else "[EXPR]"
                )
            elif node.execution_type == "function_block":
                node_symbol = "⚙️"  # Execution block
                type_info = f"[EXEC: {getattr(node, 'function_name', 'unknown')}]"
            else:
                node_symbol = "📄"  # Regular statement
                type_info = "[STMT]"
        else:
            if node.is_function_call:
                node_symbol = "📞"  # Function call
                type_info = f"[CALL: {getattr(node, 'function_name', 'unknown')}]"
            else:
                node_symbol = "📄"  # Regular statement
                type_info = "[STMT]"

        # LLM and completion status
        llm_flag = "🔥" if node.contains_llm_call else "❄️"
        completed = "✅" if node.is_completed else "⏳"

        # Tree structure symbols
        if depth == 0:
            tree_symbol = "🌳"
            new_prefix = ""
        else:
            tree_symbol = "└── " if is_last else "├── "
            new_prefix = prefix + ("    " if is_last else "│   ")

        # Node description - truncate and clean up
        code_desc = node.code.replace("\n", " ").strip()
        if len(code_desc) > 40:
            code_desc = code_desc[:37] + "..."

        # Print the node
        print(f"{prefix}{tree_symbol}{node_symbol} {llm_flag}{completed} {code_desc}")
        print(
            f"{prefix}{'    ' if depth == 0 else ('    ' if is_last else '│   ')}{type_info}"
        )

        # Show key information based on node type
        if hasattr(node, "execution_type") and node.execution_type == "expression":
            # Expression node - show schedule and current state
            if hasattr(node, "execution_schedule") and node.execution_schedule:
                scheduled_funcs = [
                    exec_info["function_name"] for exec_info in node.execution_schedule
                ]
                print(
                    f"{prefix}{'    ' if depth == 0 else ('    ' if is_last else '│   ')}⏰ Scheduled: {scheduled_funcs}"
                )

            if (
                hasattr(node, "current_expression")
                and node.current_expression != node.code
            ):
                current_clean = node.current_expression.strip().replace("\n", " ")
                if len(current_clean) > 50:
                    current_clean = current_clean[:47] + "..."
                print(
                    f"{prefix}{'    ' if depth == 0 else ('    ' if is_last else '│   ')}🔄 Current: {current_clean}"
                )

            if hasattr(node, "pending_calls") and node.pending_calls:
                print(
                    f"{prefix}{'    ' if depth == 0 else ('    ' if is_last else '│   ')}⏸️  Pending: {list(node.pending_calls)}"
                )

        # Show content preview for different node types
        content_preview = self._get_content_preview(node)
        if content_preview:
            # Split content preview by lines and add proper indentation
            preview_lines = content_preview.split("\n")
            for preview_line in preview_lines:
                if preview_line.strip():  # Skip empty lines
                    print(
                        f"{prefix}{'    ' if depth == 0 else ('    ' if is_last else '│   ')}{preview_line.strip()}"
                    )

        # Print children
        if node.children:
            for i, child in enumerate(node.children):
                is_child_last = i == len(node.children) - 1
                self._print_node_enhanced(child, depth + 1, is_child_last, new_prefix)

    def _get_content_preview(self, node: PythonCall) -> str:
        """Get a content preview for the node."""
        # Show nonexpanded first (original), then expanded if different
        nonexp_preview = ""
        exp_preview = ""

        if node.nonexpanded:
            # Clean up nonexpanded content
            nonexp_clean = []
            for line in node.nonexpanded[:3]:  # Show max 3 lines
                clean_line = line.strip().replace("\n", " ")
                if len(clean_line) > 60:
                    clean_line = clean_line[:57] + "..."
                nonexp_clean.append(clean_line)

            if len(node.nonexpanded) > 3:
                nonexp_clean.append(f"... +{len(node.nonexpanded) - 3} more")

            nonexp_preview = f"📋 Code: {' | '.join(nonexp_clean)}"

        if node.expanded and node.expanded != node.nonexpanded:
            # Show expanded content if different
            exp_clean = []
            for line in node.expanded[:3]:  # Show max 3 lines
                clean_line = line.strip().replace("\n", " ")
                if len(clean_line) > 60:
                    clean_line = clean_line[:57] + "..."
                exp_clean.append(clean_line)

            if len(node.expanded) > 3:
                exp_clean.append(f"... +{len(node.expanded) - 3} more")

            exp_preview = f"🔧 Expanded: {' | '.join(exp_clean)}"

        # Combine previews
        previews = []
        if nonexp_preview:
            previews.append(nonexp_preview)
        if exp_preview:
            previews.append(exp_preview)

        return (
            "\n".join(f"{' ' * 4}{preview}" for preview in previews) if previews else ""
        )

    def trace_code(
        self, code: str, globals_dict: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Trace the execution of the given code and return flattened representation."""
        if self.is_tracing:
            return []  # Prevent recursive tracing

        self.is_tracing = True
        globals_dict = globals_dict or {}

        self.log("=== STARTING TRACE ===")

        # Analyze scopes
        analyzer = ScopeAnalyzer(code)
        self.scope_map = analyzer.analyze()
        self.source_lines = [""] + code.splitlines()  # 1-indexed

        # ENHANCED: Build statement mapping for multi-line support
        self.statement_map = self._build_statement_map(code)

        # Initialize root node (p0 from the document) - this ensures current_parent is never None
        self.root = PythonCall("root", nonexpanded=[])
        self.parent_stack = [self.root]
        self.current_parent = self.root
        self.log_parent_stack_operation("INIT", "created root node")
        self.log("Created root node (p0)")

        # Set up tracing
        old_trace = sys.gettrace()
        sys.settrace(self._trace_function)

        try:
            # Execute synchronous code only - avoid async complications
            exec(compile(code, self.target_filename, "exec"), globals_dict)
        except Exception as e:
            print(f"Execution error: {e}")
            traceback.print_exc()
        finally:
            # Disable tracing
            sys.settrace(old_trace)
            self.is_tracing = False

        self.log("=== EXECUTION COMPLETE ===")
        self.print_tree("After Execution")

        # Finalize any remaining nodes
        self._finalize_remaining_nodes()
        self.log("=== AFTER FINALIZING REMAINING NODES ===")
        self.print_tree("After Finalizing Remaining")

        # Finalize root and return its accumulated code
        self.root.finalize()
        self.log("=== FINAL TREE ===")
        self.print_tree("Final Tree")

        self.log(f"Final result: {self.root.expanded}")
        return self.root.expanded  # Always return expanded, never nonexpanded

    def _build_statement_map(self, code: str) -> Dict[int, Dict[str, Any]]:
        """Build a mapping of line numbers to complete statements using AST."""
        statement_map = {}

        self.log("=== BUILDING STATEMENT MAP ===")

        # Parse the entire code
        try:
            tree = ast.parse(code)
            source_lines = code.splitlines()

            # Walk through all statements in the AST
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.stmt)
                    and hasattr(node, "lineno")
                    and hasattr(node, "end_lineno")
                ):
                    start_line = node.lineno
                    end_line = node.end_lineno

                    # Extract complete statement text
                    stmt_lines = []
                    for line_num in range(start_line, end_line + 1):
                        if line_num <= len(source_lines):
                            stmt_lines.append(
                                source_lines[line_num - 1]
                            )  # Convert to 0-indexed

                    # Join into complete statement
                    complete_statement = "\n".join(stmt_lines).strip()

                    # Map each line in this statement to the complete statement info
                    for line_num in range(start_line, end_line + 1):
                        statement_map[line_num] = {
                            "start_line": start_line,
                            "end_line": end_line,
                            "complete_text": complete_statement,
                            "is_first_line": line_num == start_line,
                            "ast_node": node,
                        }

                    self.log(
                        f"  Statement lines {start_line}-{end_line}: {complete_statement[:50]}..."
                    )

        except Exception as e:
            self.log(f"Error building statement map: {e}")

        return statement_map

    def _trace_function(self, frame: Any, event: str, arg: Any) -> Optional[Callable]:
        """Trace function called by sys.settrace for each execution event."""
        filename = frame.f_code.co_filename
        
        # CRITICAL: Only trace the target code and repo files
        
        # FIRST: Skip Python internals and generated code completely
        if (filename.startswith('<frozen') or 
            filename.startswith('<built-in') or
            filename.startswith('<string>') or
            filename == '<string>'):
            return None
        
        # SECOND: Skip standard library files
        if '/lib/python' in filename and '/site-packages/' not in filename:
            return None
        
        # THIRD: Check if this is our target file
        if filename == self.target_filename:
            # Always trace the target file
            pass
        elif self.repo_path:
            # DISABLED: Always trace ALL code, regardless of repo path
            # The LLM call detection will filter what gets expanded
            # Only trace files within the repo path
            import os
            # Convert to absolute path for comparison
            abs_filename = os.path.abspath(filename)
            abs_repo_path = os.path.abspath(self.repo_path)
            
            # Check if the file is within the repo path
            # DISABLED: Always trace everything
            # if not abs_filename.startswith(abs_repo_path):
            #     return None  # Don't trace external code
            
            # DISABLED: Skip pattern filtering - trace everything
            # Also skip some common external patterns even within repo
            skip_patterns = [
                '/.venv/',
                '/venv/',
                '/site-packages/',
                '/dist-packages/',
                '/node_modules/',
                '/__pycache__/',
                '.pyc',
            ]
            
            # DISABLED: Always trace everything
            # if any(pattern in abs_filename for pattern in skip_patterns):
            #     return None  # Don't trace virtual env or cache files
            
            print(f"[TRACER] 📁 Tracing repo file: {filename}")
        else:
            # DISABLED: Always trace everything, even external code
            # return None  # Don't trace anything outside target code
            print(f"[TRACER] 🌐 Tracing external file: {filename}")
            pass  # Allow tracing of external code

        # Skip special functions
        func_name = frame.f_code.co_name
        if func_name in ["<module>"]:
            # Allow module-level execution
            pass
        elif func_name.startswith("__") and func_name.endswith("__"):
            return None  # Skip dunder methods

        try:
            if event == "call":
                self._handle_call(frame, arg)
            elif event == "line":
                self._handle_line(frame)
            elif event == "return":
                self._handle_return(frame, arg)
        except Exception as e:
            print(f"Tracer error in {event}: {e}")
            traceback.print_exc()

        return self._trace_function

    def _handle_line(self, frame: Any) -> None:
        """Enhanced line handler - process complete statements, not individual lines."""
        line_no = frame.f_lineno
        code_line = self._get_code_line(line_no, frame)

        # Skip empty lines
        if not code_line or not code_line.strip():
            return

        # current_parent is guaranteed to exist (at least p0)
        if self.current_parent is None:
            return  # Safety check, should not happen

        # Check if this line is part of a statement we've already processed
        stmt_info = self.statement_map.get(line_no)
        if not stmt_info:
            # Fallback for lines not in AST (shouldn't happen)
            self.log(f"LINE EVENT: line {line_no}: '{code_line.strip()}' [NOT IN AST]")
            return

        # Create unique statement key
        stmt_key = (
            stmt_info["start_line"],
            stmt_info["end_line"],
            stmt_info["complete_text"],
        )

        # Skip if we've already processed this statement
        if stmt_key in self.processed_statements:
            self.log(
                f"SKIPPING line {line_no}: already processed statement {stmt_info['start_line']}-{stmt_info['end_line']}"
            )
            return

        # Skip if this line is part of an unrolled loop
        if line_no in self.unrolled_loop_lines:
            self.log(f"SKIPPING line {line_no}: part of unrolled loop")
            return

        # Only process if this is the FIRST line of the statement
        if not stmt_info["is_first_line"]:
            self.log(
                f"SKIPPING line {line_no}: not first line of statement {stmt_info['start_line']}-{stmt_info['end_line']}"
            )
            return

        # Mark this statement as processed
        self.processed_statements.add(stmt_key)

        self.node_counter += 1
        self.log(
            f"STATEMENT EVENT #{self.node_counter}: lines {stmt_info['start_line']}-{stmt_info['end_line']}"
        )
        self.log(f"  Complete statement: {stmt_info['complete_text'][:100]}...")
        self.log(f"  Current parent: {self.current_parent.code[:30]}...")

        # Update scope for this line - this handles scope exits and finalizes nodes
        self._update_scope(line_no)

        # Process the complete statement
        self._handle_complete_statement(frame, stmt_info)

    def _handle_complete_statement(self, frame: Any, stmt_info: Dict[str, Any]) -> None:
        """Handle a complete statement identified by AST."""
        complete_text = stmt_info["complete_text"]
        start_line = stmt_info["start_line"]
        end_line = stmt_info["end_line"]
        ast_node = stmt_info["ast_node"]

        # Handle function definitions specially to capture full body
        if isinstance(ast_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Function definitions - extract complete function
            node = PythonCall(
                code=complete_text,
                nonexpanded=[
                    complete_text.split("\n")[0].strip()
                ],  # Just signature for nonexpanded
                line_number_start=start_line,
                line_number_end=end_line,
                line_number=start_line,  # Backward compatibility
            )
            self.log(
                f"  Created FUNCTION DEF: {complete_text.split()[0]} {complete_text.split()[1]}"
            )

            # Attach to parent and analyze function for LLM calls
            self.current_parent.add_child(node)
            self._analyze_function_definition_for_llm_calls(node, frame, complete_text)

            # Function definitions are always leaves
            node.finalize()
            return

            # For loops, we need to let them go through the normal flow
        # The scope system will handle loop tracking and unrolling
        # Don't treat loops as special statements here

        # Check for LLM calls in statements
        temp_node = PythonCall(code=complete_text, nonexpanded=[complete_text.strip()])
        self._check_for_llm_call(temp_node, frame, complete_text)

        # Check if this is a structural statement (for, while, if) that CONTAINS LLM calls
        # vs a direct LLM call statement
        is_structural = any(
            complete_text.strip().startswith(kw) for kw in ["for ", "while ", "if "]
        )

        if temp_node.contains_llm_call and not is_structural:
            # This is a DIRECT LLM call - treat as simple statement, DO NOT expand
            node = PythonCall(
                code=complete_text,
                original_code=complete_text.strip(),
                nonexpanded=[complete_text.strip()],
                line_number_start=start_line,
                line_number_end=end_line,
                line_number=start_line,
                is_ready_for_execution=True,
                contains_llm_call=True,  # Preserve LLM flag
                function_ref=temp_node.function_ref,  # Preserve reference
            )
            self.log(
                f"  Created LLM CALL statement (no expansion): {complete_text.strip()}"
            )

            # CRITICAL: Apply parameter substitution if we're inside a function execution block
            if id(frame) in self.active_substitutions:
                param_mapping = self.active_substitutions[id(frame)]
                substituted_code = self._apply_parameter_substitution(
                    node.nonexpanded, param_mapping
                )
                node.nonexpanded = substituted_code
                self.log(
                    f"  Applied parameter substitution to LLM call: {node.nonexpanded}"
                )

            # Attach to parent and propagate LLM flag
            self.current_parent.add_child(node)
            node.propagate_llm_flag()

            # LLM calls are always finalized immediately
            node.finalize()
            return

        # Parse expression with AST-based preemptive substitution
        expr_info = self._parse_expression_with_preemptive_substitution(complete_text)

        if expr_info["execution_schedule"]:
            # This expression contains function calls - create expression node with execution blocks
            expr_node = ExpressionNode(
                code=complete_text,
                original_code=complete_text.strip(),
                original_expression=expr_info["original"],
                current_expression=expr_info["original"],
                final_expression=expr_info["final_expression"],
                execution_schedule=expr_info["execution_schedule"],
                nonexpanded=[expr_info["final_expression"]],  # Store final result
                pending_calls=set(
                    exec_info["function_name"]
                    for exec_info in expr_info["execution_schedule"]
                ),
                is_ready_for_execution=False,
                line_number_start=start_line,
                line_number_end=end_line,
                line_number=start_line,
            )

            # Register for execution tracking
            for exec_info in expr_info["execution_schedule"]:
                func_name = exec_info["function_name"]
                if func_name not in self.pending_execution_blocks:
                    self.pending_execution_blocks[func_name] = []
                self.pending_execution_blocks[func_name].append(expr_node)

            self.current_parent.add_child(expr_node)
            self.log(
                f"  Created expression node with {len(expr_info['execution_schedule'])} execution blocks scheduled"
            )
            self.log(f"  Final expression will be: {expr_info['final_expression']}")

            # Check for function call tracking (preserve existing logic)
            func_call_match = self._extract_function_call(complete_text)
            if func_call_match:
                expr_node.is_function_call = True
                expr_node.function_name = func_call_match
                self.pending_function_calls[id(frame)] = expr_node
                self.log(f"  Also marked as function call: {func_call_match}")

        else:
            # Regular statement - no function calls
            node = PythonCall(
                code=complete_text,
                original_code=complete_text.strip(),
                nonexpanded=[complete_text.strip()],
                line_number_start=start_line,
                line_number_end=end_line,
                line_number=start_line,
                is_ready_for_execution=True,
            )
            self.log(f"  Created simple statement node: {node.nonexpanded}")

            # Check if this is a function call (preserve existing logic for tracking)
            func_call_match = self._extract_function_call(complete_text)
            if func_call_match:
                node.is_function_call = True
                node.function_name = func_call_match
                self.log(f"  Detected function call: {func_call_match}")
                self.pending_function_calls[id(frame)] = node

            # Special handling for different constructs
            if self._is_for_loop(complete_text):
                # Check if loop contains LLM calls
                temp_check = PythonCall(code="temp", nonexpanded=[complete_text])
                self._check_for_llm_call(temp_check, frame, complete_text)

                if temp_check.contains_llm_call:
                    # Unroll loops containing LLM calls
                    self.log("  Detected for loop with LLM calls - unrolling")
                    if self._unroll_for_loop(node, complete_text, frame):
                        # Successfully unrolled - attach to parent and finalize
                        self.current_parent.add_child(node)
                        node.propagate_llm_flag()
                        node.finalize()

                        # Mark the lines inside the loop as part of an unrolled loop
                        # This prevents them from being processed separately
                        for line_num in range(start_line + 1, end_line + 1):
                            self.unrolled_loop_lines.add(line_num)

                        return

                # Otherwise let scope system handle it
                self.log("  Detected for loop - will be handled by scope system")
            elif self._is_return_statement(complete_text):
                self._transform_return(node, complete_text, frame)

            # Apply parameter substitution if we're inside a function execution block
            if id(frame) in self.active_substitutions:
                param_mapping = self.active_substitutions[id(frame)]
                substituted_code = self._apply_parameter_substitution(
                    node.nonexpanded, param_mapping
                )
                node.nonexpanded = substituted_code
                self.log(f"  Applied parameter substitution: {node.nonexpanded}")

            # Attach to parent
            self.current_parent.add_child(node)
            self.log(f"  Attached to parent: {self.current_parent.code[:30]}...")

            # Check for LLM calls (might be missed in simple statements)
            self._check_for_llm_call(node, frame, complete_text)

            # Only finalize if ready for execution AND is leaf AND not function call
            if (
                node.is_ready_for_execution
                and node.is_leaf()
                and not node.is_function_call
            ):
                self.log("  Finalizing ready leaf node")
                node.finalize()
            elif node.is_function_call:
                self.log("  Function call node - waiting for execution")
            else:
                self.log("  Node not ready or not leaf - waiting")

    def _extract_function_call(self, code_line: str) -> Optional[str]:
        """Extract function name if this line is a function call."""
        # Skip function definitions
        if code_line.strip().startswith("def "):
            return None

        # Look for function calls - improved regex
        # Matches: func(), obj.method(), func(args), var = func(), etc.
        patterns = [
            r"(\w+)\s*\(",  # Simple function call
            r"=\s*(\w+)\s*\(",  # Assignment with function call
            r"\.(\w+)\s*\(",  # Method call
        ]

        for pattern in patterns:
            match = re.search(pattern, code_line)
            if match:
                func_name = match.group(1)
                # Exclude control structures
                if func_name not in ["if", "while", "for", "with", "except", "elif"]:
                    return func_name

        return None

    def _substitute_and_check_ready(
        self, expr_node: PythonCall, func_name: str, return_var: str
    ) -> None:
        """Substitute function call and check if expression is ready using AST."""

        current_code = (
            expr_node.nonexpanded[0]
            if expr_node.nonexpanded
            else expr_node.original_code
        )

        self.log(f"  Substituting {func_name} -> {return_var} in: {current_code}")

        try:
            # Parse and substitute using AST
            tree = ast.parse(current_code)

            class CallSubstitutor(ast.NodeTransformer):
                def visit_Call(self, node):
                    if isinstance(node.func, ast.Name) and node.func.id == func_name:
                        # Replace function call with return variable
                        return ast.Name(id=return_var, ctx=ast.Load())
                    elif (
                        isinstance(node.func, ast.Attribute)
                        and node.func.attr == func_name
                    ):
                        # Replace method call with return variable
                        return ast.Name(id=return_var, ctx=ast.Load())
                    return self.generic_visit(node)

            substitutor = CallSubstitutor()
            new_tree = substitutor.visit(tree)
            new_code = ast.unparse(new_tree)

            # Update the expression
            expr_node.nonexpanded = [new_code]
            expr_node.pending_calls.discard(func_name)
            expr_node.substitutions_made[func_name] = return_var

            self.log(f"  Result: {new_code}")

            # Check if expression is now ready
            if not expr_node.pending_calls:
                expr_node.is_ready_for_execution = True
                self.log("  Expression now ready!")

                # If it's a leaf, finalize it immediately
                if expr_node.is_leaf():
                    expr_node.finalize()
            else:
                self.log(f"  Still waiting for: {expr_node.pending_calls}")

        except Exception as e:
            self.log(f"  Substitution failed: {e}")
            # Fallback: mark as ready anyway to avoid hanging
            expr_node.is_ready_for_execution = True
            if expr_node.is_leaf():
                expr_node.finalize()

    def _handle_call(self, frame: Any, arg: Any) -> None:
        """Handle function call events - create clean execution blocks for function calls."""
        # Get line number and code
        line_no = frame.f_lineno
        func_name = frame.f_code.co_name

        # Skip module-level calls
        if func_name == "<module>":
            return

        self.node_counter += 1
        self.log(f"CALL EVENT #{self.node_counter}: {func_name}() at line {line_no}")
        
        # CRITICAL: Check if this is an LLM function call by comparing actual object references
        # Get the actual function object being called
        actual_func = self._get_actual_function_from_frame(frame)
        if actual_func:
            # Compare with our LLM references using proper object comparison
            for llm_ref in self.llm_call_references:
                if self._functions_match(actual_func, llm_ref):
                    self.log(f"  🔥 LLM FUNCTION CALL DETECTED (OBJECT MATCH): {func_name}() matches {getattr(llm_ref, '__name__', str(llm_ref))}")
                    self._mark_parent_as_containing_llm_calls()
                    return
        
        # If not a direct LLM call, check if this function contains LLM calls by analyzing its source
        if self._function_contains_llm_calls(frame):
            self.log(f"  🔥 FUNCTION WITH LLM CALLS DETECTED: {func_name}()")
            self._mark_parent_as_containing_llm_calls()

        # Track call count for this function
        if func_name not in self.function_call_counter:
            self.function_call_counter[func_name] = 0
        call_index = self.function_call_counter[func_name]
        self.function_call_counter[func_name] += 1

        # Find expression nodes waiting for this function
        waiting_expressions = self.pending_execution_blocks.get(func_name, [])

        if waiting_expressions:
            # New execution block approach
            for expr_node in waiting_expressions:
                # Create execution block
                execution_block = PythonCall(
                    code=f"{func_name}_execution_block",
                    nonexpanded=[],
                    execution_type="function_block",
                    function_name=func_name,
                    line_number_start=line_no,
                    line_number_end=line_no
                    + 1,  # Simplified for now, as execution block is single line
                )

                self.log(f"  Creating execution block for {func_name}")

                # Add to expression node
                expr_node.add_execution_block(func_name, execution_block)

                # Extract parameters from CURRENT expression state
                params = self._extract_function_params(frame)
                current_expr = expr_node.current_expression

                self.log(f"  Current expression state: {current_expr}")

                # Create parameter assignments using AST based on current expression
                param_assignments = self._create_parameter_assignments_from_expression(
                    func_name, params, current_expr, frame
                )

                # Add parameter assignment nodes to execution block
                for assignment in param_assignments:
                    param_node = PythonCall(
                        code=assignment,
                        nonexpanded=[assignment],
                        execution_type="statement",
                    )
                    execution_block.add_child(param_node)
                    param_node.finalize()
                    self.log(f"    Added parameter: {assignment}")

                # Set execution block as current parent for function body
                self.push_parent(
                    execution_block,
                    f"CALL {func_name}() - setting execution block as parent",
                )

                # Track this frame
                self.active_frames[id(frame)] = execution_block
                self.frame_stack.append(frame)

                # Store parameter mapping for substitution in function body
                param_mapping = {}
                for assignment in param_assignments:
                    # Parse "input_data_helper_multiline_0 = 'test input'" to get mapping: input_data -> input_data_helper_multiline_0
                    if " = " in assignment:
                        namespaced_param, value = assignment.split(" = ", 1)
                        # Extract original param name by removing _{func_name}_0 suffix
                        suffix_to_remove = f"_{func_name}_0"
                        if namespaced_param.endswith(suffix_to_remove):
                            original_param = namespaced_param[: -len(suffix_to_remove)]
                        else:
                            raise ValueError(
                                f"Unexpected parameter name format: {namespaced_param}"
                            )
                        param_mapping[original_param] = namespaced_param

                # Store parameter mapping for this frame for substitution
                if param_mapping:
                    self.active_substitutions[id(frame)] = param_mapping
                    self.log(f"  Parameter mapping for {func_name}: {param_mapping}")

                self.log(f"  Execution block {func_name} is now current parent")
                return  # Handle first expression only

        # Store function parameters for substitution (preserve existing functionality)
        self.function_params[func_name] = self._extract_function_params(frame)

    def _handle_return(self, frame: Any, arg: Any) -> None:
        """Handle function return events with execution block finalization and expression state updates."""
        frame_id = id(frame)
        func_name = frame.f_code.co_name

        # Skip creating duplicate return nodes
        if func_name == "<module>":
            return

        self.log(f"RETURN EVENT: {func_name}() -> {arg}")

        # Create return variable
        return_var = f"return_{func_name}"

        # Clean up substituted code for this frame (preserve existing)
        if frame_id in self.substituted_code:
            del self.substituted_code[frame_id]
            self.log("  Cleaned up substituted code for frame")

        if frame_id in self.active_substitutions:
            del self.active_substitutions[frame_id]

        # Find the execution block or function node for this frame
        if frame_id in self.active_frames:
            execution_node = self.active_frames[frame_id]

            # Return value will be handled by return statement nodes, no need for direct assignment

            # Check if this is an execution block (new approach)
            if (
                hasattr(execution_node, "execution_type")
                and execution_node.execution_type == "function_block"
            ):
                # This is an execution block - finalize it and update expression state
                execution_node.finalize()
                self.log(f"  Finalized execution block for {func_name}")

                # Find and update all expressions waiting for this function
                waiting_expressions = self.pending_execution_blocks.get(func_name, [])
                for expr_node in waiting_expressions:
                    # Update expression state
                    expr_node.update_expression_after_completion(func_name, return_var)

                    # Check if expression is now complete
                    if not expr_node.pending_calls:
                        expr_node.is_ready_for_execution = True
                        self.log(
                            f"  Expression now complete: {expr_node.current_expression}"
                        )

                        # If it's a leaf, finalize it
                        if expr_node.is_leaf():
                            expr_node.finalize()

                # Clean up execution block tracking
                if func_name in self.pending_execution_blocks:
                    del self.pending_execution_blocks[func_name]

            else:
                # Fallback: existing logic for backward compatibility
                execution_node.finalize()

                # Handle old-style pending expressions
                if func_name in self.pending_expressions:
                    for expr_node in self.pending_expressions[func_name]:
                        self._substitute_and_check_ready(
                            expr_node, func_name, return_var
                        )
                    del self.pending_expressions[func_name]

            # Restore parent from stack (preserve existing)
            if self.parent_stack and self.current_parent == execution_node:
                self.pop_parent(f"RETURN {func_name}() - restoring parent")
                self.log(f"  Restored parent to: {self.current_parent.code[:30]}...")

            # Clean up tracking (preserve existing)
            del self.active_frames[frame_id]
            if self.frame_stack and self.frame_stack[-1] == frame:
                self.frame_stack.pop()

    def _update_scope(self, line_no: int) -> None:
        """Update scope tracking based on line number and handle scope exits properly."""
        current_scope = self.scope_map.get(line_no, [])

        # If we're currently inside a function call (managed by CALL/RETURN),
        # don't update scopes based on AST - let CALL/RETURN handle it
        if self.current_parent and (
            self.current_parent.is_function_call
            or getattr(self.current_parent, "execution_type", None) == "function_block"
        ):
            self.last_line = line_no
            self.last_scope = current_scope
            return

        # Handle scope changes
        if self.last_line is not None:
            last_scope = self.last_scope

            # Find common ancestor depth
            common_depth = 0
            for i in range(min(len(last_scope), len(current_scope))):
                if last_scope[i] == current_scope[i]:
                    common_depth += 1
                else:
                    break

            # Exit scopes (multi-level exit handling from design doc)
            scopes_to_exit = len(last_scope) - common_depth
            for i in range(scopes_to_exit):
                # Determine which scope we're exiting
                exiting_scope_idx = len(last_scope) - i - 1
                exiting_scope = (
                    last_scope[exiting_scope_idx] if exiting_scope_idx >= 0 else None
                )

                # Check if we're exiting a function scope
                if exiting_scope and exiting_scope.scope_type == "function":
                    # For function scopes, don't pop if we're managed by CALL/RETURN
                    # Check if current_parent is a function call node
                    if self.current_parent and self.current_parent.is_function_call:
                        # This function is managed by CALL/RETURN, skip the pop
                        continue

                if len(self.parent_stack) > 1:  # Keep root (p0)
                    exiting_node = self.current_parent
                    # Don't finalize function call nodes - they're handled by return events
                    # Also don't finalize loop scopes - they can be re-entered
                    if (
                        exiting_node
                        and not exiting_node.is_completed
                        and not exiting_node.is_function_call
                    ):
                        # Check if this is a reusable loop scope
                        if (
                            hasattr(exiting_node, "is_loop_scope")
                            and exiting_node.is_loop_scope
                        ):
                            # Don't finalize loop scopes yet - they might be re-entered
                            pass
                        else:
                            exiting_node.finalize()
                    self.pop_parent("SCOPE exit")

                    # Ensure current_parent is never None
                    assert self.current_parent is not None, (
                        "Popped too many items from parent_stack"
                    )

            # Enter new scopes
            new_scopes = current_scope[common_depth:]
            for scope in new_scopes:
                # Functions are handled by call events, skip them here
                if scope.scope_type != "function":
                    # Check if we already have a scope node for this loop
                    if (
                        scope.scope_type in ["for_loop", "while_loop"]
                        and scope.start_line in self.active_loop_scopes
                    ):
                        # Reuse existing loop scope node
                        scope_node = self.active_loop_scopes[scope.start_line]
                        self.push_parent(scope_node, f"SCOPE reuse {scope}")
                    else:
                        # Create new scope node
                        scope_node = PythonCall(
                            code=f"# Entering {scope}",
                            nonexpanded=[],
                            line_number_start=line_no,
                            line_number_end=line_no + 1,  # Simplified for now
                        )

                        # If this is a loop scope, add initialization
                        if scope.scope_type in ["for_loop", "while_loop"]:
                            # Store loop info on the scope node
                            scope_node.is_loop_scope = True
                            scope_node.loop_line = scope.start_line
                            scope_node.loop_iteration_count = {}  # Track iterations per loop line

                            # Parse the loop line to get details
                            loop_line = self._get_code_line(scope.start_line)
                            if scope.scope_type == "for_loop":
                                match = re.match(
                                    r"^\s*for\s+(\w+)\s+in\s+(.+):", loop_line.strip()
                                )
                                if match:
                                    loop_var = match.group(1)
                                    collection = match.group(2).strip()
                                    scope_node.loop_var = loop_var
                                    scope_node.loop_collection = collection
                                    # Initialize index variable in expanded only once
                                    idx_var = f"index_loop_line_{scope.start_line}"
                                    scope_node.expanded.append(f"{idx_var} = 0")

                            # Store in active loop scopes
                            self.active_loop_scopes[scope.start_line] = scope_node

                        # current_parent is guaranteed to exist
                        assert self.current_parent is not None, (
                            "current_parent should never be None"
                        )
                        self.current_parent.add_child(scope_node)
                        self.push_parent(scope_node, f"SCOPE new {scope}")

        self.last_line = line_no
        self.last_scope = current_scope

    def _check_for_llm_call(self, node: PythonCall, frame: Any, code_line: str) -> None:
        """Check if this node represents an LLM call using AST analysis and reference comparison."""
        self.log(f"    🔍 Checking LLM call for: '{code_line.strip()}'")
        self.log(
            f"    🔍 LLM references to find: {[ref.__name__ if hasattr(ref, '__name__') else str(ref) for ref in self.llm_call_references]}"
        )
        
        # First try AST-based detection for more accurate results
        if self._ast_check_for_llm_call(node, frame, code_line):
            return

        # CRITICAL: Check BOTH locals AND globals
        all_vars = {}
        all_vars.update(frame.f_globals)  # Add globals first
        all_vars.update(frame.f_locals)  # Add locals (override globals if same name)

        # Look through both frame locals and globals for LLM objects
        for var_name, var_value in all_vars.items():
            if var_name.startswith("__"):  # Skip dunder variables
                continue

            # Check if this object has any of our LLM methods
            for llm_ref in self.llm_call_references:
                if hasattr(var_value, "__class__"):
                    # Check instance methods
                    for attr_name in dir(var_value):
                        if attr_name.startswith("_"):
                            continue
                        try:
                            attr = getattr(var_value, attr_name)
                            if not callable(attr):
                                continue

                            # NEW: Compare unbound methods instead of bound methods
                            # Check if this method's underlying function matches our LLM reference
                            matches_llm = False
                            if hasattr(attr, "__func__") and attr.__func__ == llm_ref:
                                # Bound method comparison via __func__
                                matches_llm = True
                            elif attr == llm_ref:
                                # Direct function comparison (fallback)
                                matches_llm = True

                            if matches_llm and (
                                f"{var_name}.{attr_name}" in code_line
                                or f".{attr_name}(" in code_line
                            ):
                                node.function_ref = llm_ref
                                node.contains_llm_call = True
                                self.log(
                                    f"  🔥 LLM CALL DETECTED: {var_name}.{attr_name} in '{code_line.strip()}'"
                                )
                                node.propagate_llm_flag()
                                return
                        except Exception:
                            pass

        # No text pattern fallback - AST analysis is more reliable

        self.log(f"    ❄️ No LLM call detected in: '{code_line.strip()}'")

    def _ast_check_for_llm_call(self, node: PythonCall, frame: Any, code_line: str) -> bool:
        """Use AST analysis to detect LLM calls by comparing actual function references."""
        try:
            # Parse the code line as an AST
            tree = ast.parse(code_line.strip())
            
            # Get all variables available in the frame
            all_vars = {}
            all_vars.update(frame.f_globals)
            all_vars.update(frame.f_locals)
            
            # Walk the AST to find function calls
            found_calls = False
            for ast_node in ast.walk(tree):
                if isinstance(ast_node, ast.Call):
                    found_calls = True
                    self.log(f"    🔍 AST: Found function call in '{code_line.strip()}'")
                    # Try to resolve the function being called
                    func_obj = self._resolve_function_from_ast_call(ast_node, all_vars)
                    self.log(f"    🔍 AST: Resolved function: {func_obj} (type: {type(func_obj)})")
                    
                    if func_obj:
                        # Check if this function matches any of our LLM references
                        for llm_ref in self.llm_call_references:
                            llm_name = getattr(llm_ref, "__name__", str(llm_ref))
                            self.log(f"    🔍 AST: Comparing with LLM ref {llm_name}: {llm_ref}")
                            if self._functions_match(func_obj, llm_ref):
                                node.function_ref = llm_ref
                                node.contains_llm_call = True
                                func_name = getattr(llm_ref, "__name__", str(llm_ref))
                                self.log(f"  🔥 LLM CALL DETECTED (AST): {func_name} in '{code_line.strip()}'")
                                node.propagate_llm_flag()
                                return True
            
            if not found_calls:
                self.log(f"    🔍 AST: No function calls found in '{code_line.strip()}'")
                        
        except Exception as e:
            self.log(f"    ⚠️ AST analysis failed: {e}")
            import traceback
            self.log(f"    ⚠️ AST traceback: {traceback.format_exc()}")
            
        return False
    
    def _resolve_function_from_ast_call(self, call_node: ast.Call, variables: dict) -> Any:
        """Resolve the actual function object from an AST Call node."""
        try:
            if isinstance(call_node.func, ast.Name):
                # Simple function call: func()
                func_name = call_node.func.id
                return variables.get(func_name)
                
            elif isinstance(call_node.func, ast.Attribute):
                # Method call: obj.method()
                obj = self._resolve_ast_expr(call_node.func.value, variables)
                if obj is not None:
                    method_name = call_node.func.attr
                    if hasattr(obj, method_name):
                        method = getattr(obj, method_name)
                        # Return the bound method for comparison
                        return method
                        
        except Exception as e:
            self.log(f"    ⚠️ Function resolution failed: {e}")
            
        return None
    
    def _resolve_ast_expr(self, expr: ast.AST, variables: dict) -> Any:
        """Resolve an AST expression to its runtime value."""
        try:
            if isinstance(expr, ast.Name):
                return variables.get(expr.id)
            elif isinstance(expr, ast.Attribute):
                obj = self._resolve_ast_expr(expr.value, variables)
                if obj is not None and hasattr(obj, expr.attr):
                    return getattr(obj, expr.attr)
        except Exception:
            pass
        return None
    
    def _functions_match(self, func1: Any, func2: Any) -> bool:
        """Check if two function references match, handling bound/unbound methods."""
        if func1 == func2:
            return True
            
        # Handle bound vs unbound method comparison
        if hasattr(func1, '__func__') and func1.__func__ == func2:
            return True
        if hasattr(func2, '__func__') and func2.__func__ == func1:
            return True
        if hasattr(func1, '__func__') and hasattr(func2, '__func__') and func1.__func__ == func2.__func__:
            return True
            
        return False
    
    def _analyze_function_definition_for_llm_calls(self, node: PythonCall, frame: Any, func_code: str) -> None:
        """Analyze a function definition to detect if it contains LLM calls."""
        try:
            # Parse the function code as AST
            tree = ast.parse(func_code.strip())
            
            # Get all variables available in the frame (for resolving references)
            all_vars = {}
            all_vars.update(frame.f_globals)
            all_vars.update(frame.f_locals)
            
            # Walk the AST to find function calls within the function body
            for ast_node in ast.walk(tree):
                if isinstance(ast_node, ast.Call):
                    # Try to resolve what function is being called
                    call_info = self._analyze_ast_call_for_llm(ast_node, all_vars)
                    if call_info:
                        func_name, is_llm_call = call_info
                        if is_llm_call:
                            node.contains_llm_call = True
                            self.log(f"  🔥 FUNCTION CONTAINS LLM CALL: {func_name} found in function definition")
                            node.propagate_llm_flag()
                            return
                            
        except Exception as e:
            self.log(f"    ⚠️ Function analysis failed: {e}")
    
    def _analyze_ast_call_for_llm(self, call_node: ast.Call, variables: dict) -> tuple[str, bool]:
        """Analyze an AST Call node to see if it's an LLM call."""
        try:
            if isinstance(call_node.func, ast.Attribute):
                # Method call like obj.method()
                if isinstance(call_node.func.value, ast.Name):
                    obj_name = call_node.func.value.id
                    method_name = call_node.func.attr
                    
                    # Check if this matches our LLM call patterns
                    for llm_ref in self.llm_call_references:
                        llm_method_name = getattr(llm_ref, "__name__", str(llm_ref))
                        # Check for patterns like mock_llm.forward, llm.ainvoke, etc.
                        if method_name == llm_method_name:
                            if (obj_name in ["mock_llm", "llm", "model"] or 
                                obj_name in variables and 
                                hasattr(variables[obj_name], method_name)):
                                return f"{obj_name}.{method_name}", True
                                
        except Exception as e:
            self.log(f"    ⚠️ Call analysis failed: {e}")
            
        return "", False
    
    def _function_contains_llm_calls(self, frame: Any) -> bool:
        """Check if the function in the given frame contains LLM calls by analyzing its source."""
        try:
            import inspect
            
            # Get the function object
            func_obj = frame.f_locals.get(frame.f_code.co_name)
            if not func_obj:
                # Try to get from globals
                func_obj = frame.f_globals.get(frame.f_code.co_name)
            
            if not func_obj or not callable(func_obj):
                return False
                
            # Get the source code of the function
            try:
                source = inspect.getsource(func_obj)
                self.log(f"    🔍 Analyzing function source for LLM calls")
                
                # Parse and analyze the source
                tree = ast.parse(source)
                for ast_node in ast.walk(tree):
                    if isinstance(ast_node, ast.Call):
                        call_info = self._analyze_ast_call_for_llm(ast_node, frame.f_globals)
                        if call_info and call_info[1]:  # is_llm_call
                            return True
                            
            except (OSError, TypeError):
                # Can't get source, skip
                pass
                
        except Exception as e:
            self.log(f"    ⚠️ Function source analysis failed: {e}")
            
        return False
    
    def _mark_parent_as_containing_llm_calls(self):
        """Mark the parent function as containing LLM calls."""
        # Walk up the parent stack to find execution blocks
        current = self.current_parent
        found_parent = False
        while current:
            if hasattr(current, 'execution_type') and current.execution_type == "function_block":
                if hasattr(current, 'function_name'):
                    func_name_to_mark = current.function_name
                    self.log(f"  🔥 Marking function execution block {func_name_to_mark} as containing LLM calls")
                    current.contains_llm_call = True
                    current.propagate_llm_flag()
                    found_parent = True
                    break
            elif hasattr(current, 'function_name') and current.function_name:
                func_name_to_mark = current.function_name
                self.log(f"  🔥 Marking parent function {func_name_to_mark} as containing LLM calls")
                current.contains_llm_call = True
                current.propagate_llm_flag()
                found_parent = True
                break
            current = current.parent
        
        if not found_parent:
            self.log(f"  ⚠️ No suitable parent found for LLM-containing function")
    
    def _get_actual_function_from_frame(self, frame: Any) -> Any:
        """Get the actual function/method object being called from the frame."""
        try:
            # Get the code object
            code_obj = frame.f_code
            func_name = code_obj.co_name
            
            # Look in frame locals first (for bound methods)
            if func_name in frame.f_locals:
                return frame.f_locals[func_name]
                
            # Look in frame globals
            if func_name in frame.f_globals:
                return frame.f_globals[func_name]
                
            # For method calls, try to reconstruct the method object
            # Look for 'self' in locals (indicates method call)
            if 'self' in frame.f_locals:
                self_obj = frame.f_locals['self']
                if hasattr(self_obj, func_name):
                    method = getattr(self_obj, func_name)
                    self.log(f"    🔍 Found method object: {method} on {type(self_obj)}")
                    return method
                    
            self.log(f"    ⚠️ Could not resolve function object for {func_name}")
            return None
            
        except Exception as e:
            self.log(f"    ⚠️ Error getting function from frame: {e}")
            return None

    def _finalize_remaining_nodes(self) -> None:
        """Finalize any nodes still in the parent stack."""
        # First, finalize any active loop scopes that haven't been finalized
        # Sort by line number in reverse order to finalize inner loops first
        sorted_loop_scopes = sorted(
            self.active_loop_scopes.items(), key=lambda x: x[0], reverse=True
        )
        for line_no, loop_scope in sorted_loop_scopes:
            if not loop_scope.is_completed:
                self.log(f"  Finalizing active loop scope: {loop_scope.code[:30]}...")
                loop_scope.finalize()

        # Then finalize nodes in the parent stack
        while len(self.parent_stack) > 1:  # Keep root (p0)
            if self.current_parent and not self.current_parent.is_completed:
                self.current_parent.finalize()
            self.pop_parent("FINALIZE remaining nodes")

        # Ensure we end up with root as current_parent
        assert self.current_parent == self.root, "Should end up back at root"

    def _extract_function_params(self, frame: Any) -> Dict[str, Any]:
        """Extract function parameters from frame."""
        code = frame.f_code
        param_count = code.co_argcount
        param_names = code.co_varnames[:param_count]

        params = {}
        for name in param_names:
            if name in frame.f_locals:
                params[name] = frame.f_locals[name]

        return params

    def _is_for_loop(self, code_line: str) -> bool:
        """Check if a line is a for loop."""
        return bool(re.match(r"^\s*for\s+\w+\s+in\s+", code_line.strip()))

    def _is_return_statement(self, code_line: str) -> bool:
        """Check if a line is a return statement."""
        return bool(re.match(r"^\s*return\s+", code_line.strip()))

    def _transform_for_loop(
        self, node: PythonCall, code_line: str, line_no: int
    ) -> None:
        """Transform for loop line to variable assignment - NOT USED in current implementation."""
        # This method is kept for compatibility but not actively used
        # Loop unrolling now happens in _unroll_for_loop
        node.nonexpanded = [code_line.strip()]

    def _unroll_for_loop(
        self, node: PythonCall, complete_text: str, frame: Any
    ) -> bool:
        """Unroll a for loop containing LLM calls into individual iterations.

        Returns True if successfully unrolled, False otherwise.
        """
        # Parse the complete for loop statement including body
        lines = complete_text.strip().split("\n")
        first_line = lines[0]

        # Parse the for loop header
        match = re.match(r"^\s*for\s+(\w+)\s+in\s+(.+):", first_line)
        if not match:
            self.log("  Failed to parse for loop header")
            return False

        loop_var = match.group(1)
        collection_expr = match.group(2).strip()

        # Extract loop body (everything after the first line, properly indented)
        if len(lines) > 1:
            # Get the indentation of the loop body
            body_lines = lines[1:]
            # Find minimum indentation
            min_indent = float("inf")
            for line in body_lines:
                if line.strip():  # Skip empty lines
                    indent = len(line) - len(line.lstrip())
                    min_indent = min(min_indent, indent)

            # Remove the common indentation
            loop_body = []
            for line in body_lines:
                if line.strip():
                    loop_body.append(line[min_indent:])
                else:
                    loop_body.append("")
        else:
            self.log("  No loop body found")
            return False

        # Try to evaluate the collection
        try:
            # Get the collection value from the frame
            collection = eval(collection_expr, frame.f_globals, frame.f_locals)
            if not hasattr(collection, "__iter__"):
                self.log(f"  Collection {collection_expr} is not iterable")
                return False
            collection_list = list(collection)
        except Exception as e:
            self.log(f"  Failed to evaluate collection {collection_expr}: {e}")
            return False

        # Generate unrolled code
        idx_var = f"idx_forloop_local_line_{node.line_number}"
        namespaced_var = f"{loop_var}_forloop_local_line_{node.line_number}"

        # Clear expanded and build unrolled version
        node.expanded = []

        # Initialize index
        node.expanded.append(f"{idx_var} = 0")

        # Unroll each iteration
        for i, item in enumerate(collection_list):
            if i > 0:
                node.expanded.append(f"{idx_var} += 1")

            # Assign item with namespaced variable
            node.expanded.append(f"{namespaced_var} = {collection_expr}[{idx_var}]")

            # Add transformed loop body
            for body_line in loop_body:
                if body_line.strip():
                    # Replace loop variable with namespaced version
                    transformed_line = re.sub(
                        rf"\b{loop_var}\b", namespaced_var, body_line
                    )
                    node.expanded.append(transformed_line)

        # Final increment (matching expected output pattern)
        if len(collection_list) > 0:
            node.expanded.append(f"{idx_var} += 1")

        self.log(f"  Successfully unrolled loop with {len(collection_list)} iterations")
        self.log(f"  Unrolled code: {node.expanded}")

        # Mark node as having expanded content
        node.contains_llm_call = True  # Ensure it uses expanded form

        return True

    def _transform_return(self, node: PythonCall, code_line: str, frame: Any) -> None:
        """Transform return statement."""
        func_name = frame.f_code.co_name
        if func_name != "<module>":
            match = re.match(r"^\s*return\s+(.+)", code_line.strip())
            if match:
                return_value = match.group(1)
                return_var = f"return_{func_name}"
                # Transform to assignment
                node.nonexpanded = [f"{return_var} = {return_value}"]

    def _get_code_line(self, line_no: int, frame: Optional[Any] = None) -> str:
        """Get source code for a line number (1-indexed), using substituted code if available."""
        # Check if we have substituted code for the current frame
        if frame:
            frame_id = id(frame)
            if (
                frame_id in self.substituted_code
                and line_no in self.substituted_code[frame_id]
            ):
                return self.substituted_code[frame_id][line_no]

        # Fallback to original source
        if 0 < line_no < len(self.source_lines):
            return self.source_lines[line_no]
        return ""

    def _apply_parameter_substitution(
        self, lines: List[str], param_mapping: Dict[str, str]
    ) -> List[str]:
        """Applies parameter substitution to a list of lines."""
        substituted_lines = []
        for line in lines:
            new_line = line
            for original_param, namespaced_param in param_mapping.items():
                # Use word boundaries to avoid partial matches
                pattern = r"\b" + re.escape(original_param) + r"\b"
                new_line = re.sub(pattern, namespaced_param, new_line)
            substituted_lines.append(new_line)
        return substituted_lines

    def _extract_full_function_def(self, start_line: int) -> List[str]:
        """Extract complete function definition starting from def line."""
        try:
            # Find the function in our source lines
            lines = []
            current_line = start_line

            # Add the def line
            def_line = self._get_code_line(current_line)
            lines.append(def_line.strip())

            # Add function body by looking at indentation
            current_line += 1
            base_indent = len(def_line) - len(def_line.lstrip())

            while current_line < len(self.source_lines):
                line = self._get_code_line(current_line)
                if not line.strip():  # Skip empty lines
                    current_line += 1
                    continue

                line_indent = len(line) - len(line.lstrip())
                if line_indent <= base_indent and line.strip():
                    # We've reached the end of the function
                    break

                lines.append(line.strip())
                current_line += 1

            return lines
        except:
            # Fallback to just the def line
            return [self._get_code_line(start_line).strip()]

    def _create_substituted_function_code(
        self, func_name: str, start_line: int, params: Dict[str, Any], call_index: int
    ) -> Dict[int, str]:
        """Create a substituted version of function code with parameters replaced."""
        # Get the function definition scope
        func_scope = None
        for scope in self.scope_map.get(start_line, []):
            if scope.scope_type == "function" and scope.name == func_name:
                func_scope = scope
                break

        if not func_scope:
            return {}

        # Create parameter mapping
        param_mapping = {}
        for i, (param_name, param_value) in enumerate(params.items()):
            namespaced_param = f"{param_name}_{func_name}_{call_index}"
            param_mapping[param_name] = namespaced_param

        # Extract function code and apply substitution
        substituted_lines = {}

        try:
            # Get all lines in the function
            func_lines = []
            for line_no in range(func_scope.start_line, func_scope.end_line + 1):
                if line_no < len(self.source_lines):
                    func_lines.append((line_no, self.source_lines[line_no]))

            # Parse the function as AST
            func_code = "\n".join(line for _, line in func_lines)
            tree = ast.parse(func_code)

            # Apply parameter substitution
            substitutor = ParameterSubstitutor(param_mapping)
            new_tree = substitutor.visit(tree)

            # Convert back to source code using ast.unparse()
            # This is much more reliable than regex substitution
            unparsed_code = ast.unparse(new_tree)

            # Split the unparsed code back into lines
            new_lines = unparsed_code.split("\n")

            # Map the new lines back to the original line numbers
            for i, (line_no, _) in enumerate(func_lines):
                if i < len(new_lines):
                    substituted_lines[line_no] = new_lines[i]

        except Exception as e:
            self.log(f"  Error creating substituted code: {e}")
            return {}

        return substituted_lines