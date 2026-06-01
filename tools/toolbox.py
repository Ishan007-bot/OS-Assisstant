"""Tool use: a tiny, backend-agnostic tool-calling layer.

Instead of relying on each provider's native function-calling API (which
differs across Ollama / Groq / transformers), we use a simple text protocol
that works identically on every backend:

    The model is told it may emit ONE line:  TOOL: <name> <args>
    The app detects it, runs the tool, feeds back  TOOL_RESULT: <result>,
    and asks the model again for the final answer (a small ReAct-style loop).

Two safe, local, deterministic tools are provided:
    calculator <expr>   - evaluate an arithmetic expression (safe AST eval)
    now                 - the current date and time

These cover things an LLM genuinely cannot do reliably on its own (exact
arithmetic, the real current date). The calculator uses a restricted AST
evaluator — NOT Python eval() — so arbitrary code cannot run.
"""
import ast
import math
import operator
import re
from datetime import datetime

TOOL_SYSTEM_PROMPT = (
    "\n\nYou have access to tools. If a question needs an exact calculation or "
    "the current date/time, reply with EXACTLY one line and nothing else:\n"
    "TOOL: calculator <math expression>\n"
    "TOOL: now\n"
    "Do not explain when calling a tool. After you receive a line starting with "
    "'TOOL_RESULT:', use it to give your final answer in plain language. "
    "Available tools:\n"
    "- calculator: evaluates a math expression, e.g. `TOOL: calculator 23*47+19`\n"
    "- now: returns the current date and time, e.g. `TOOL: now`"
)

_TOOL_RE = re.compile(r"^\s*TOOL:\s*(\w+)\s*(.*)$", re.IGNORECASE | re.MULTILINE)

# --- safe calculator (restricted AST, no eval) -------------------------------
_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_NAMES = {"pi": math.pi, "e": math.e}
_FUNCS = {
    "sqrt": math.sqrt, "abs": abs, "round": round, "log": math.log,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "exp": math.exp,
}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Name) and node.id in _NAMES:
        return _NAMES[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_eval_node(a) for a in node.args])
    raise ValueError("unsupported expression")


def calculator(expr: str) -> str:
    try:
        tree = ast.parse(expr.strip(), mode="eval")
        return str(_eval_node(tree.body))
    except Exception:
        return f"error: could not evaluate '{expr.strip()}'"


def now(_arg: str = "") -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TOOLS = {"calculator": calculator, "now": now}


def parse_tool_call(text: str):
    """Return (tool_name, arg) if the text contains a TOOL: line, else None."""
    m = _TOOL_RE.search(text or "")
    if not m:
        return None
    name = m.group(1).lower()
    if name not in TOOLS:
        return None
    return name, m.group(2).strip()


def run_tool(name: str, arg: str) -> str:
    fn = TOOLS.get(name.lower())
    if not fn:
        return f"error: unknown tool '{name}'"
    return fn(arg)
