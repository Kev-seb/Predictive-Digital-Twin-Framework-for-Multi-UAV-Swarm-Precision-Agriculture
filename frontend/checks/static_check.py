"""Check page syntax and missing global imports without executing workflows."""
import ast
import builtins
from pathlib import Path
import symtable

ROOT = Path(__file__).resolve().parents[1]
failures = []
for path in ROOT.rglob("*.py"):
    if "runtime" in path.parts:
        continue
    source = path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(path))
    if path.parent.name != "views":
        continue
    table = symtable.symtable(source, str(path), "exec")
    defined = {symbol.get_name() for symbol in table.get_symbols() if symbol.is_assigned() or symbol.is_imported()}
    defined.update(dir(builtins))
    defined.update({"__name__", "__file__"})
    pending = [table]
    missing = set()
    while pending:
        child = pending.pop()
        pending.extend(child.get_children())
        missing.update(symbol.get_name() for symbol in child.get_symbols() if symbol.is_global() and symbol.is_referenced() and symbol.get_name() not in defined)
    if missing:
        failures.append((path.name, sorted(missing)))
assert not failures, failures
print("Frontend syntax and page imports PASS")
