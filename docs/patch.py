# Patch the function documenter
import re
from typing import Any
from sphinx.ext.autodoc import FunctionDocumenter, MethodDocumenter


def _pretify(s: str) -> str:
    for keyword in ("ECEF", "LTP", "Quantity", "Path", "Time"):
        s = re.sub(f"[a-zA-Z0-9_.]*?[.]{keyword}", keyword, s)
    return s


class MyFunctionDocumenter(FunctionDocumenter):
    def format_args(self, **kwargs) -> str:
        result = super().format_args(**kwargs)
        if result:
            return _pretify(result)
        return result


class MyMethodDocumenter(MethodDocumenter):
    def format_args(self, **kwargs) -> str:
        result = super().format_args(**kwargs)
        if result:
            return _pretify(result)
        return result


def setup(app: Any) -> None:
    app.add_autodocumenter(MyFunctionDocumenter)
    app.add_autodocumenter(MyMethodDocumenter)
