"""Parameters sections for dataclasses, built from their fields' own documentation.

The dataclasses of GRANDlib -- the data trees, ``Event``, ``ShowerEvent`` and
others -- document each field where it is declared: a string literal after
the field, or a ``##`` comment before it.  Neither reaches ``help()`` or the
API reference, so their constructors read as undocumented (#261).  Copying
the text into the class docstring would drift from the fields; this module
builds the section from the source instead, when the class is created.
"""
import ast
import functools
import inspect
import re
import sys

__all__ = ["document_fields"]

#: A comment line that is commented-out code (an assignment or annotation)
_CODE = re.compile(r"^[A-Za-z_][\w.]*\s*(:\s*[^=#]+)?=(?!=)")


@functools.lru_cache(maxsize=None)
def _parsed_module(path):
    r"""The classes of the module at `path`, by name (the first of each name), with its source lines.

    Each module is read and parsed once: asking ``inspect`` for every class
    and base class cost 0.9 s at import.
    """
    try:
        with open(path, encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
    except (OSError, SyntaxError, ValueError):
        return {}, []
    classes = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes.setdefault(node.name, node)
    return classes, source.splitlines()


def _source_class(cls):
    r"""The ``ast.ClassDef`` of `cls` and the source lines of its module, or ``(None, None)``."""
    path = getattr(sys.modules.get(cls.__module__), "__file__", None)
    if not path or not path.endswith(".py"):
        return None, None
    classes, lines = _parsed_module(path)
    node = classes.get(cls.__name__)
    return (node, lines) if node is not None else (None, None)


def _field_type(statement):
    r"""A readable type for a field: the dtype or vector type of a tree descriptor, else the annotation."""
    annotation = ast.unparse(statement.annotation)
    value = statement.value
    if isinstance(value, ast.Call) and getattr(value.func, "id", "") == "field":
        default = next((k.value for k in value.keywords if k.arg == "default"), None)
        if isinstance(default, ast.Call):
            value = default
    if not (isinstance(value, ast.Call) and annotation.endswith("Desc")):
        return annotation
    args = [ast.unparse(a).replace("np.", "") for a in value.args]
    kwargs = {k.arg: ast.literal_eval(k.value) for k in value.keywords
              if k.arg == "unit" and isinstance(k.value, ast.Constant)}
    if annotation == "StdStringDesc":
        text = "str"
    elif annotation == "StdVectorListDesc" and args:
        text = "std::vector<%s>" % args[0].strip("'\"")
    elif annotation == "TTreeArrayDesc" and len(args) >= 2:
        text = "array of shape %s, %s" % (args[0], args[1])
    elif annotation == "TTreeScalarDesc" and args:
        text = args[0]
    else:
        text = annotation
    if kwargs.get("unit"):
        text += ", in %s" % kwargs["unit"]
    return text


def _fields(cls):
    r"""``(name, type, description)`` of each public annotated field of `cls` and its bases, base first."""
    found = {}
    for klass in reversed(cls.__mro__):
        if klass is object or klass.__module__ == "builtins":
            continue
        node, lines = _source_class(klass)
        if node is None:
            continue
        body = node.body
        for i, statement in enumerate(body):
            if not (isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name)):
                continue
            name = statement.target.id
            if name.startswith("_"):
                continue
            text = ""
            following = body[i + 1] if i + 1 < len(body) else None
            if (isinstance(following, ast.Expr) and isinstance(following.value, ast.Constant)
                    and isinstance(following.value.value, str)):
                text = inspect.cleandoc(following.value.value)
            else:
                # Comment lines directly above the field (``##`` or ``#``),
                # leaving out commented-out code such as ``# x: int = 0``
                comments = []
                row = node.lineno - 1 + statement.lineno - node.lineno - 1
                while row >= 0 and lines[row].strip().startswith("#"):
                    text_line = lines[row].strip().lstrip("#").strip()
                    if not _CODE.match(text_line):
                        comments.insert(0, text_line)
                    row -= 1
                text = " ".join(comments)
            found[name] = (_field_type(statement), text)
    return [(name, kind, text) for name, (kind, text) in found.items()]


def document_fields(cls):
    r"""Appends a numpydoc ``Parameters`` section listing the fields of `cls`.

    Each field's description is the string literal that follows it in the
    class body, or else the ``##`` comment lines just above it.  A class whose
    docstring already has a ``Parameters`` or ``Attributes`` section, or whose
    source cannot be read, is left as it is.

    Parameters
    ----------
    cls : type
        A class whose fields are declared with annotations, usually a
        dataclass.

    Returns
    -------
    type
        `cls`, with its ``__doc__`` extended; usable as a class decorator.
    """
    doc = inspect.cleandoc(cls.__doc__ or "")
    if "\nParameters\n" in "\n" + doc or "\nAttributes\n" in "\n" + doc:
        return cls
    fields = _fields(cls)
    if not fields:
        return cls
    section = ["Parameters", "----------"]
    for name, kind, text in fields:
        section.append("%s : %s" % (name, kind))
        if text:
            section.extend("    " + line for line in text.splitlines())
    cls.__doc__ = (doc + "\n\n" if doc else "") + "\n".join(section) + "\n"
    return cls
