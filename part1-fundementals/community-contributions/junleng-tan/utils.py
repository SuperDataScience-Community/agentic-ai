import inspect
import sys
import typing
from typing import get_origin, get_args, Literal, Union, Optional

try:
    from typing import TypedDict  # py3.8+
except ImportError:
    TypedDict = None

def _is_typeddict(t):
    try:
        return isinstance(t, type) and TypedDict is not None and issubclass(t, TypedDict)
    except TypeError:
        return False

def _is_dataclass(t):
    try:
        import dataclasses
        return dataclasses.is_dataclass(t)
    except Exception:
        return False

def _docstring_split_sections(doc: str):
    """Very small parser to extract:
       - short summary (first non-empty line)
       - param descriptions from sections like 'Args:', 'Parameters:', ':param x:'.
    """
    if not doc:
        return "", {}

    lines = [l.rstrip() for l in doc.strip().splitlines()]
    # Summary = first nonempty line
    summary = next((l for l in lines if l.strip()), "")
    params_desc = {}

    # Gather “Args/Parameters/Arguments” blocks (Google/Numpy style)
    markers = {"args:", "parameters:", "arguments:"}
    i = 0
    while i < len(lines):
        line = lines[i].strip().lower()
        if line in markers:
            i += 1
            while i < len(lines):
                raw = lines[i]
                if raw.strip() == "" or raw.startswith(" "):
                    # keep reading indented or blank continuation lines
                    # detect "name (type): desc" or "name: desc"
                    stripped = raw.strip()
                    if stripped:
                        # Try common patterns
                        if ":" in stripped:
                            name, desc = stripped.split(":", 1)
                            name = name.strip().split()[0].split("(")[0]
                            params_desc.setdefault(name, desc.strip())
                        else:
                            # continuation line: append to last desc if any
                            if params_desc:
                                last = list(params_desc.keys())[-1]
                                params_desc[last] += " " + stripped
                    i += 1
                else:
                    break
            continue
        i += 1

    # Sphinx-style ":param name: desc"
    for l in lines:
        ls = l.strip()
        if ls.lower().startswith(":param "):
            try:
                rest = ls[len(":param "):]
                name, desc = rest.split(":", 1)
                name = name.strip().split()[0]
                params_desc[name] = desc.strip()
            except ValueError:
                pass

    return summary, params_desc


def _json_type_for_python(t):
    """Return a JSON Schema fragment for python/typing type t."""
    origin = get_origin(t)
    args = get_args(t)

    # NoneType
    if t is type(None):
        return {"type": "null"}

    # Builtins
    if t is str:
        return {"type": "string"}
    if t is int:
        return {"type": "integer"}
    if t is float:
        return {"type": "number"}
    if t is bool:
        return {"type": "boolean"}

    # datetime-like
    try:
        import datetime as _dt
        if t in (_dt.datetime,):
            return {"type": "string", "format": "date-time"}
        if t in (_dt.date,):
            return {"type": "string", "format": "date"}
        if t in (_dt.time,):
            return {"type": "string", "format": "time"}
        if t in (_dt.timedelta,):
            # no standard JSON Schema, fallback to string
            return {"type": "string", "description": "Duration (ISO 8601 or human-readable)."}
    except Exception:
        pass

    # Enum
    import enum
    if isinstance(t, type) and issubclass(t, enum.Enum):
        values = [e.value for e in t]
        # infer primitive type of the enum values
        if all(isinstance(v, str) for v in values):
            return {"type": "string", "enum": values}
        if all(isinstance(v, int) for v in values):
            return {"type": "integer", "enum": values}
        # mixed types
        return {"enum": values}

    # TypedDict
    if _is_typeddict(t):
        props = {}
        required = []
        # __annotations__ holds fields
        ann = t.__annotations__
        total = getattr(t, "__total__", True)
        for k, v in ann.items():
            props[k] = _json_type_for_python(v)
            # In total=True, all are required unless Optional/Union[..., None]
            if total:
                if not _is_optional(v):
                    required.append(k)
            else:
                # total=False => all optional
                pass
        schema = {"type": "object", "properties": props}
        if required:
            schema["required"] = required
        return schema

    # dataclass
    if _is_dataclass(t):
        import dataclasses
        props = {}
        required = []
        for f in dataclasses.fields(t):
            props[f.name] = _json_type_for_python(f.type)
            has_default = f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING
            if not has_default and not _is_optional(f.type):
                required.append(f.name)
        schema = {"type": "object", "properties": props}
        if required:
            schema["required"] = required
        return schema

    # Literal
    if origin is Literal:
        vals = list(args)
        # infer a base type if uniform
        if all(isinstance(v, str) for v in vals):
            return {"type": "string", "enum": vals}
        if all(isinstance(v, int) for v in vals):
            return {"type": "integer", "enum": vals}
        if all(isinstance(v, (int, float)) for v in vals):
            # number enum
            return {"type": "number", "enum": vals}
        return {"enum": vals}

    # Optional[T] == Union[T, None]
    if _is_optional(t):
        # caller should handle required vs optional; here return the underlying schema
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _json_type_for_python(non_none[0])
        # Optional of Union[…, None] falls through to anyOf
        return {"anyOf": [_json_type_for_python(a) for a in non_none] + [{"type": "null"}]}

    # Union
    if origin is Union:
        return {"anyOf": [_json_type_for_python(a) for a in args]}

    # List/Tuple/Set
    if origin in (list, tuple, set, typing.Sequence, typing.MutableSequence):
        item_t = args[0] if args else typing.Any
        return {"type": "array", "items": _json_type_for_python(item_t)}

    # Dict / Mapping
    if origin in (dict, typing.Mapping, typing.MutableMapping):
        key_t, val_t = (args + (typing.Any, typing.Any))[:2]
        # JSON keys must be strings; if key_t != str, we note it in description
        schema = {"type": "object", "additionalProperties": _json_type_for_python(val_t)}
        if key_t is not str:
            schema["description"] = (schema.get("description", "") + " Keys will be stringified.").strip()
        return schema

    # Fallbacks
    if t is typing.Any or t is None:
        return {}  # unconstrained
    # Unknown type => treat as string with note
    return {"type": "string", "description": f"Serialized {getattr(t, '__name__', str(t))}."}


def _is_optional(t):
    origin = get_origin(t)
    if origin is Union:
        args = get_args(t)
        return any(a is type(None) for a in args)
    return False


def function_to_tool(
    func,
    *,
    name: str | None = None,
    description: str | None = None,
    param_overrides: dict | None = None,
) -> dict:
    """
    Build an OpenAI-style 'tool' schema from a Python function.

    - `name`: override function name.
    - `description`: override function description (otherwise from docstring summary).
    - `param_overrides`: dict of per-param overrides, e.g.
        {
          "city": {"description": "City name", "enum": ["Dubai", "Abu Dhabi"]},
          "units": {"default": "metric"}  # note: default isn't used by schema; make param optional instead
        }
    """
    sig = inspect.signature(func)
    hints = typing.get_type_hints(func, include_extras=True)
    doc = inspect.getdoc(func) or ""
    summary, param_descs = _docstring_split_sections(doc)

    tool_name = name or func.__name__
    tool_desc = description or summary or f"Callable function `{tool_name}`."

    properties = {}
    required = []

    for pname, param in sig.parameters.items():
        if pname == "self":
            continue

        ann = hints.get(pname, typing.Any)
        schema = _json_type_for_python(ann)

        # base description from docstring, if any
        if param_descs.get(pname):
            schema["description"] = param_descs[pname]

        # overrides
        if param_overrides and pname in param_overrides:
            schema.update(param_overrides[pname])

        # required vs optional
        is_required = (
            param.default is inspect._empty
            and not _is_optional(ann)
        )
        if is_required:
            required.append(pname)

        # ensure at least a basic type if none inferred
        if not schema:
            schema = {"type": "string"}

        properties[pname] = schema

    parameters = {
        "type": "object",
        "properties": properties,
    }
    if required:
        parameters["required"] = required

    return {
        "type": "function",
        "name": tool_name,
        "description": tool_desc,
        "parameters": parameters,
    }
