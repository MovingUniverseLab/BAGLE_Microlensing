"""
fitter_param_names must match the Param-class constructors.

The fitters build models positionally, in ``fitter_param_names`` order
(``model_fitter.get_model``), and append GP hyperparameters as optional
per-filter dictionaries. So for every Param mixin:

* the constructor's leading arguments are ``fitter_param_names``, in order,
  and every other required argument is a GP hyperparameter or a fixed
  keyword supplied from the data;
* each positional argument a constructor forwards to its parent's
  ``__init__`` lands on the parent parameter of the same name (except for
  intentional renames, listed below).
"""
from __future__ import annotations

import ast
import inspect
import re

import pytest

from bagle import model, model_jax

# Arguments that are never fitted positionally.
SKIP = {"self", "raL", "decL", "obsLocation", "root_tol", "args", "kwargs"}
FIXED_KEYWORDS = {"t0par"}

# Intentional child -> parent renames in super().__init__ calls.
RENAMES = {
    ("sepL", "aL"),  # circular binary-source orbits: a = separation
    ("sep12", "sep"), ("alpha12", "alpha"), ("q_12", "q"),  # PSTL: pair 1-2
    ("i_12", "i"), ("e_12", "e"), ("tp_12", "tp"), ("a_12", "a"),
    # Keplerian CoM-input classes pass xS0_com_* to a static parent's xS0_*,
    # and PSTL passes its xS0_* to the PSBL Keplerian xS0_com_*.
    ("xS0_com_E", "xS0_E"), ("xS0_com_N", "xS0_N"),
    ("xS0_E", "xS0_com_E"), ("xS0_N", "xS0_com_N"),
}


def param_classes(mod):
    """Param mixins (not composed model classes) defined in a module."""
    out = []
    for name, cls in sorted(vars(mod).items()):
        if not (inspect.isclass(cls) and hasattr(cls, "fitter_param_names")):
            continue
        if not re.search(r"[Pp]aram\d", name):
            continue
        if "ModelClassABC" in [b.__name__ for b in cls.__mro__]:
            continue
        out.append((mod.__name__.split(".")[-1], name, cls))
    return out


CLASSES = param_classes(model) + param_classes(model_jax)
IDS = [f"{m}.{n}" for m, n, _ in CLASSES]


@pytest.mark.parametrize("mod_name,name,cls", CLASSES, ids=IDS)
def test_constructor_order_matches_fitter_names(mod_name, name, cls):
    """Constructor arguments start with fitter_param_names, in order."""
    params = [p for p in inspect.signature(cls.__init__).parameters.values()
              if p.name not in SKIP]
    args = [p.name for p in params]
    names = list(cls.fitter_param_names)
    assert args[:len(names)] == names

    # Any other required argument must be a GP hyperparameter or fixed keyword.
    extra = [p.name for p in params[len(names):]
             if p.default is inspect.Parameter.empty
             and not p.name.startswith("gp_") and p.name not in FIXED_KEYWORDS]
    assert extra == []
    return None


SUPER_CASES = [(m, n, c) for m, n, c in CLASSES if "__init__" in c.__dict__]


@pytest.mark.parametrize("mod_name,name,cls", SUPER_CASES,
                         ids=[f"{m}.{n}" for m, n, _ in SUPER_CASES])
def test_super_init_forwards_same_names(mod_name, name, cls):
    """Positional arguments forwarded to the parent keep their meaning."""
    tree = ast.parse("class _X:\n" + inspect.getsource(cls.__dict__["__init__"]))
    parent = next(b for b in cls.__mro__[1:] if "__init__" in b.__dict__)
    parent_args = [p for p in inspect.signature(parent.__init__).parameters
                   if p != "self"]
    own_args = set(inspect.signature(cls.__init__).parameters)

    bad = []
    for node in ast.walk(tree):
        is_super = (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "__init__"
                    and isinstance(node.func.value, ast.Call)
                    and getattr(node.func.value.func, "id", "") == "super")
        if not is_super:
            continue
        for i, arg in enumerate(node.args):
            child = ast.unparse(arg)
            if child not in own_args:
                continue
            target = parent_args[i] if i < len(parent_args) else None
            if child != target and (child, target) not in RENAMES:
                bad.append(f"{child} -> {target}")
    assert bad == []
    return None
