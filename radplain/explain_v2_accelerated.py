"""Runtime-only wrapper for an otherwise byte-identical V2 reference package.

The final packager places this alongside the OLD frozen V2 sources, not alongside
the newer V2 checkout. Both compared arms use the same checked FLA operators.
"""
from .kernel_probe_v3 import prepare, activate
from .explain_v2 import explain as explain_algorithm


def explain(report,generate,**kwargs):
    state=prepare()
    activate(state,True)
    return explain_algorithm(report,generate,**kwargs)
