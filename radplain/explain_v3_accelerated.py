"""The same V3 algorithm, with checked official FLA operators on the A100."""
from .kernel_probe_v3 import prepare, activate
from .explain_v3 import explain as explain_algorithm


def explain(report,generate,**kwargs):
    state=prepare()
    activate(state,True)
    return explain_algorithm(report,generate,**kwargs)
