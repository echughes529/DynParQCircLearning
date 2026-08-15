"""T27: ToricCodeAnsatz.__hash__/__eq__ vs jit static_argnums (find_gs.purity/purity_vec)."""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
_sys.path.insert(0, _ROOT); _sys.path.insert(0, _HERE)
from src.utilities.ansatz_classes import ToricCodeAnsatz
import io, contextlib
def mk(**kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return ToricCodeAnsatz(Lx=2,Ly=2,nlayers=1,howoften_toreset=1,trials=2,maxiter=2,
                               sparse=False,**kw)
pairs = [("range_initial_parameters 1.0 vs pi", dict(range_initial_parameters=1.0), dict()),
         ("perform_noisy_simulations T vs F",   dict(perform_noisy_simulations=True), dict()),
         ("unitary True vs False",              dict(unitary=True), dict())]
for lbl,k1,k2 in pairs:
    a,b = mk(**k1), mk(**k2)
    same_keys = set(a.__dict__)==set(b.__dict__)
    print(f"{lbl}:")
    print(f"   hash equal = {hash(a)==hash(b)}   same dict keys = {same_keys}")
    try:
        print(f"   a == b -> {a==b}")
    except Exception as ex:
        print(f"   a == b RAISES {type(ex).__name__}: {str(ex)[:110]}")
