"""Count SVD/QR/eigh/rq calls made through TensorCircuit's MPS gate-splitting path.

Canonical copy of the ``tap_backends`` helper originally written in
``src/examples/test.py`` (that file deliberately stays dependency-free for
its own Sections 1-3, so it keeps its own copy rather than importing this
one). New diagnostics -- see ``test_mps_gate_paths.py`` and
``benchmark_mps_gate_paths.py`` -- should import from here instead of adding
a third copy.
"""

import tensorcircuit as tc


def tap_backends():
    """Wrap svd/qr/eigh/rq on tc.backend and TensorNetwork's JaxBackend to
    count calls. tc.backend is NOT what performs MPS two-qubit-gate splits --
    FiniteMPS.apply_two_site_gate calls self.backend.svd, and self.backend is
    the (monkeypatched by tensorcircuit at import time) tensornetwork
    JaxBackend class.

    Returns (called, called_tn, restore); call restore() to undo the patch.
    """
    from tensornetwork.backends.jax.jax_backend import JaxBackend as TNJaxBackend

    called = {"svd": 0, "qr": 0, "eigh": 0, "rq": 0}
    called_tn = {"svd": 0, "qr": 0, "eigh": 0, "rq": 0}
    originals = []

    for name in list(called):
        if hasattr(tc.backend, name):
            orig = getattr(tc.backend, name)
            originals.append((tc.backend, name, orig))
            def make(n, f):
                def w(*a, **k):
                    called[n] += 1
                    return f(*a, **k)
                return w
            setattr(tc.backend, name, make(name, orig))
        if hasattr(TNJaxBackend, name):
            orig_tn = getattr(TNJaxBackend, name)
            originals.append((TNJaxBackend, name, orig_tn))
            def make_tn(n, f):
                def w(self, *a, **k):
                    called_tn[n] += 1
                    return f(self, *a, **k)
                return w
            setattr(TNJaxBackend, name, make_tn(name, orig_tn))

    def restore():
        for obj, name, orig in originals:
            setattr(obj, name, orig)

    return called, called_tn, restore
