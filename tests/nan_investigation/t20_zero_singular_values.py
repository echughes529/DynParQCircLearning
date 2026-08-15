"""T20c: does the second (KEPT) singular value ever come out exactly 0 in float32?
   Build the operator-Schmidt matrix analytically and batch-SVD it."""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
_sys.path.insert(0, _ROOT); _sys.path.insert(0, _HERE)
import numpy as np
I2=np.eye(2); X=np.array([[0,1],[1,0]]); Y=np.array([[0,-1j],[1j,0]]); Z=np.diag([1,-1])
def schmidt_mats(P, ths, dt):
    # U = cos(t/2) I(x)I - i sin(t/2) P(x)P ; M[(o0,i0),(o1,i1)] = U[o0,o1,i0,i1]
    n=len(ths)
    U = (np.cos(ths/2)[:,None,None,None,None]*np.einsum('ac,bd->abcd',I2,I2)
         -1j*np.sin(ths/2)[:,None,None,None,None]*np.einsum('ac,bd->abcd',P,P))
    return np.ascontiguousarray(U.transpose(0,1,3,2,4).reshape(n,4,4)).astype(dt)

pts = np.unique(np.concatenate([
    np.linspace(0,2*np.pi,400001),
    np.array([c+s*10.0**(-k) for c in (0,np.pi/2,np.pi,3*np.pi/2,2*np.pi)
              for k in range(1,20) for s in (1,-1)]),
    np.array([0,np.pi/2,np.pi,3*np.pi/2,2*np.pi],float)]))
print("scanning", len(pts), "theta values")
for name,P in [("rxx",X),("ryy",Y),("rzz",Z)]:
    for dtname,dt,cast in [("float32",np.complex64,np.float32),("float64",np.complex128,np.float64)]:
        th = cast(pts).astype(np.float64)
        S = np.linalg.svd(schmidt_mats(P, cast(pts).astype(cast), dt), compute_uv=False)
        z  = np.where(S[:,1]==0.0)[0]
        t1 = np.where(S[:,1] < 1e-7)[0]
        deg= np.where(np.abs(S[:,0]-S[:,1]) == 0.0)[0]
        print(f"  {name} {dtname}: s1==0 for {len(z):5d} thetas | s1<1e-7 for {len(t1):5d} | "
              f"s0==s1 exactly for {len(deg):5d}", end="")
        if len(z): print(f"  e.g. theta/pi={np.round(pts[z]/np.pi,10)[:5]}")
        else: print()
