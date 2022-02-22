import jax.numpy as jnp
from jax.experimental import loops
from jax import jit
import jax.ops as ops

class SAODE_DW(object):
    ## The SA-ODE rhs for the Double-well model with Karhunen–Loève expansion (Fourier basis)
    def __init__(self, end_point=10, num_bases=50):
        self._num_bases = num_bases
        self._T = end_point

        flam = lambda x,y,p:p*(jnp.sqrt(2/self._T)*jnp.cos(((\
             ((2.0*(x+1))-1.0)*jnp.pi)/(2.0*self._T))*y))
        self.jflam = jit(flam)
    def __call__(self, z, t, theta):
        e = self._num_bases
        theta1, theta2, p1 = theta[...,0], theta[...,1], theta[...,2:]

        with loops.Scope() as s:
            s.expn = 0.0  
            s.bases = jnp.arange(e)
            s.expn = self.jflam(e,t,p1)
        
        # The SA-ODE
        dz = 4*z*(theta1 - (z**2))  + theta2*s.expn.sum() 
        return dz
