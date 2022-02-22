from jax import random
import jax.numpy as jnp
from jax.experimental import loops
from jax import jit
import jax.ops as ops

class SAODE_OU(object):
    ## The SA-ODE rhs for the Ornstein Uhlenbeck model with Karhunen–Loève expansion (Fourier basis)
    def __init__(self, end_point=10, num_bases=10):
        self._num_bases = num_bases
        self._T = end_point

        flam = lambda x,y,p:p*(jnp.sqrt(2/self._T)*jnp.cos(((\
             ((2.0*(x+1))-1.0)*jnp.pi)/(2.0*self._T))*y))
        self.jflam = jit(flam)

    def __call__(self, z, t, theta):

        e = self._num_bases
        theta1, theta2, theta3 = theta[...,0], theta[...,1], theta[...,2]
        p1 = theta[...,3:3+e]

        with loops.Scope() as s:
            s.expn = 0.0  
            s.bases = jnp.arange(e)
            s.expn = self.jflam(e,t,p1)
        
        # The SA-ODE
        dz = theta1*(theta2-z) + theta3*s.expn.sum() 
        return dz
