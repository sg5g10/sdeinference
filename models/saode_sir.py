from jax import random
import jax.numpy as jnp
from jax.experimental import loops
from jax import  grad, jacobian
import jax.ops as ops

class SAODE_SIR(object):
    ## The SA-ODE rhs for the SIR model with Karhunen–Loève expansion (Fourier basis)
    def __init__(self, end_point, num_bases=10):
        self._num_bases = num_bases
        self._T = end_point
        self._frac_pop = 1/763

    def __call__(self, z, t, theta):
        
        S = z[0]
        I = z[1]
        e = self._num_bases
        beta, gamma = theta[...,0], theta[...,1]
        p1, p2 = theta[...,2:2+e], theta[...,2+e:2+(2*e)]


        with loops.Scope() as s:
            s.expn1 = jnp.zeros(e)
            s.expn2 = jnp.zeros(e)    
            s.Tt = self._T
            for i in s.range(s.expn1.shape[0]):
                s.expn1 = ops.index_update(s.expn1, i, \
                p1[i]*((2/s.Tt)**(1/2))*jnp.cos(((((2*i)-1.0)*jnp.pi)/(2*s.Tt))*t))
                s.expn2 = ops.index_update(s.expn2, i, \
                p2[i]*((2/s.Tt)**(1/2))*jnp.cos(((((2*i)-1.0)*jnp.pi)/(2*s.Tt))*t))
        expn = jnp.array([s.expn1.sum(),s.expn2.sum()])
        
        B = jnp.array([[beta*S*I,-beta*S*I],[-beta*S*I,(beta*S*I) + (gamma*I)]])
        sqB = jnp.linalg.cholesky(B*self._frac_pop)    

        def func(x):
            x1, x2 = x
            X = jnp.array([[jnp.sqrt(beta*x1*x2), 0],[-jnp.sqrt(beta*x1*x2), \
            jnp.sqrt(gamma*x2)]])*jnp.sqrt(self._frac_pop)
            return X.flatten()

        # This bit is for the Ito to Stratonovich convertion
        jacfn = jacobian(func)
        jacsqB = jacfn(z)
        drift_corr1 = 0.5*( 
            (sqB[0,0]*jacsqB[0,0]) + (sqB[0,1]*jacsqB[1,0]) +
            (sqB[1,0]*jacsqB[0,1]) + (sqB[1,1]*jacsqB[1,1]) 
            )
        drift_corr2 = 0.5*( 
            (sqB[0,0]*jacsqB[2,0]) + (sqB[0,1]*jacsqB[3,0]) +
            (sqB[1,0]*jacsqB[2,1]) + (sqB[1,1]*jacsqB[3,1])
            )
        
        drift_a = jnp.array([(-beta*S*I)-drift_corr1,\
        ((beta*S*I)-(gamma*I))-drift_corr2])
        
        # The SA-ODE
        dz = drift_a + jnp.matmul(sqB,expn)
        return dz
