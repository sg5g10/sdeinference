import jax.numpy as jnp
from jax.experimental import loops
from jax import jacobian
import jax.ops as ops

class SAODE_LV(object):
    ## The SA-ODE rhs of the Lotka-Volterra with 
    ## 'KL': Karhunen–Loève expansion (Fourier basis)
    ## or the 'WV': Wavelet basis function
    def __init__(self, end_point, num_bases=10, expansion_type='KL'):
        self._num_bases = num_bases
        self._T = end_point
        self._expansion_type = expansion_type

    def __call__(self, z, t, theta):
        
        u = z[0]
        v = z[1]
        e = self._num_bases
        c1, c2, c3 = theta[..., 0], theta[..., 1], theta[..., 2] 
        p1, p2 = theta[..., 3:3+e], theta[..., 3+e:3+(2*e)]

        if self._expansion_type == 'KL':

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
        elif self._expansion_type == 'WV':

            with loops.Scope() as s:
                s.Tt = self._T
                s.mother = 1.0
                s.expn1 = (1/jnp.sqrt(s.Tt))*p1[0]*s.mother
                s.expn2 = (1/jnp.sqrt(s.Tt))*p2[0]*s.mother
                
                s.n = 0.0
                s.pwr = 0.0
                s.tau = 0.0
                s.k = 0.0    
                
                for i in s.range(1,e):
                    s.n = jnp.floor(jnp.log2(i))
                    s.k = (i) - (2**(s.n))
                    s.pwr = (s.n)/2
                    s.tau = ((2**(s.n))*t) - ((s.k)*(s.Tt))
                
                    s.mother = jnp.where((s.tau) < 0.0, 0.0,jnp.where((s.tau) > \
                    (s.Tt),0.0,jnp.where((s.tau) < (s.Tt/2),1.0,-1.0)))
                    s.expn1 += p1[i]*( (2**(s.pwr)/jnp.sqrt(s.Tt))*(s.mother))
                    s.expn2 += p2[i]*( (2**(s.pwr)/jnp.sqrt(s.Tt))*(s.mother))
            expn = jnp.array([s.expn1,s.expn2])            
    
        sqB = jnp.array([[jnp.sqrt(c1*u + (c2*u*v)), (-c2*u*v)/jnp.sqrt(c1*u + (c2*u*v))],
                    [0, jnp.sqrt((c3*v) + (c2*u*v) - (((c2*u*v)**2)/(c1*u + (c2*u*v))))]])
        
        def func(x):
            x1, x2 = x
            X = jnp.array([[jnp.sqrt(c1*x1 + (c2*x1*x2)), \
            (-c2*x1*x2)/jnp.sqrt(c1*x1 + (c2*x1*x2))],
                    [0, jnp.sqrt((c3*x2) + (c2*x1*x2) - \
                    (((c2*x1*x2)**2)/(c1*x1 + (c2*x1*x2))))]])
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
        
        drift_a = jnp.array([((c1*u)-(c2*u*v))-drift_corr1,\
        ((c2*u*v)-(c3*v))-drift_corr2])
        
        # The SA-ODE
        dz = drift_a + jnp.matmul(sqB,expn)
        return dz
