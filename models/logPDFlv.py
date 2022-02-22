import numpy as np
import scipy
import scipy.stats as stats
import numpyro.distributions as dist
from jax import jit
from functools import partial
from jax import random, device_get, device_put
import jax.numpy as jnp
from jax.experimental import loops
import jax.ops as ops
from jax.config import config
config.update("jax_enable_x64", True)

@partial(jit, static_argnums=(4,5,6,7))
def EulerMaruyama(key, param, init_val, t, dt, P, N, D):
    with loops.Scope() as s:
        s.c1 = param[0]
        s.c2 = param[1]
        s.c3 = param[2]
        s.dt = dt
        s.z = jnp.zeros((P,N,D))
        s.t = t
        s.z = s.z.at[...,0,:].set(init_val)
        
        s.B = jnp.zeros((P,D,D))
        s.Ad = jnp.zeros((P,D))
        s.ep = jnp.zeros((P,D))
        s.subkey = key
        for i in s.range(N-1):
            key, s.subkey = random.split(s.subkey,2)

            s.B= s.B.at[...,0,0].set(s.c1*s.z[...,i,0] + (s.c2*s.z[...,i,0]*s.z[...,i,1]))
            s.B= s.B.at[...,1,0].set(-s.c2*s.z[...,i,0]*s.z[...,i,1])
            s.B= s.B.at[...,0,1].set(-s.c2*s.z[...,i,0]*s.z[...,i,1])
            s.B= s.B.at[...,1,1].set(s.c3*s.z[...,i,1] +(s.c2*s.z[...,i,0]*s.z[...,i,1]))
            
            s.Ad = s.Ad.at[...,0].set((s.c1*s.z[...,i,0])-(s.c2*s.z[...,i,0]*s.z[...,i,1]))
            s.Ad = s.Ad.at[...,1].set((s.c2*s.z[...,i,0]*s.z[...,i,1])-(s.c3*s.z[...,i,1]))
            s.ep = random.multivariate_normal(s.subkey,s.z[...,i,:] + (s.Ad*s.dt),s.B*s.dt)
            s.z = s.z.at[...,i+1,:].set(s.ep)
            s.t = s.t + s.dt
    return s.z, s.t

@partial(jit, static_argnums=(4,5,6,7,8))
def BootStrapp(key, y, param, init_val, dt, P, N, D, num_steps):
    with loops.Scope() as s:
        s.x_0 = jnp.ones((P,D))*init_val
        s.z = jnp.zeros((P,D))
        s.X = jnp.zeros((num_steps,P,D))

        s.w = jnp.exp(dist.Normal(s.x_0,10.).log_prob(y[0,:]).sum(axis=1))
        s.mLik = jnp.log(s.w.mean())
        s.wt = s.w/jnp.sum(s.w)
        s.ind = random.choice(key, a=P, shape=(P,), p=s.wt)
        s.particles = s.x_0[s.ind,:]
        s.t = jnp.zeros(num_steps)
        s.key2 = key
        s.temp_state = jnp.zeros((P,N,D))
        s.temp_time = 0.
        for i in s.range(num_steps): 
            s.z = s.particles
            s.key2, euler_subkey, resample_subkey = random.split(s.key2,3)
            s.temp_state, s.temp_time = EulerMaruyama(euler_subkey, param, \
                s.z, s.temp_time, dt, P, N, D)  
            s.z = s.temp_state[:,-1,:]
            s.t = s.t.at[i].set(s.temp_time)
            s.w = jnp.exp(dist.Normal(s.z,10.).log_prob(y[i+1,:]).sum(axis=1))
            s.mLik += jnp.log(s.w.mean())
            s.wt = s.w/jnp.sum(s.w)
            s.particles = s.z
            
            s.ind = random.choice(resample_subkey, a=P, shape=(P,), p=s.wt)
            s.particles = s.particles[s.ind,:]
            s.X = s.X.at[i,:,:].set(s.particles)
    return s.mLik  

class LogPosterior(object):
    def __init__(self, data, init_val, num_particles=500, num_steps=9, \
        num_euler_steps=51, dt=0.1, transform=False):

        self._y = data
        self._P = num_particles
        self._D = 2
        self._dt = dt
        self._N = num_euler_steps
        self._num_steps = num_steps
        self._init_val = init_val

        self.n_params = 3
        self._transform = transform       
 

    def _transform_to_constraint(self, transformed_parameters):
        Tx_THETA = transformed_parameters
        Utx_c1  = np.exp(Tx_THETA[0])
        Utx_c2  = np.exp(Tx_THETA[1])
        Utx_c3  = np.exp(Tx_THETA[2])
        return np.array([Utx_c1, Utx_c2, Utx_c3])

    def _transform_from_constraint(self, untransformed_parameters):
        
        Utx_THETA = untransformed_parameters
        tx_c1  = np.log(Utx_THETA[0])  
        tx_c2  = np.log(Utx_THETA[1])
        tx_c3  = np.log(Utx_THETA[2])
        return np.array([tx_c1, tx_c2, tx_c3])

    def __call__(self, parameters, key):
        
        if self._transform:
            _Tx_THETA = parameters.copy()
            THETA = self._transform_to_constraint(_Tx_THETA)
            _Tx_c1  = _Tx_THETA[0]
            _Tx_c2  = _Tx_THETA[1]
            _Tx_c3  = _Tx_THETA[2]
        else:
            THETA = parameters.copy()

        c1  = THETA[0]
        c2  = THETA[1]
        c3  = THETA[2]
        theta_scaled = THETA
        theta_scaled[1] = theta_scaled[1]/100
        theta = device_put(theta_scaled)

        log_likelihood = BootStrapp(key, self._y, theta, self._init_val, self._dt, \
            self._P, self._N, self._D, self._num_steps)

        if self._transform:
            logPrior_c1 = stats.beta.logpdf(c1,a=2.,b=1.) + _Tx_c1     
            logPrior_c2 = stats.halfnorm.logpdf(c2,scale = 1.) + _Tx_c2   
            logPrior_c3 = stats.beta.logpdf(c3,a=1.,b=2.) + _Tx_c3   
     
        else:
            logPrior_c1 = stats.beta.logpdf(c1,a=2.,b=1.) 
            logPrior_c2 = stats.halfnormal.logpdf(c2,scale = 1.)    
            logPrior_c3 = stats.beta.logpdf(c3,a=1.,b=2.) 
     

        log_prior = logPrior_c1 + logPrior_c2 + logPrior_c3 
        return log_likelihood + log_prior
                   
    def n_parameters(self):
        return self.n_params

            
        


