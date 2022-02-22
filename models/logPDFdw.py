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

@partial(jit, static_argnums=(4,5,6))
def EulerMaruyama(key, param, init_val, t, dt, P, N):
    D=1
    with loops.Scope() as s:
        s.beta1 = param[0]
        s.beta2 = param[1]
    
        s.dt = dt
        s.z = jnp.zeros((P,N+1,D))
        s.t = t
        s.z = s.z.at[:,0,:].set(init_val)
        s.dw = jnp.zeros(P)
        s.subkey = key
        for i in s.range(N):
            key, s.subkey = random.split(s.subkey,2)
            s.dw = random.normal(s.subkey, shape=(P,)) #
            s.z = s.z.at[:,i+1,0].set(s.z[:,i,0] + ((4*s.z[:,i,0]*(s.beta1 - s.z[:,i,0]**2))*s.dt) +\
                 (jnp.sqrt(s.dt)*s.beta2*s.dw))
               
            s.t = s.t + s.dt
    zar = s.z
    return zar[:,1:,:], s.t

@partial(jit, static_argnums=(3,4,5))
def SDElikelihood(key, y, param, dt, P, obs_interval):
    N = int(((obs_interval + dt) /dt)) - 1
    D = 1
    num_steps = y.shape[0]
    with loops.Scope() as s:

        s.zz = jnp.ones((P,D))*param[-2]
        s.X = jnp.zeros((P,num_steps,N,D))
        s.temp_state = jnp.zeros((P,N,D))

        s.w = jnp.exp(dist.Normal(s.zz,param[-1]).log_prob(y[0])).squeeze()
        s.mLik = jnp.log(s.w.mean())
        s.wt = s.w/jnp.sum(s.w)
        s.ind = random.choice(key, a=P, shape=(P,), p=s.wt)
        s.zz = s.zz[s.ind,:]

        s.t = jnp.zeros(num_steps)
        s.key2 = key

        s.temp_time = 0.
        for i in s.range(num_steps):      

            s.key2, euler_subkey, resample_subkey = random.split(s.key2,3)
            s.temp_state, s.temp_time = EulerMaruyama(euler_subkey, param[:2], s.zz, s.temp_time, dt, P, N)  
            s.zz = s.temp_state[:,-1,:]
            s.t = s.t.at[i].set(s.temp_time)
            s.w = jnp.exp(dist.Normal(s.zz,param[-1]).log_prob(y[i+1])).squeeze()
            s.mLik += jnp.log(s.w.mean())
            s.wt = s.w/jnp.sum(s.w)
    
            s.ind = random.choice(resample_subkey, a=P, shape=(P,), p=s.wt)
            s.zz = s.zz[s.ind,:]
            s.X = s.X.at[:,i,:,:].set(s.temp_state[s.ind,...])
            
        s.X =  s.X.reshape((P,num_steps*N,D),order='C')  
    return s.mLik , s.X[0,...]

class LogPosterior(object):
    def __init__(self, data, num_particles=500, obs_interval=0.1, dt=0.05, \
        transform=False):

        self._y = data
        self._P = num_particles
        self._D = 4
        self._dt = dt
        self._obs_interval = obs_interval
        self.n_params = 4
        self._transform = transform       
 

    def _transform_to_constraint(self, transformed_parameters):
        Tx_THETA = transformed_parameters
        Utx_beta1  = np.exp(Tx_THETA[0]) # Ga(2,1)
        Utx_beta2  = np.exp(Tx_THETA[1]) # Ga(2,1)   
        Utx_x0  = Tx_THETA[2] # N()  
        Utx_sigm = np.exp(Tx_THETA[3])
        return np.array([Utx_beta1, Utx_beta2, \
            Utx_x0, Utx_sigm])

    def _transform_from_constraint(self, untransformed_parameters):
        
        Utx_THETA = untransformed_parameters
        tx_beta1  = np.log(Utx_THETA[0])
        tx_beta2  = np.log(Utx_THETA[1])
        tx_x0  = Utx_THETA[2]
        tx_sigm = np.log(Utx_THETA[3])
        return np.array([tx_beta1, tx_beta2, \
            tx_x0, tx_sigm])

    def __call__(self, parameters, key):
        
        if self._transform:
            _Tx_THETA = parameters.copy()
            THETA = self._transform_to_constraint(_Tx_THETA)
            _Tx_beta1  = _Tx_THETA[0]
            _Tx_beta2  = _Tx_THETA[1]
            _Tx_x0  = _Tx_THETA[2]
            _Tx_sigm = _Tx_THETA[3]

        else:
            THETA = parameters.copy()

        beta1  = THETA[0]
        beta2  = THETA[1]
        x0  = THETA[2]
        sigm = THETA[3]

        theta_scaled = THETA
        theta = device_put(theta_scaled)

        log_likelihood, X = SDElikelihood(key, self._y, theta, self._dt, \
            self._P, self._obs_interval)

        if self._transform:
            logPrior_beta1 = stats.gamma.logpdf(beta1,a=2.,scale=1/2) + _Tx_beta1     
            logPrior_beta2 = stats.gamma.logpdf(beta2,a=2, scale = 1/2) + _Tx_beta2 
            logPrior_x0 = stats.norm.logpdf(x0,loc=0.,scale=1)
            logPrior_sigm = stats.halfnorm.logpdf(sigm,scale=1) + _Tx_sigm


        log_prior = logPrior_beta1 + logPrior_beta2 +\
             logPrior_x0 + logPrior_sigm

        return log_likelihood + log_prior, X
                   
    def n_parameters(self):
        return self.n_params

            
        


