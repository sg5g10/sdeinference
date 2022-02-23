import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import seaborn as sns
import jax.numpy as jnp
from jax.experimental.ode import odeint
from jax import random
import numpyro
import numpyro.distributions as dist   
import time as timer
from models.logPDFou import LogPosterior
from mcmc.adaptiveMetropolis import AdaptiveMetropolis
from mcmc.block_mcmc import MCMC
from models.saode_ou import SAODE_OU
from npr_inference.run_inference import run_nuts, run_vi

def numpyro_model(data=None, solver=None, num_bases=10, solver_times=None):

        num_states = 1
        num_coeffs = num_bases*num_states
        # initial population
        z_init = numpyro.sample("z_init", dist.Normal(0,1))
        # measurement times
        ts = solver_times
        # priors

        c1 = numpyro.sample("c1", dist.Gamma(2,2))
        c2 = numpyro.sample("c2", dist.Gamma(2,2))
        c3 = numpyro.sample("c3", dist.Gamma(2,2))
        coeffs = numpyro.sample("coeffs", dist.Normal(0,1).expand([num_coeffs]))
        theta = jnp.array([c1,c2,c3,*coeffs])
        # solve SA-ODE
        x = odeint(solver, z_init, ts, theta, rtol=1e-5, atol=1e-5, mxstep=1000)
        mu = numpyro.deterministic("mu", x)
        sigma = numpyro.sample("sigma", dist.HalfNormal(1))
        # likelihood
        y= numpyro.sample("y", dist.Normal(x, sigma), obs=data)
        return y

def main():
        
        time=jnp.arange(0,10,.1)
        Y=np.loadtxt('./data/ou_data.txt')
        logP = LogPosterior(Y, num_particles=args.pmmh_nparticles, transform=True)
        X0 = np.array([0.8383164,0.8383164,0.8383164,0.,0.67487067])
        X0 = logP._transform_from_constraint(X0)
        Init_scale = 1*np.abs(X0)
        cov = None
        # Now run the AMGS sampler
        sampler = AdaptiveMetropolis(logP, mean_est=X0, cov_est=cov, tune_interval = 1)
        t0 = timer.time()
        MCMCsampler = MCMC(sampler, logP, X0, True, args.pmmh_iters)
        trace, sample_path_pm = MCMCsampler.run(random.PRNGKey(1), X0)
        t1 = timer.time()
        total = t1-t0
        print(total)
        
        pmmh_trace = trace[args.pmmh_warmup:,:]
        pmmh_trace_thin = pmmh_trace[::args.pmmh_thin,:]
        print('PMMH estimates: ', np.mean(pmmh_trace_thin, axis=0))
        

        # Now run VI with SA-ODE with KL expansion
        n = args.num_qsamples
        times = jnp.arange(0,10,.1)
        saode = SAODE_OU(args.end_point, num_bases=args.num_bases)
        vb_ppc_samples, sample_path_vb = run_vi(Y, numpyro_model, saode, random.PRNGKey(3), \
        num_bases = args.num_bases, iterations=args.vi_iters, ppc_n=args.num_qsamples, \
                solver_times=times, batchsize=10)
        
        vb_params=jnp.concatenate((vb_ppc_samples['c1'].reshape((n,1)),
                        vb_ppc_samples['c2'].reshape((n,1)),
                        vb_ppc_samples['c3'].reshape((n,1)),
                        vb_ppc_samples['z_init'].reshape((n,1)),
                        vb_ppc_samples['sigma'].reshape((n,1))),axis=1)
        print('VI estimates: ', np.mean(vb_params, axis=0))
        path_pm = np.array(sample_path_pm)[-20000:,::2,:].squeeze()
        path_pm = path_pm[::20,:]
        mean_ppc_pm = np.percentile(path_pm,q=50,axis=0)
        CriL_ppc_pm = np.percentile(path_pm,q=2.5,axis=0)
        CriU_ppc_pm = np.percentile(path_pm,q=97.5,axis=0)
        

        path_vb = sample_path_vb["mu"]
        mean_ppc_vb = np.percentile(path_vb,q=50,axis=0)
        CriL_ppc_vb = np.percentile(path_vb,q=2.5,axis=0)
        CriU_ppc_vb = np.percentile(path_vb,q=97.5,axis=0)


        times = np.arange(0,10,0.1)
        sns.set_context("paper", font_scale=1)
        sns.set(rc={"figure.figsize":(14,6),"font.size":26,"axes.titlesize":30,"axes.labelsize":30,\
                "xtick.labelsize":25, "ytick.labelsize":25},style="white")
        plt.subplot(1,2,1)
        plt.plot(times,mean_ppc_pm, color='blue', lw=2, label='PMMH')
        plt.plot(times,CriL_ppc_pm, '--', color='blue', lw=1.5)
        plt.plot(times,CriU_ppc_pm, '--',  color='blue', lw=1.5)
        plt.plot(times,Y,color='k', lw=1.5, label='Data')
        plt.legend(fontsize=25)
        plt.ylabel('State',fontsize=30)
        plt.xlabel('times',fontsize=30)
        plt.tight_layout()
        
        plt.subplot(1,2,2)
        plt.plot(times,mean_ppc_vb, color='blue', lw=2, label='VI')
        plt.plot(times,CriL_ppc_vb, '--', color='blue', lw=1.5)
        plt.plot(times,CriU_ppc_vb,  '--', color='blue', lw=1.5)
        plt.plot(times,Y,color='k', lw=1.5)
        plt.legend(fontsize=25)
        plt.ylabel('State',fontsize=30)
        plt.xlabel('times',fontsize=30)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.8)
        plt.show() 

if __name__ == '__main__':

        parser = argparse.ArgumentParser(
                description='Fit OU SDE')
        parser.add_argument('--vi_iters', type=int, default=2000, metavar='N',
                        help='number of VI iterations') 
        parser.add_argument('--num_qsamples', type=int, default=1000, metavar='N',
                        help='number of draws from variational posterior ')        
        parser.add_argument('--pmmh_iters', type=int, default=100000, metavar='N',
                        help='number of PMMH iterations')    
        parser.add_argument('--pmmh_warmup', type=int, default=50000, metavar='N',
                        help='number of PMMH warmup_steps')  
        parser.add_argument('--pmmh_thin', type=int, default=50, metavar='N',
                        help='thinning ratio')                          
        parser.add_argument('--pmmh_nparticles', type=int, default=100, metavar='N',
                        help='number of particles to go with PMMH') 
        parser.add_argument('--num_bases', type=int, default=10, metavar='N',
                        help='number of coefficients for saode') 
        parser.add_argument('--end_point', type=int, default=10, metavar='N',
                        help='the value of T for the KL expansion')  
                                              
                                                          
        args = parser.parse_args()
        main()

