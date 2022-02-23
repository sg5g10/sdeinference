#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import scipy.stats as stats
import jax.numpy as jnp
from jax.experimental.ode import odeint
from jax import random
import numpyro
import numpyro.distributions as dist
import arviz as az
from joblib import Parallel, delayed
import multiprocessing
import time as timer
from models.logPDFsir import LogPosterior
from mcmc.adaptiveMetropolis import AdaptiveMetropolis
from mcmc.block_mcmc import MCMC
from models.saode_sir import SAODE_SIR
from npr_inference.run_inference import run_nuts, run_vi

def mcmc_runner(init, logprob, iters):
        sampler = AdaptiveMetropolis(logprob, mean_est=init, cov_est=None, tune_interval = 1)
        MCMCsampler = MCMC(sampler, logprob, init, iterations=iters)  
        return MCMCsampler.run(random.PRNGKey(1),init)


def numpyro_model(data=None, solver=None, num_bases=10):
        num_states = 2
        num_coeffs = num_bases*num_states
        # initial population
        s0 = numpyro.sample("s0", dist.Beta(2,1))
        # measurement times
        ts = jnp.array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12.,
        13.])
        # parameters alpha, beta        
        beta_gamma = numpyro.sample("beta_gamma", dist.Gamma(2,2).expand([2]))
        z = numpyro.sample("z", dist.Normal(0,1).expand([num_coeffs]))

        theta = jnp.array([*beta_gamma,*z])
        x_init = jnp.array([s0,1-s0])
        # solve SA-ODE        
        x = odeint(solver, x_init, ts, theta, rtol=1e-8, atol=1e-7, mxstep=1000)[:,1]
        # likelihood
        y = numpyro.sample("y", dist.Poisson(x*763), obs=data)
        return y

def main():

        # load data (Real)
        Y= jnp.array([3,8,26,76,225,298,258,233,189,128,68,29,14,4])
        
        # setup for PMMH
        logP = LogPosterior(Y, num_particles=args.pmmh_nparticles, transform=True)
        x0s = []
        start = [1,0.5,760/763]#.55
        X0 = np.hstack(start)
        X0 = logP._transform_from_constraint(X0)
        x0s.append(X0)
        start = [1,0.55,759/763]#1.65
        X0 = np.hstack(start)
        X0 = logP._transform_from_constraint(X0)
        x0s.append(X0)

        # run PMMH
        t0=timer.time()
        iterations = args.pmmh_iters
        burnin = args.pmmh_warmup
        thin = int((iterations - burnin)/args.nuts_iters) # to get same sample size as nuts/vi
        chains_amgs = Parallel(n_jobs=2)(delayed(mcmc_runner)(start, logP, \
        iterations) for start in x0s)
        t1=timer.time()
        total = t1-t0
        print('Time', total)
        # Report PMMH ESS and save chains/params
        dict_amgs = {'theta': np.array(chains_amgs)[:,burnin:,:]}
        data_amgs = az.convert_to_inference_data(dict_amgs)
        amgs_ess = az.ess(data_amgs,relative=True).to_dataframe()['theta'].mean()
        amgstrace_post_burn = np.array(chains_amgs)[:,burnin:,:]
        py_thinned_trace_amgs = amgstrace_post_burn[:,::thin*2,:].reshape((int((iterations-burnin)/thin),3))
        print('SIR PMMH (relative) ESS: ', amgs_ess)
        pm_params = py_thinned_trace_amgs
        
        # Now run VI with SA-ODE with KL expansion
        n = args.num_qsamples
        saode = SAODE_SIR(args.end_point, num_bases=args.num_bases)
        vb_ppc_samples = run_vi(Y, numpyro_model, saode, random.PRNGKey(3), \
        num_bases=args.num_bases, iterations=args.vi_iters, ppc_n=args.num_qsamples)
        vb_params=np.concatenate((vb_ppc_samples['beta_gamma'].reshape((n,2)),\
                          vb_ppc_samples['s0'].reshape((n,1))),axis=1)

        # Now run NUTS with SA-ODE with KL expansion
        n = int(args.nuts_iters)
        mc_ppc_samples = run_nuts(Y, numpyro_model, saode, random.PRNGKey(10), \
        num_bases=args.num_bases, iterations=args.nuts_iters, warmup=args.nuts_warmup)[0]

        mc_params=np.concatenate((mc_ppc_samples['beta_gamma'][::2].reshape((n,2)),
                          mc_ppc_samples['s0'][::2].reshape((n,1))                                             
                          ),axis=1)
        sns.set_context("paper", font_scale=1)
        sns.set(rc={"figure.figsize":(14,6),"font.size":16,"axes.titlesize":16,"axes.labelsize":16,
           "xtick.labelsize":15, "ytick.labelsize":15},style="white")
        param_names = [r"$\beta$",r"$\gamma$", r"$s_0$"]

        for i, p in enumerate(param_names):        
                plt.subplot(1, 3, i+1)
                
                if i==0:

                        sns.kdeplot(pm_params[:, i], color='green', linewidth = 2.5, label='PMMH')
                        sns.kdeplot(vb_params[:, i], color='magenta', linewidth = 2.5, label='VI')
                        sns.kdeplot(mc_params[:, i], color='orange', linewidth = 2.5, label='NUTS')
                        plt.ylabel('Frequency')
                        plt.xlabel(param_names[i])  
                else:
                        sns.kdeplot(pm_params[:, i], linewidth = 2.5, color='green')
                        sns.kdeplot(vb_params[:, i], linewidth = 2.5, color='magenta')
                        sns.kdeplot(mc_params[:, i], linewidth = 2.5, color='orange')  

                        plt.ylabel('Frequency')
                        plt.xlabel(param_names[i])        
                if i<1:
                        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower center', ncol=2,fontsize=18)
        plt.subplots_adjust(hspace=0.7)
        plt.tight_layout()
        plt.show()        


if __name__ == '__main__':

        parser = argparse.ArgumentParser(
                description='Fit SIR model')
        parser.add_argument('--vi_iters', type=int, default=30000, metavar='N',
                        help='number of VI iterations') 
        parser.add_argument('--num_qsamples', type=int, default=1000, metavar='N',
                        help='number of draws from variational posterior ')  
        parser.add_argument('--nuts_iters', type=int, default=1000, metavar='N',
                        help='number of NUTS post warm-up samples')    
        parser.add_argument('--nuts_warmup', type=int, default=1000, metavar='N',
                        help='number of NUTS warmup_steps')       
        parser.add_argument('--pmmh_iters', type=int, default=100000, metavar='N',
                        help='number of PMMH iterations')    
        parser.add_argument('--pmmh_warmup', type=int, default=50000, metavar='N',
                        help='number of PMMH warmup_steps')  
        parser.add_argument('--pmmh_nparticles', type=int, default=100, metavar='N',
                        help='number of particles to go with PMMH')  
        parser.add_argument('--num_bases', type=int, default=10, metavar='N',
                        help='number of coefficients for saode') 
        parser.add_argument('--end_point', type=int, default=14, metavar='N',
                        help='the value of T for the KL/Wavelet expansion')  
                                              
                                                          
        args = parser.parse_args()
        numpyro.set_host_device_count(multiprocessing.cpu_count())
        main()




