from jax import random
import numpyro
from numpyro.infer import SVI, Trace_ELBO, MCMC, NUTS, Predictive, init_to_median
from numpyro.infer.autoguide import AutoMultivariateNormal
from jax import device_get, device_put


def run_vi(data, model, saode_solver, rng_key, \
num_bases=10, iterations=20000, step_size=1e-3, ppc_n=1000, solver_times=None, batchsize=1):
    print('Running VI')
    inference_key, ppc_key = random.split(rng_key,2)
    #random.PRNGKey(10)
    if batchsize == 1:
        guide = AutoMultivariateNormal(model, init_loc_fn=init_to_median)
        optimizer = numpyro.optim.RMSProp(step_size=step_size)
        svin = SVI(model, guide, optimizer, loss=Trace_ELBO())
        svi_result = svin.run(inference_key, iterations, data=device_put(data),\
        solver=saode_solver, num_bases=num_bases, stable_update=True, progress_bar=True)  
        vb_ppc_samples = device_get(guide.sample_posterior(ppc_key, \
        svi_result.params, sample_shape=(ppc_n,1)))      
        return vb_ppc_samples
    else:
        guide = AutoMultivariateNormal(model)
        optimizer = numpyro.optim.RMSProp(step_size=1e-2)        
        svin = SVI(model, guide, optimizer, loss=Trace_ELBO(num_particles=batchsize))
        svi_result = svin.run(inference_key, iterations, data=device_put(data),\
        solver=saode_solver, num_bases=num_bases, solver_times=solver_times, \
        stable_update=True, progress_bar=True)
    vb_ppc_samples = device_get(guide.sample_posterior(ppc_key, \
    svi_result.params, sample_shape=(ppc_n,1)))
    pred = Predictive(model, guide=guide, params = svi_result.params, num_samples=ppc_n)
    sample_path = pred(random.PRNGKey(10), data=device_put(data),\
        solver=saode_solver, num_bases=num_bases, solver_times=solver_times)
    return vb_ppc_samples, sample_path

def run_nuts(data, model, saode_solver, rng_key, \
num_bases=10, iterations=1000, warmup=1000, num_chains=2, target_accept_prob=0.8):
    print('Running NUTS')
    MCMC_KWARGS = dict(
    num_warmup=1000,
    num_samples=1000,
    num_chains=2,
    chain_method="parallel",
    )
    nuts_kernel = NUTS(model, dense_mass=True, init_strategy=init_to_median)
    mcmc = MCMC(nuts_kernel,**MCMC_KWARGS)
    mcmc.run(rng_key, data, saode_solver, num_bases, \
    extra_fields=('potential_energy', 'energy', 'num_steps') )
    
    mcmc.print_summary(exclude_deterministic=False)
    
    return mcmc.get_samples(), mcmc