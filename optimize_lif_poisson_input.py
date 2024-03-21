import numpy as np
import matplotlib

matplotlib.rcParams["font.size"] = 8
import matplotlib.pyplot as plt
from functools import partial
import jax, optax
import jax.lax as lax
import jax.numpy as jnp
from mesogif_jax import (
    Params,
    StaticParams,
    State,
    integrate_log_prob,
    update_state,
    compute_current_propagators,
    compute_current_voltage_propagators,
    compute_voltage_propagators,
    simulate,
)
import pickle

###############
## Simulate
###############
steps = 5000
M = 2
K = 200
synport = np.array([[0], [1]], dtype=np.int32)
staticparams = StaticParams(
    synport=synport,
    u_reset=np.array(0.0, dtype=np.float32),
    dt=0.5,
    num_ref=4,
    delay=2.0,
)

w = np.ones((M, 2), dtype=jnp.float32)
w[:, 0] = 10.0 * np.exp(1) * 0.1
w[:, 1] = -50.0 * np.exp(1) * 0.1
tau_s = np.ones((M, 2), dtype=np.float32)
tau_s[:, 0] = 3.0
tau_s[:, 1] = 3.0
N = np.zeros(2, dtype=np.float32)
N[0] = 800
N[1] = 200
params = Params(
    RI=jnp.ones(M, dtype=jnp.float32) * 27.0,
    tau_m=jnp.ones(M, dtype=jnp.float32) * 20.0,
    N=jnp.array(N),
    tau_s=jnp.array(tau_s),
    u_thr=jnp.ones(M, dtype=jnp.float32) * 20.0,
    c=jnp.ones(M, dtype=jnp.float32) * 0.0,
    delta_u=jnp.ones(M, dtype=jnp.float32) * 1.0,
    C_mem=jnp.ones(M, dtype=jnp.float32) * 250.0,
    w=jnp.array(w),
)

rec = simulate(params, staticparams, K, steps, record=["u", "h", "y", "spikes", "m"])

###############
# Optimize
###############
spikes = np.load("spikes1.npy").astype(np.float32)[500:]

m_init = np.zeros((M, K), dtype=np.float32)
m_init[:, staticparams.num_ref + 1] = np.ones(M, dtype=np.float32) * params.N
m_init = jnp.array(m_init)


def optimize(params, spikes, m_init):
    update_state_partial = partial(update_state, staticparams=staticparams)
    update_state_vec = jax.vmap(update_state_partial, in_axes=(0, 0, None))

    # set up vectorization
    update_state_partial = partial(update_state, staticparams=staticparams)
    update_state_vec = jax.vmap(update_state_partial, in_axes=(0, 0, None))

    # functions to integrate
    def integrate(initial_state, params, spikes):
        carry = (initial_state, params)

        def update_state_wrap(carry, spikes):
            state, log_prob = update_state_vec(carry[0], carry[1], spikes)
            return (state, carry[1]), log_prob

        carry, log_probs = lax.scan(update_state_wrap, carry, spikes)
        return carry, log_probs

    def integrate_log_prob(params, initial_state, spikes, num_burn_in):
        _, log_probs = integrate(initial_state, params, spikes)
        return -log_probs[num_burn_in:].sum()

    def compute_log_prob(params, spikes, m_init, num_burn_in):
        # set up state

        current_propagators = compute_current_propagators(params.tau_s, staticparams)
        current_voltage_propagators = compute_current_voltage_propagators(
            params.tau_s, params.tau_m, params.C_mem, staticparams
        )
        voltage_propagators = compute_voltage_propagators(params.tau_m, staticparams)
        state = State(
            y=jnp.zeros((M, 2, 2), dtype=jnp.float32),
            u=jnp.zeros((M, K), dtype=jnp.float32),
            v=jnp.zeros((M, K), dtype=jnp.float32),
            current_prop=current_propagators,
            current_voltage_prop=current_voltage_propagators,
            voltage_prop=voltage_propagators,
            m=m_init,
            lambd_old=jnp.zeros((M, K), dtype=jnp.float32),
            h=jnp.zeros((M), dtype=jnp.float32),
            x=jnp.zeros((M), dtype=jnp.float32),
            z=jnp.zeros((M), dtype=jnp.float32),
            lambd_free=jnp.zeros((M), dtype=jnp.float32),
            index=jnp.arange(M),
        )

        log_prob = integrate_log_prob(params, state, spikes, num_burn_in)

        return log_prob

    # grad and jit
    value_grad_log_prob = jax.value_and_grad(compute_log_prob)
    value_grad_log_prob = jax.jit(value_grad_log_prob, static_argnums=(3,))

    optimizer = optax.adam(0.01)
    # Obtain the `opt_state` that contains statistics for the optimizer.
    opt_state = optimizer.init(params)
    paramlist = []
    log_probs = []
    for i in range(300):
        value, grads = value_grad_log_prob(params, spikes, m_init, 250)
        log_probs.append(value)
        grads.N = jnp.zeros(2, dtype=jnp.float32)
        # log_probs = compute_log_prob(params, spikes, m_init, 250)
        # return log_probs
        print(f"{i:04d}: {value:.0f}")
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        paramlist.append(params)
    return paramlist, log_probs


ps, log_probs = optimize(params, spikes, m_init)
RIs = np.array([np.asarray(p.RI) for p in ps])

#################################
# Simulate optimized parameters
#################################

nparams = ps[-1]
nrec = simulate(
    nparams,
    staticparams,
    K,
    steps,
    record=["u", "h", "y", "spikes", "m", "x", "lambd_old"],
)

from scipy.signal import welch

fs, psd1 = welch(spikes[-4000:].T, fs=2000)
fs, psd2 = welch(rec["spikes"][-4000:].T, fs=2000)
fs, psd3 = welch(nrec["spikes"][-4000:].T, fs=2000)
fig, ax = plt.subplots(ncols=2, nrows=3)
fig.set_size_inches([10, 6])
fig.subplots_adjust(hspace=0.5)
fig.suptitle("Poisson input, $J_{ext} = 5.$")
ax[0, 0].plot(spikes)
ax[1, 0].plot(rec["spikes"])
ax[2, 0].plot(nrec["spikes"])
ax[0, 1].plot(fs, psd1.T)
ax[1, 1].plot(fs, psd2.T)
ax[2, 1].plot(fs, psd3.T)
ax[0, 0].set_title(f"target, mean = {spikes.mean(0)}")
ax[1, 0].set_title(f'unoptimized mesoscopic, mean = {rec["spikes"].mean(0)}')
ax[2, 0].set_title(f'optimized mesoscopic, mean = {nrec["spikes"].mean(0)}')
# fig.savefig('glif_optimization_5.pdf')
plt.show()
