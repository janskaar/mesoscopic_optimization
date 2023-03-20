from typing import Union
from functools import partial
import jax
import jax.numpy as jnp
from jax.scipy.special import erf
from jax import lax
import numpy as np
import chex

import matplotlib.pyplot as plt

@chex.dataclass
class Params:
    N:        jnp.ndarray
    tau_m:    jnp.ndarray
    tau_s:    jnp.ndarray
    u_thr:    jnp.ndarray
    C_mem:    jnp.ndarray
    RI:       jnp.ndarray
    w:        jnp.ndarray

@chex.dataclass(frozen=True)
class StaticParams:
    synport:   np.ndarray
    u_reset:   np.ndarray
    sigma_noise: float
    tau_noise: float
    delay:     float
    num_steps: int = 10000
    dt:        Union[int, float] = 0.1
    num_ref:   int = 50

@chex.dataclass
class State:
    current_prop:           jnp.ndarray
    current_voltage_prop:   jnp.ndarray
    voltage_prop:           jnp.ndarray
    m:                      jnp.ndarray
    h:                      jnp.ndarray
    x:                      jnp.ndarray
    z:                      jnp.ndarray
    lambd_free:             jnp.ndarray
    u:                      jnp.ndarray
    v:                      jnp.ndarray
    lambd_old:              jnp.ndarray
    y:                      jnp.ndarray
    index:                  jnp.ndarray
    rndunif:          Union[jnp.ndarray, None] = None
    spikes:           Union[jnp.ndarray, None] = None

def compute_current_propagators(tau_s, staticparams):
    t1 = jnp.exp(-staticparams.dt / tau_s)
    return t1

compute_current_propagators = jax.vmap(compute_current_propagators, in_axes=(0, None)) # over synaptic ports
compute_current_propagators = jax.vmap(compute_current_propagators, in_axes=(0, None)) # over populations

def compute_current_voltage_propagators(tau_s, tau_m, C_mem, staticparams):
    t0 = tau_m * tau_s * (jnp.exp(-staticparams.dt/tau_m) - jnp.exp(-staticparams.dt / tau_s))
    denom = C_mem * (tau_m - tau_s)
    return t0/denom

# vmap over synaptic ports
compute_current_voltage_propagators \
    = jax.vmap(compute_current_voltage_propagators, in_axes=(0, None, None, None))

# vmap over populations
compute_current_voltage_propagators \
    = jax.vmap(compute_current_voltage_propagators, in_axes=(0, 0, 0, None))

def compute_voltage_propagators(tau_m, staticparams):
    t0 = jnp.exp(-staticparams.dt / tau_m)
    t1 = 1 - t0
    return jnp.stack((t0, t1), axis=-1)

def roll_buffers(u, lambd_tilde, v, m):
    u = jnp.roll(u, 1)
    v = jnp.roll(v, 1)
    m = jnp.roll(m, 1)
    lambd_tilde = jnp.roll(lambd_tilde, 1)
    return u, lambd_tilde, v, m

def cg_escape(u, noise, C_m, tau_m, RI, tau_noise, V_th):
    k = tau_m / tau_noise
    sigma_u = 1. / jnp.sqrt(1. + k)
    R = tau_m / C_m
    T = ((1 + k)/2)**0.5 * (V_th-u) / (R * noise)
    U_dot = (-u + RI) / tau_m
    T_dot = -1. / (R * noise * 2.**0.5 * sigma_u) * U_dot
    F = (2. / jnp.pi)**0.5 * jnp.exp(-T**2.) / (1. + erf(T))
    B = -2.**0.5 * tau_m * jnp.clip(T_dot, a_max=0.) * F
    A_inf = jnp.exp(6.1e-3 - 1.12*T - 0.257*T**2 - 0.072*T**3 - 0.0117*T**4)
    A = A_inf * (1. - (1 + k)**(-0.71 + 0.0825*(T+3.)))
    lam = (A + B) / tau_m
    return lam

vcg_escape = jax.vmap(cg_escape, in_axes=(0, None, None, None, None, None, None))

# def exponential_escape(x, c, delta):
#     return jnp.exp(c + x / delta)
# 
# vexponential_escape = jax.vmap(exponential_escape, in_axes=(0, None, None))

def compute_current_contrib(current_voltage_prop, y):
    return current_voltage_prop * y

vcompute_current_contrib = jax.vmap(compute_current_contrib)

def compute_passive_contrib(voltage_prop, u, RI_e):
    vec = jnp.stack((u, RI_e))
    return voltage_prop.dot(vec)

def update_u(current_voltage_prop, voltage_prop, y, u, RI_e):
    # gives contribution from 2 synaptic ports, therefore sum()
    current_contrib = vcompute_current_contrib(current_voltage_prop, y).sum()

    # gives updated membrane potential after depolarization and constant current input
    passive_contrib = compute_passive_contrib(voltage_prop, u, RI_e)
    return current_contrib + passive_contrib

vupdate_u = jax.vmap(update_u, in_axes=(None, None, None, 0, None))

def update_synapses(current_prop, y, s):
    y1 = current_prop * y
    y1 = y1 + s
    return y1

vupdate_synapses = jax.vmap(update_synapses, in_axes=(0, 0, 0))

def update_state(state: State, params: Params, spikes: jnp.ndarray, staticparams: StaticParams) -> State:
    print('TRACING')
    def P_fn(P_lambd):
        return 1-jnp.exp(-P_lambd)

    # upate currents
    # synport[i] gives indices of all populations connected to synaptic port i
    s0 = params.w[staticparams.synport[0]].dot(
                            spikes[staticparams.synport[0]])
    s1 = params.w[staticparams.synport[1]].dot(
                            spikes[staticparams.synport[1]])
    s = jnp.stack((s0, s1)) # additions to synaptic ports 0 and 1
    y = vupdate_synapses(state.current_prop, state.y, s)

    X = state.m.sum()

    # free neurons
    h = update_u(state.current_voltage_prop,
                  state.voltage_prop,
                  y,
                  state.h,
                  params.RI)

    lambd_tilde_free = cg_escape(h,
                    staticparams.sigma_noise,
                    params.C_mem,
                    params.tau_m,
                    params.RI,
                    staticparams.tau_noise,
                    params.u_thr)


#    lambd_tilde_free = exponential_escape(h - params.u_thr, params.c, params.delta_u)
    P_free = 1 - jnp.exp(-0.5 * (lambd_tilde_free + state.lambd_free) * staticparams.dt)

    # refractory neurons
    # grab non-refractory indices
    u = state.u[staticparams.num_ref:]
    v = state.v[staticparams.num_ref:]
    m = state.m[staticparams.num_ref:]
    lambd_old = state.lambd_old[staticparams.num_ref:]

    # update membrane potentials
    u = vupdate_u(state.current_voltage_prop,
                  state.voltage_prop,
                  y,
                  u,
                  params.RI)

    # compute escape rate
#    lambd_tilde = vexponential_escape(u - params.u_thr, params.c, params.delta_u)
    lambd_tilde = vcg_escape(u,
                    staticparams.sigma_noise,
                    params.C_mem,
                    params.tau_m,
                    params.RI,
                    staticparams.tau_noise,
                    params.u_thr)


    P_lambd = 0.5 * (lambd_tilde + state.lambd_old[staticparams.num_ref:]) * staticparams.dt
    P_lambd = P_fn(P_lambd)

    Y = P_lambd.dot(v)
    Z = v.sum()

    Pm = P_lambd * m
    W = Pm.sum()

    neg_P_lambd = 1 - P_lambd
    v = (neg_P_lambd)**2 * v + Pm
    m = (neg_P_lambd) * m

    def P_fn_true(Y, P_free, Z, z):
        return (Y + P_free * z) / (Z + z)

    def P_fn_false(Y, P_free, Z, z):
        return 0.

    z = state.z
    x = state.x
    P_Lambd = lax.cond(Z + z > 0, P_fn_true, P_fn_false, Y, P_free, Z, z)

    z = (1 - P_free)**2 * z + P_free * x + v[-1]
    x = (1 - P_free) * x + m[-1]

    n_bar = W + P_free * x + P_Lambd * (params.N - X - x)
    p = jnp.clip(n_bar / params.N, 1e-5, 1-1e-5)
    u = state.u.at[staticparams.num_ref:].set(u)
    v = state.v.at[staticparams.num_ref:].set(v)
    m = state.m.at[staticparams.num_ref:].set(m)

    m = m.at[-1].set(spikes[state.index])
    u = u.at[-1].set(staticparams.u_reset)
    lambd_old = state.lambd_old.at[staticparams.num_ref:].set(lambd_tilde)
    lambd_old = lambd_old.at[-1].set(0.)

    u, lambd_old, v, m = roll_buffers(u, lambd_old, v, m)

    log_prob = -lax.lgamma(spikes[state.index] + 1) - lax.lgamma(params.N - spikes[state.index] + 1) \
               + spikes[state.index] * jnp.log(p) + (params.N - spikes[state.index]) * jnp.log(1 - p)

    state = State(y = y,
                  u = u,
                  v = v,
                  m = m,
                  voltage_prop = state.voltage_prop,
                  current_prop = state.current_prop,
                  current_voltage_prop = state.current_voltage_prop,
                  lambd_old = lambd_old,
                  lambd_free = lambd_tilde_free,
                  h = h,
                  x = state.x,
                  z = state.z,
                  index = state.index,
                  )
    return state, log_prob

def integrate(initial_state, params, spikes):
    carry = (initial_state, params)
    def update_state_wrap(carry, spikes):
        state, log_prob = update_state_vec(carry[0], carry[1], spikes)
        return (state, carry[1]), log_prob
    carry, log_probs = lax.scan(update_state_wrap, carry, spikes)
    return carry, log_probs

def integrate_log_prob(params, initial_state, spikes):
    _, log_probs = integrate(initial_state, params, spikes)
    return log_probs.sum()

def integrate_state(params, staticparams, spikes, K, initial_state=None, record=['u', 'h', 'y']):
    # set up state
    current_propagators = compute_current_propagators(params.tau_s, staticparams)
    current_voltage_propagators = compute_current_voltage_propagators(params.tau_s, params.tau_m, params.C_mem, staticparams)
    voltage_propagators = compute_voltage_propagators(params.tau_m, staticparams)

    M = params.tau_m.shape[0]
    m_init = np.zeros((M, K), dtype=np.float32)
    m_init[:,0] = np.ones(M, dtype=np.float32) * params.N
    m_init = jnp.array(m_init)

    u_init = np.array([np.linspace(0,15,K) for _ in range(M)], dtype=np.float32)
    state = State(y = jnp.zeros((M, 2, 2), dtype=jnp.float32),
                  u = jnp.zeros((M, K), dtype=jnp.float32),
                  v = jnp.zeros((M, K), dtype=jnp.float32),
                  current_prop = current_propagators,
                  current_voltage_prop = current_voltage_propagators,
                  voltage_prop = voltage_propagators,
                  m = m_init,
                  lambd_old = jnp.zeros((M, K), dtype=jnp.float32),
                  h = jnp.zeros((M), dtype=jnp.float32),
                  x = jnp.zeros((M), dtype=jnp.float32),
                  z = jnp.zeros((M), dtype=jnp.float32),
                  lambd_free = jnp.zeros((M), dtype=jnp.float32),
                  index = jnp.arange(M),
                  )

    # set up vectorization and jit functions
    update_state_partial = partial(update_state, staticparams=staticparams)
    update_state_vec = jax.vmap(update_state_partial, in_axes=(0, 0, None))
    update_state_jit = jax.jit(update_state_vec)

    num_steps = len(spikes)
    rec = dict()
    for key in record:
        rec[key] = np.zeros((num_steps, *getattr(state, key).shape), dtype=np.float32)
    for i in range(num_steps):
        print(i, end='\r')
        state, log_p = update_state_jit(state, params, spikes[i])
        for key in record:
            rec[key][i] = state[key].to_py()
    return rec

def update_state_simulate(state, params, input_spikes, staticparams):
    print('TRACING')
    def P_fn(P_lambd):
        return 1-jnp.exp(-P_lambd)

    # upate currents
    # synport[i] gives indices of all populations connected to synaptic port i
    s0 = params.w[staticparams.synport[0]].dot(
                        input_spikes[staticparams.synport[0]]) / (params.tau_s[0])
    s1 = params.w[staticparams.synport[1]].dot(
                        input_spikes[staticparams.synport[1]]) / (params.tau_s[1])
    s = jnp.stack((s0, s1)) # additions to synaptic ports 0 and 1
    y = vupdate_synapses(state.current_prop, state.y, s)

    X = state.m.sum()

    # free neurons
    h = update_u(state.current_voltage_prop,
                 state.voltage_prop,
                 y,
                 state.h,
                 params.RI)

    lambd_tilde_free = cg_escape(h,
                    staticparams.sigma_noise,
                    params.C_mem,
                    params.tau_m,
                    params.RI,
                    staticparams.tau_noise,
                    params.u_thr)


#     lambd_tilde_free = exponential_escape(h - params.u_thr, params.c, params.delta_u)
    P_free = 1 - jnp.exp(-0.5 * (lambd_tilde_free + state.lambd_free) * staticparams.dt)

    # refractory neurons
    # grab non-refractory indices
    u = state.u[staticparams.num_ref:]
    v = state.v[staticparams.num_ref:]
    m = state.m[staticparams.num_ref:]
    lambd_old = state.lambd_old[staticparams.num_ref:]

    # update membrane potentials
    u = vupdate_u(state.current_voltage_prop,
                  state.voltage_prop,
                  y,
                  u,
                  params.RI)

    # compute escape rate
#    lambd_tilde = vexponential_escape(u - params.u_thr, params.c, params.delta_u)
    lambd_tilde = vcg_escape(u,
                    staticparams.sigma_noise,
                    params.C_mem,
                    params.tau_m,
                    params.RI,
                    staticparams.tau_noise,
                    params.u_thr)


    P_lambd = 0.5 * (lambd_tilde + state.lambd_old[staticparams.num_ref:]) * staticparams.dt
    P_lambd = P_fn(P_lambd)

    Y = P_lambd.dot(v)
    Z = v.sum()

    Pm = P_lambd * m
    W = Pm.sum()

    neg_P_lambd = 1 - P_lambd
    v = (neg_P_lambd)**2 * v + Pm
    m = (neg_P_lambd) * m

    def P_fn_true(Y, P_free, Z, z):
        return (Y + P_free * z) / (Z + z)

    def P_fn_false(Y, P_free, Z, z):
        return 0.

    z = state.z
    x = state.x
    P_Lambd = lax.cond(Z + z > 0, P_fn_true, P_fn_false, Y, P_free, Z, z)

    z = (1 - P_free)**2 * z + P_free * x + v[-1]
    x = (1 - P_free) * x + m[-1]

    n_bar = W + P_free * x + P_Lambd * (params.N - X - x)
    #p = jnp.clip(n_bar / params.N, 1e-3, 1-1e-3)
    p = n_bar / params.N
    #spikes = jax.random.poisson(subkey, n_bar).astype(jnp.float32)
    spikes = (state.rndunif < p).sum().astype(jnp.float32)
    u = state.u.at[staticparams.num_ref:].set(u)
    v = state.v.at[staticparams.num_ref:].set(v)
    m = state.m.at[staticparams.num_ref:].set(m)

    m = m.at[-1].set(spikes)
    u = u.at[-1].set(staticparams.u_reset)
    lambd_old = state.lambd_old.at[staticparams.num_ref:].set(lambd_tilde)
    lambd_old = lambd_old.at[-1].set(0.)

    u, lambd_old, v, m = roll_buffers(u, lambd_old, v, m)

    log_prob = -lax.lgamma(spikes + 1) - lax.lgamma(params.N - spikes + 1) \
               + spikes * jnp.log(p) + (params.N - spikes) * jnp.log(1 - p)

    state = State(y = y,
                  u = u,
                  v = v,
                  m = m,
                  voltage_prop = state.voltage_prop,
                  current_prop = state.current_prop,
                  current_voltage_prop = state.current_voltage_prop,
                  lambd_old = lambd_old,
                  lambd_free = lambd_tilde_free,
                  h = h,
                  x = x,
                  z = z,
                  index = state.index,
                  spikes = spikes
                  )
    return state, log_prob

def simulate(params, staticparams, K, num_steps, initial_state=None, record=['u', 'h', 'y']):
    # set up state
    current_propagators = compute_current_propagators(params.tau_s, staticparams)
    current_voltage_propagators = compute_current_voltage_propagators(params.tau_s, params.tau_m, params.C_mem, staticparams)
    voltage_propagators = compute_voltage_propagators(params.tau_m, staticparams)

    num_delay = int(staticparams.delay / staticparams.dt)
    M = params.tau_m.shape[0]
    m_init = np.zeros((M, K), dtype=np.float32)
    m_init[:,staticparams.num_ref+1] = np.ones(M, dtype=np.float32) * params.N
    m_init = jnp.array(m_init)

    spikes = np.zeros((num_steps, M), dtype=np.float32)
    N = params.N.to_py().astype(np.int32)
    N_max = N.max()
    key = jax.random.PRNGKey(0)
    rndunif = jnp.ones((M, N_max), dtype=jnp.float32)
    for j in range(M):
        key, subkey = jax.random.split(key)
        rndunif = rndunif.at[j,:N[j]].set(jax.random.uniform(subkey, shape=(N[j],)))

    state = State(y = jnp.zeros((M, 2, 2), dtype=jnp.float32),
                  u = jnp.zeros((M, K), dtype=jnp.float32),
                  v = jnp.zeros((M, K), dtype=jnp.float32),
                  current_prop = current_propagators,
                  current_voltage_prop = current_voltage_propagators,
                  voltage_prop = voltage_propagators,
                  m = m_init,
                  lambd_old = jnp.zeros((M, K), dtype=jnp.float32),
                  h = jnp.zeros((M), dtype=jnp.float32),
                  x = jnp.zeros((M), dtype=jnp.float32),
                  z = jnp.zeros((M), dtype=jnp.float32),
                  lambd_free = jnp.zeros((M), dtype=jnp.float32),
                  index = jnp.arange(M),
                  spikes = jnp.zeros(M, dtype=jnp.float32),
                  rndunif = rndunif
                  )

    # set up vectorization and jit functions
    update_state_partial = partial(update_state_simulate, staticparams=staticparams)
    update_state_vec = jax.vmap(update_state_partial, in_axes=(0, 0, None))
    update_state_jit = jax.jit(update_state_vec)

    rec = dict()
    for dictkey in record:
        rec[dictkey] = np.zeros((num_steps, *getattr(state, dictkey).shape), dtype=np.float32)
    for i in range(num_steps):
        print(i, end='\r')
        state, log_p = update_state_jit(state, params, spikes[i-num_delay])
        # if i == 1000:
        #     state.spikes = jnp.array([1., 1.])
        for j in range(M):
            key, subkey = jax.random.split(key)
            rndunif = rndunif.at[j,:N[j]].set(jax.random.uniform(subkey, shape=(N[j],)))
        state.rndunif = rndunif
        spikes[i] = state.spikes.to_py()
        for dictkey in record:
            rec[dictkey][i] = state[dictkey].to_py()
    if 'spikes' in record:
        rec['spikes'] = spikes
    return rec

# steps = 10000
# M = 2
# K = 1000
# synport = np.array([[0], [1]], dtype=np.int32)
# staticparams = StaticParams(synport = synport,
#                             u_reset = np.array(0., dtype=np.float32),
#                             dt = 0.1,
#                             num_ref=4,
#                             delay = 2.,
#                             sigma_noise = 0.,
#                             k=4.,
#                             R=0.08
#                             sigma_u = 1.
# 
#                             )
# 
# w = np.ones((M, 2), dtype=jnp.float32)
# w[:,0] = 10. * np.exp(1) * 0.1
# w[:,1] = -50. * np.exp(1) * 0.1
# tau_s = np.ones((M, 2), dtype=np.float32)
# tau_s[:,0] = 3.
# tau_s[:,1] = 3.
# N = np.zeros(2, dtype=np.float32)
# N[0] = 800
# N[1] = 200
# params = Params(RI = jnp.ones(M, dtype=jnp.float32)*27.,
#                 tau_m = jnp.ones(M, dtype=jnp.float32)*20.,
#                 N = jnp.array(N),
#                 tau_s = jnp.array(tau_s),
#                 u_thr = jnp.ones(M, dtype=jnp.float32) * 20.,
#                 c = jnp.ones(M, dtype=jnp.float32) * 0.,
#                 delta_u = jnp.ones(M, dtype=jnp.float32) * 1.,
#                 C_mem = jnp.ones(M, dtype=jnp.float32) * 250.,
#                 w = jnp.array(w)
#                 )
# 
# 
# spikes = np.zeros((steps, M))
# spikes[80,:] = 50
# 
# current_propagators = compute_current_propagators(params.tau_s, staticparams)
# current_voltage_propagators = compute_current_voltage_propagators(params.tau_s, params.tau_m, params.C_mem, staticparams)
# voltage_propagators = compute_voltage_propagators(params.tau_m, staticparams)
# 
# grad_log_prob = jax.grad(integrate_log_prob)
# grad_log_prob_jit = jax.jit(grad_log_prob)
# 
# update_state_partial = partial(update_state, staticparams=staticparams)
# update_state_vec = jax.vmap(update_state_partial, in_axes=(0, 0, None))
# 
# 
# steps = 5000
# 
# m_init = np.zeros((M, K), dtype=np.float32)
# m_init[:,0] = np.ones(M, dtype=np.float32) * 1000
# m_init = jnp.array(m_init)
# 
# state = State(y = jnp.zeros((M, 2, 2), dtype=jnp.float32),
#               u = jnp.zeros((M, K), dtype=jnp.float32),
#               v = jnp.zeros((M, K), dtype=jnp.float32),
#               current_prop = current_propagators,
#               current_voltage_prop = current_voltage_propagators,
#               voltage_prop = voltage_propagators,
#               m = m_init,
#               lambd_old = jnp.zeros((M, K), dtype=jnp.float32),
#               h = jnp.zeros((M), dtype=jnp.float32),
#               x = jnp.zeros((M), dtype=jnp.float32),
#               z = jnp.zeros((M), dtype=jnp.float32),
#               lambd_free = jnp.zeros((M), dtype=jnp.float32),
#               index = jnp.arange(M),
#               )
# 
# def integrate(initial_state, params, spikes):
#     carry = (initial_state, params)
#     def update_state_wrap(carry, spikes):
#         state, log_prob = update_state_vec(carry[0], carry[1], spikes)
#         return (state, carry[1]), log_prob
#     carry, log_probs = lax.scan(update_state_wrap, carry, spikes)
#     return carry, log_probs
# 
# def integrate_log_prob(params, initial_state, spikes, num_burn_in):
#     _, log_probs = integrate(initial_state, params, spikes)
#     return -log_probs[num_burn_in:].sum()

# num_burn_in = 500
# log_prob = integrate_log_prob(params, state, spikes, num_burn_in)




