from typing import Union
from functools import partial
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import chex

import matplotlib.pyplot as plt

@chex.dataclass
class Params:
    # J:        jnp.ndarray = jnp.array([1.]*K, dtype=jnp.float32)
    N:        jnp.ndarray
    tau_m:    jnp.ndarray
    tau_s:    jnp.ndarray
    # u_reset:  jnp.ndarray = jnp.array([0.]*K, dtype=jnp.float32)
    u_thr:    jnp.ndarray
    c:        jnp.ndarray
    delta_u:  jnp.ndarray
    # k:        jnp.ndarray = jnp.array([5.]*K, dtype=jnp.float32)
    # k_asc:    jnp.ndarray = jnp.array([5.]*K, dtype=jnp.float32)
    # f_asc:    jnp.ndarray = jnp.array([5.]*K, dtype=jnp.float32)
    C_mem:    jnp.ndarray
    RI:       jnp.ndarray
    w:        jnp.ndarray

@chex.dataclass(frozen=True)
class StaticParams:
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

def compute_current_propagators(tau_s, staticparams):
    t1 = jnp.exp(-staticparams.dt / tau_s)
    t2 = staticparams.dt * t1
    t3 = t1
    return jnp.stack((t1, t2, t3))

compute_current_propagators = jax.vmap(compute_current_propagators, in_axes=(0, None)) # over synaptic ports
compute_current_propagators = jax.vmap(compute_current_propagators, in_axes=(0, None)) # over populations

def compute_current_voltage_propagators(tau_s, tau_m, C_mem, staticparams):
    t00 = jnp.exp(-staticparams.dt / tau_s)
    t01 = jnp.exp(-staticparams.dt / tau_m)
    denom = (1/tau_s - 1/tau_m)

    t0 = ((t01 - t00) / denom**2 - (staticparams.dt * t00 / denom)) / C_mem
    t1 = (t01 - t00) / (C_mem * denom)
    return jnp.stack((t0, t1))

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

def exponential_escape(x, c, delta):
    return jnp.exp(c + x / delta)

vexponential_escape = jax.vmap(exponential_escape, in_axes=(0, None, None))

def compute_current_contrib(current_voltage_prop, y):
    return current_voltage_prop.dot(y)

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

def update_synapses(current_prop, y, spikes, w):
    y1 = current_prop[0] * y[0]
    y2 = current_prop[1] * y[0] + current_prop[2] * y[1]
    y1 = y1 + spikes.dot(w)
    return jnp.stack((y1, y2))

vupdate_synapses = jax.vmap(update_synapses)

def update_state(state: State, params: Params, spikes: jnp.ndarray, staticparams: StaticParams) -> State:
    print('TRACING')
    def P_fn(P_lambd):
        return 1-jnp.exp(-P_lambd)

    # upate current
    #y = state.current_prop.dot(state.y)
    y = vupdate_synapses(state.current_prop, state.y, spikes, params.w)

    X = state.m.sum()

    # free neurons
    h = update_u(state.current_voltage_prop,
                  state.voltage_prop,
                  y,
                  state.h,
                  params.RI)

    lambd_tilde_free = exponential_escape(h - params.u_thr, params.c, params.delta_u)
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
    lambd_tilde = vexponential_escape(u - params.u_thr, params.c, params.delta_u)

    P_lambd = 0.5 * (lambd_tilde + state.lambd_old[staticparams.num_ref:]) * staticparams.dt
    P_lambd = P_fn(P_lambd)

    Y = P_lambd.dot(state.v[staticparams.num_ref:])
    Z = state.v[staticparams.num_ref:].sum()

    Pm = P_lambd * state.m[staticparams.num_ref:]
    W = Pm.sum()

    neg_P_lambd = 1 - P_lambd
    v = (neg_P_lambd)**2 * state.v[staticparams.num_ref:] - Pm
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
    p = jnp.clip(n_bar / params.N, 1e-3, 1-1e-3)
    u = state.u.at[staticparams.num_ref:].set(u)
    v = state.v.at[staticparams.num_ref:].set(v)
    m = state.m.at[staticparams.num_ref:].set(m)

    m = m.at[-1].set(spikes[state.index])

    lambd_old = state.lambd_old.at[staticparams.num_ref:].set(lambd_tilde)

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
                  x = state.x,
                  z = state.z,
                  index = state.index
                  )
    return state, log_prob

def integrate(initial_state, params, spikes):
    carry = (initial_state, params)
    def update_state_wrap(carry, spikes):
        state, log_prob = update_state_vec(carry[0], carry[1], spikes)
        return (state, carry[1]), log_prob
    carry, log_probs = lax.scan(update_state_wrap, carry, spikes)
    return carry, log_probs

def integrate_state(params, staticparams, spikes, initial_state=None, record=['u', 'h', 'y']):
    # set up state
    current_propagators = compute_current_propagators(params.tau_s, staticparams)
    current_voltage_propagators = compute_current_voltage_propagators(params.tau_s, params.tau_m, params.C_mem, staticparams)
    voltage_propagators = compute_voltage_propagators(params.tau_m, staticparams)
    print(voltage_propagators)
    print(current_voltage_propagators)
    print(current_propagators)
    M = params.tau_m.shape[0]
    K = 1000
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
                  index = jnp.arange(M)
                  )

    # set up vectorization and jit functions
    update_state_partial = partial(update_state, staticparams=staticparams)
    update_state_vec = jax.vmap(update_state_partial)
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

def integrate_log_prob(params, initial_state, spikes):
    _, log_probs = integrate(initial_state, params, spikes)
    return log_probs.sum()

# steps = 10000
# M = 1
# K = 1000
# staticparams = StaticParams()
#
# params = Params(RI = jnp.ones(M, dtype=jnp.float32)*20.,
#                 tau_m = jnp.ones(M, dtype=jnp.float32)*20.,
#                 N = jnp.ones(M, dtype=jnp.float32) * 1000,
#                 tau_s = jnp.ones((M, 2), dtype=jnp.float32),
#                 u_thr = jnp.zeros(M, dtype=jnp.float32),
#                 c = jnp.zeros(M, dtype=jnp.float32),
#                 delta_u = jnp.ones(M, dtype=jnp.float32) * 2,
#                 C_mem = jnp.zeros(M, dtype=jnp.float32)
#                 )
#
# spikes = np.zeros((steps, M))
# spikes[80,:] = 50
#
# current_propagators = compute_current_propagators(params.tau_s, staticparams)
# current_voltage_propagators = compute_current_voltage_propagators(params.tau_s, params.tau_m, params.C_mem, staticparams)
# voltage_propagators = compute_voltage_propagators(params.tau_m, staticparams)

# grad_log_prob = jax.grad(integrate_log_prob)
# grad_log_prob_jit = jax.jit(grad_log_prob)
#staticparams = StaticParams()

# staticaxes = StaticParams(dt=None,
#                           num_steps=None,
#                           num_ref=None)
#
# update_state_partial = partial(update_state, staticparams=staticparams)
# update_state_vec = jax.vmap(update_state_partial)
# update_state_jit = jax.jit(update_state_vec)
#
# steps = 5000
#
# params = Params(RI = jnp.ones(M)*20.,
#                 tau_m = jnp.ones(M)*20.,
#                 N = jnp.ones(M) * 1000,
#                 tau_s = jnp.ones((M, 2), dtype=jnp.float32))
#
# spikes = np.zeros((steps, M))
# spikes[80,:] = 50
# current_propagators = compute_current_propagators(params.tau_s, staticparams)
# current_voltage_propagators = compute_current_voltage_propagators(params.tau_s, params.tau_m, params.C_mem, staticparams)
# voltage_prop = compute_voltage_propagators(params.tau_m, staticparams)
#
# m_init = np.zeros((M, K), dtype=np.float32)
# m_init[:,0] = np.ones(M, dtype=np.float32) * 1000
# m_init = jnp.array(m_init)
#
# u_init = np.array([np.linspace(0,15,K) for _ in range(M)], dtype=np.float32)
# state = State(y = jnp.zeros((M, 2, 2), dtype=jnp.float32),
#               u = jnp.zeros((M, K), dtype=jnp.float32),
#               v = jnp.zeros((M, K), dtype=jnp.float32),
#               current_prop = current_propagators,
#               current_voltage_prop = current_voltage_propagators,
#               voltage_prop = voltage_prop,
#               m = m_init,
#               lambd_old = jnp.zeros((M, K), dtype=jnp.float32),
#               h = jnp.zeros((M), dtype=jnp.float32),
#               x = jnp.zeros((M), dtype=jnp.float32),
#               z = jnp.zeros((M), dtype=jnp.float32),
#               lambd_free = jnp.zeros((M), dtype=jnp.float32),
#               )

# carry, log_probs = jax.jit(integrate)(state, params, spikes)

# rec = integrate_state(params, staticparams, spikes)
