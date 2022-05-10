from typing import Union
from functools import partial
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import chex

import matplotlib.pyplot as plt

M = 5
K = 1000
@chex.dataclass
class Params:
    # J:        jnp.ndarray = jnp.array([1.]*K, dtype=jnp.float32)
    N:        jnp.ndarray
    tau_m:    jnp.ndarray = jnp.ones(M, dtype=jnp.float32) * 20.
    tau_s:    jnp.ndarray = jnp.ones((M, 2), dtype=jnp.float32) * 5.
    # u_reset:  jnp.ndarray = jnp.array([0.]*K, dtype=jnp.float32)
    u_thr:    jnp.ndarray = jnp.ones(M, dtype=jnp.float32) * 20.
    c:        jnp.ndarray = jnp.zeros(M, dtype=jnp.float32)
    delta_u:  jnp.ndarray = jnp.ones(M, dtype=jnp.float32) * 2.
    # k:        jnp.ndarray = jnp.array([5.]*K, dtype=jnp.float32)
    # k_asc:    jnp.ndarray = jnp.array([5.]*K, dtype=jnp.float32)
    # f_asc:    jnp.ndarray = jnp.array([5.]*K, dtype=jnp.float32)
    C_mem:    jnp.ones = jnp.ones(M, dtype=jnp.float32) * 20.
    RI:       jnp.ones = jnp.ones(M, dtype=jnp.float32) * 20.

@chex.dataclass(frozen=True)
class StaticParams:
    num_steps: int = 10000
    dt:        Union[int, float] = 0.1
    num_ref:   int = 50

@chex.dataclass
class State:
    current_prop: Union[int, jnp.ndarray]
    current_voltage_prop: Union[int, jnp.ndarray]
    m:          Union[int, jnp.ndarray]
    h:          Union[int, jnp.ndarray]
    x:          Union[int, jnp.ndarray]
    z:          Union[int, jnp.ndarray]
    lambd_free: Union[int, jnp.ndarray]
    u:      Union[int, jnp.ndarray]         = jnp.zeros((M, K), dtype=jnp.float32)
    v:      Union[int, jnp.ndarray]         = jnp.zeros((M, K), dtype=jnp.float32)
    lambd_old:      Union[int, jnp.ndarray] = jnp.zeros((M, K), dtype=jnp.float32)
    y:      Union[int, jnp.ndarray]         = jnp.zeros((M, 2, 2), dtype=jnp.float32)

# def compute_current_propagators(params, staticparams):
#     args = {'tau_s': params.tau_s, 'dt': staticparams.dt}
#     def compute(args):
#         Ad = args['dt'] * jnp.array([[-1/args['tau_s'], 0.],
#                                      [1., -1/args['tau_s']]], dtype=jnp.float32)
#
#         exp_Ad = jax.scipy.linalg.expm(Ad)
#         return exp_Ad
#
#
#     in_axes = {'tau_s': 0, 'dt': None}
#     # vectorize over synaptic ports
#     compute_vec = jax.vmap(compute, in_axes=(in_axes,))
#
#     #vectorize over populations
#     compute_vec = jax.vmap(compute_vec, in_axes=(in_axes,))
#     return compute_vec(args)

#
# def compute_current_propagators(params, staticparams):
#     args = {'tau_s': params.tau_s, 'dt': staticparams.dt}
#     def compute(args):
#         Ad = args['dt'] * jnp.array([[-1/args['tau_s'], 0.],
#                                      [1., -1/args['tau_s']]], dtype=jnp.float32)
#
#         exp_Ad = jax.scipy.linalg.expm(Ad)
#         return exp_Ad
#
#
#     in_axes = {'tau_s': 0, 'dt': None}
#     # vectorize over synaptic ports
#     compute_vec = jax.vmap(compute, in_axes=(in_axes,))
#
#     #vectorize over populations
#     compute_vec = jax.vmap(compute_vec, in_axes=(in_axes,))
#     return compute_vec(args)

def compute_current_propagators(tau_s, staticparams):
    t1 = jnp.exp(-staticparams.dt / tau_s)
    t2 = staticparams.dt * t1
    t3 = t1
    return jnp.stack((t1, t2, t3))

compute_current_propagators = jax.vmap(compute_current_propagators, in_axes=(0, None)) # over synaptic ports
compute_current_propagators = jax.vmap(compute_current_propagators, in_axes=(0, None)) # over populations

# def compute_voltage_propagators(params, staticparams):
#     args = {'tau_s': params.tau_s, 'tau_m': params.tau_m, 'C_mem': params.C_mem, 'dt': staticparams.dt}
#
#     def compute(args):
#         denom = (1/args['tau_s'] - 1/args['tau_m'])
#         t00 = jnp.exp(-args['dt'] / args['tau_s'])
#         t10 = args['dt'] * jnp.exp(-args['dt'] / args['tau_s'])
#
#         t3 = jnp.exp(-args['dt'] / args['tau_m'])
#         t0 = ((t3 - t00) / denom**2 - (t10 / denom)) / args['C_mem']
#         t1 = (t3 - t00) / (args['C_mem'] * denom)
#         t2 = 1-jnp.exp(-args['dt'] / args['tau_m'])
#         return jnp.stack([t0, t1, t2, t3])
#
#     # vectorize over populations
#     in_axes = {'tau_s': 0, 'tau_m': 0, 'dt': None, 'C_mem': 0}
#     compute_vec = jax.vmap(compute, in_axes=(in_axes,))
#     return compute_vec(args)

def compute_current_voltage_propagators(tau_s, tau_m, C_mem, staticparams):
    t00 = jnp.exp(-staticparams.dt / tau_s)
    t01 = jnp.exp(-staticparams.dt / tau_m)
    denom = (1/tau_s - 1/tau_m)

    t0 = ((t01 - t00) / denom**2 - (staticparams.dt * t00 / denom)) / C_mem
    t1 = (t01 - t00) / (C_mem * denom)
    return jnp.stack((t0, t1))

def compute_voltage_propagators(tau_m, staticparams):
    t0 = jnp.exp(-staticparams.dt / tau_m)
    t1 = 1 - t0
    return jnp.stack((t0, t1))

# vmap over synaptic ports
compute_current_voltage_propagators \
    = jax.vmap(compute_current_voltage_propagators, in_axes=(0, None, None, None))

# vmap over populations
compute_current_voltage_propagators \
    = jax.vmap(compute_current_voltage_propagators, in_axes=(0, 0, 0, None))


# def update_u(y, u, RI, voltage_prop):
#     voltage_state = jnp.concatenate((y, jnp.stack((RI, u))))
#     u = voltage_prop.dot(voltage_state)
#     return u
#
# vupdate_u = jax.vmap(update_u, in_axes=(None, 0, None, None))

def current_u_contrib(y, voltage_prop):
    return y.dot(voltage_prop)

def roll_buffers(u, lambd_tilde, v, m):
    u = jnp.roll(u, 1)
    v = jnp.roll(v, 1)
    m = jnp.roll(m, 1)
    lambd_tilde = jnp.roll(lambd_tilde, 1)
    return u, lambd_tilde, v, m

def exponential_escape(x, c, delta):
    return jnp.exp(c + x / delta)

vexponential_escape = jax.vmap(exponential_escape, in_axes=(0, None, None))

def update_current_voltage(current_voltage_prop, y):
    return current_voltage_prop.dot(y)

vupdate_current_voltage = jax.vmap(update_current_voltage)

def update_voltage(current_voltage_prop, y, RI_e)

def update_synapses(current_prop, y):
    y1 = current_prop[0] * y[0]
    y2 = current_prop[1] * y[0] + current_prop[2] * y[1]
    return jnp.stack((y1, y2))

vupdate_synapses = jax.vmap(update_synapses)

def update_state(state: State, params: Params, spikes: jnp.float32, staticparams: StaticParams) -> State:
    print('TRACING')
    def P_fn(P_lambd):
        return 1-jnp.exp(-P_lambd)

    # upate current
    #y = state.current_prop.dot(state.y)
    y = vupdate_synapses(state.current_prop, state.y)

    X = state.m.sum()

    # free neurons
    h = update_u(y, state.h, params.RI, state.voltage_prop)
    lambd_tilde_free = exponential_escape(h - params.u_thr, params.c, params.delta_u)
    P_free = 1 - jnp.exp(-0.5 * (lambd_tilde_free + state.lambd_free) * staticparams.dt)

    # refractory neurons
    # grab non-refractory indices
    u = state.u[staticparams.num_ref:]
    v = state.v[staticparams.num_ref:]
    m = state.m[staticparams.num_ref:]
    lambd_old = state.lambd_old[staticparams.num_ref:]

    # update membrane potentials
    u = vupdate_u(y,
                  u,
                  params.RI,
                  state.voltage_prop)

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

    m = m.at[-1].set(spikes)

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
                  lambd_old = lambd_old,
                  lambd_free = lambd_tilde_free,
                  h = h,
                  x = state.x,
                  z = state.z
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

grad_log_prob = jax.grad(integrate_log_prob)
grad_log_prob_jit = jax.jit(grad_log_prob)
staticparams = StaticParams()

staticaxes = StaticParams(dt=None,
                          num_steps=None,
                          num_ref=None)

# update_state_vec = jax.vmap(update_state, in_axes=(0, 0, staticaxes, 0))
update_state_partial = partial(update_state, staticparams=staticparams)
update_state_vec = jax.vmap(update_state_partial)
update_state_jit = jax.jit(update_state_vec)

steps = 5000

params = Params(RI = jnp.ones(M)*20.,
                tau_m = jnp.ones(M)*20.,
                N = jnp.ones(M) * 1000,
                tau_s = jnp.ones((M, 2), dtype=jnp.float32))

spikes = np.zeros((steps, M))
spikes[80,:] = 50
current_propagators = compute_current_propagators(params.tau_s, staticparams)
current_voltage_propagators = compute_current_voltage_propagators(params.tau_s, params.tau_m, params.C_mem, staticparams)

m_init = np.zeros((M, K), dtype=np.float32)
m_init[:,0] = np.ones(M, dtype=np.float32) * 1000
m_init = jnp.array(m_init)

u_init = np.array([np.linspace(0,15,K) for _ in range(M)], dtype=np.float32)
state = State(y = jnp.zeros((M, 2, 2), dtype=jnp.float32),
              u = jnp.zeros((M, K), dtype=jnp.float32),
              v = jnp.zeros((M, K), dtype=jnp.float32),
              current_prop = current_propagators,
              current_voltage_prop = current_voltage_propagators,
              m = m_init,
              lambd_old = jnp.zeros((M, K), dtype=jnp.float32),
              h = jnp.zeros((M), dtype=jnp.float32),
              x = jnp.zeros((M), dtype=jnp.float32),
              z = jnp.zeros((M), dtype=jnp.float32),
              lambd_free = jnp.zeros((M), dtype=jnp.float32),
              )
#
# # carry, log_probs = jax.jit(integrate)(state, params, spikes)
#
# states = [state]
# ps = []
# for i in range(steps):
#     print(i, end='\r')
#     s, p = update_state_jit(states[-1], params, spikes[i])
#     ps.append(p.to_py())
#     states.append(s)
#
# ps = np.array(ps)
# us = np.array([s.u.to_py() for s in states])
#
# example_u = np.array([us[i,0,i%K] for i in range(0,1100)])
# plt.plot(example_u)
# plt.show()
