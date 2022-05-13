import nest
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from mesogif_jax import Params, StaticParams, State, integrate_state

##################
# NEST SIMULATION
##################
# np.random.seed(123)
# neuron_params = {"C_m": 250.,
#                  "tau_m": 20.,
#                  "t_ref": 2.,
#                  "E_L": 0.,
#                  "V_reset": 0.,
#                  "V_m": 0.,
#                  "V_th": 20.,
#                  "I_e": 0.,
#                  "tau_syn_ex": 5.,
#                  "tau_syn_in": 5.}
#
# grng_seed = 1234
# rng_seed = 12345
# n_spikes = []
# Vms = []
# nest.ResetKernel()
# nest.CopyModel("static_synapse", "excitatory",
#                 {'weight': 10., 'delay': 0.1})
#
# nest.SetDefaults('iaf_psc_alpha', neuron_params)
# neuron = nest.Create('iaf_psc_alpha')
# mm = nest.Create('multimeter')
# nest.SetStatus(mm, {'record_from': ['V_m', 'I_syn_ex'], 'interval': 0.1})
# spikegen = nest.Create('spike_generator')
# nest.SetStatus(spikegen, {'spike_times': [0.]})
# nest.Connect(spikegen, neuron, 'all_to_all', 'excitatory')
# nest.Connect(mm, neuron)
# nest.Simulate(1000.)
# events = nest.GetStatus(mm, 'events')[0]
# Vm = events['V_m']
# isyn = events['I_syn_ex']
# t = events['times']

########################
# MESOSCOPIC SIMULATION
########################

steps = 10000
M = 1
K = 1000
staticparams = StaticParams()

w = np.ones((M, 2), dtype=jnp.float32)
w[0,0] = 1.
w[0,1] = -1.
params = Params(RI = jnp.ones(M, dtype=jnp.float32)*0.,
                tau_m = jnp.ones(M, dtype=jnp.float32)*20.,
                N = jnp.ones(M, dtype=jnp.float32) * 1000,
                tau_s = jnp.ones((M, 2), dtype=jnp.float32) * 5.,
                u_thr = jnp.ones(M, dtype=jnp.float32) * 20.,
                c = jnp.ones(M, dtype=jnp.float32) * 0.,
                delta_u = jnp.ones(M, dtype=jnp.float32) * 2.,
                C_mem = jnp.ones(M, dtype=jnp.float32) * 250.,
                w = jnp.array(w)
                )

spikes = np.zeros((steps, M, 2))
spikes[80,:,0] = 50
spikes[800,:,1] = 50

rec = integrate_state(params, staticparams, spikes)
