import nest
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from mesogif_jax import Params, StaticParams, State, integrate_state

##################
# NEST SIMULATION
##################
np.random.seed(123)
neuron_params = {'C_m': 250.,
                 'tau_m': 20.,
                 't_ref': 2.,
                 'E_L': 0.,
                 'V_reset': 0.,
                 'V_m': 0.,
                 'V_th': 20.,
                 'I_e': 1. / 20 * 250.,
                 'tau_syn_ex': 5.,
                 'tau_syn_in': 2.}

grng_seed = 1234
rng_seed = 12345
n_spikes = []
Vms = []
nest.ResetKernel()
nest.CopyModel('static_synapse', 'excitatory',
                {'weight': 10., 'delay': 0.1})
nest.CopyModel('static_synapse', 'inhibitory',
                {'weight': -40., 'delay': 0.1})

nest.SetDefaults('iaf_psc_alpha', neuron_params)
neuron = nest.Create('iaf_psc_alpha')
mm = nest.Create('multimeter')
nest.SetStatus(mm, {'record_from': ['V_m', 'I_syn_ex'], 'interval': 0.1})
espikegen = nest.Create('spike_generator')
nest.SetStatus(espikegen, {'spike_times': [8.]})
nest.Connect(espikegen, neuron, 'all_to_all', 'excitatory')

ispikegen = nest.Create('spike_generator')
nest.SetStatus(ispikegen, {'spike_times': [80.]})
nest.Connect(ispikegen, neuron, 'all_to_all', 'inhibitory')

nest.Connect(mm, neuron)
nest.Simulate(300.)
events = nest.GetStatus(mm, 'events')[0]
Vm = events['V_m']
isyn = events['I_syn_ex']
t = events['times']

########################
# MESOSCOPIC SIMULATION
########################

steps = 3000
M = 1
K = 1000
staticparams = StaticParams()

w = np.ones((M, 2), dtype=jnp.float32)
w[0,0] = 10. * np.exp(1)
w[0,1] = -40. * np.exp(1)
params = Params(RI = jnp.ones(M, dtype=jnp.float32)*1.,
                tau_m = jnp.ones(M, dtype=jnp.float32)*20.,
                N = jnp.ones(M, dtype=jnp.float32) * 1000,
                tau_s = jnp.array([[5., 2.]], dtype=jnp.float32),
                u_thr = jnp.ones(M, dtype=jnp.float32) * 20.,
                c = jnp.ones(M, dtype=jnp.float32) * 0.,
                delta_u = jnp.ones(M, dtype=jnp.float32) * 2.,
                C_mem = jnp.ones(M, dtype=jnp.float32) * 250.,
                w = jnp.array(w)
                )

spikes = np.zeros((steps, M, 2))
spikes[81,:,0] = 1
spikes[801,:,1] = 1

rec = integrate_state(params, staticparams, spikes)

plt.plot(Vm, label='nest')
plt.plot(rec['h'], '--', label='meso')
plt.legend()
plt.show()
