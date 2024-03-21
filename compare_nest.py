import nest
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from mesogif_jax import Params, StaticParams, State, integrate_state

##################
# NEST SIMULATION
##################
np.random.seed(123)
neuron_params = {
    "C_m": 250.0,
    "tau_m": 20.0,
    "t_ref": 2.0,
    "E_L": 0.0,
    "V_reset": 0.0,
    "V_m": 0.0,
    "V_th": 20.0,
    "I_e": 1.0 / 20 * 250.0,
    "tau_syn_ex": 5.0,
    "tau_syn_in": 2.0,
}

grng_seed = 1234
rng_seed = 12345
n_spikes = []
Vms = []
nest.ResetKernel()
nest.CopyModel("static_synapse", "excitatory", {"weight": 10.0, "delay": 0.1})
nest.CopyModel("static_synapse", "inhibitory", {"weight": -40.0, "delay": 0.1})

nest.SetDefaults("iaf_psc_alpha", neuron_params)
neuron = nest.Create("iaf_psc_alpha")
mm = nest.Create("multimeter")
nest.SetStatus(mm, {"record_from": ["V_m", "I_syn_ex"], "interval": 0.1})
espikegen = nest.Create("spike_generator")
nest.SetStatus(espikegen, {"spike_times": [8.0]})
nest.Connect(espikegen, neuron, "all_to_all", "excitatory")

ispikegen = nest.Create("spike_generator")
nest.SetStatus(ispikegen, {"spike_times": [80.0]})
nest.Connect(ispikegen, neuron, "all_to_all", "inhibitory")

nest.Connect(mm, neuron)
nest.Simulate(300.0)
events = nest.GetStatus(mm, "events")[0]
Vm = events["V_m"]
isyn = events["I_syn_ex"]
t = events["times"]

########################
# MESOSCOPIC SIMULATION
########################

steps = 3000
M = 2
K = 1000
synport = np.array([[0], [1]], dtype=np.int32)
staticparams = StaticParams(synport=synport, u_reset=0)

w = np.ones((M, 2), dtype=jnp.float32)
w[:, 0] = 10.0 * np.exp(1)
w[:, 1] = -40.0 * np.exp(1)

tau_s = np.ones((M, 2), dtype=np.float32)
tau_s[:, 0] = 5.0
tau_s[:, 1] = 2.0
params = Params(
    RI=jnp.ones(M, dtype=jnp.float32) * 1.0,
    tau_m=jnp.ones(M, dtype=jnp.float32) * 20.0,
    N=jnp.ones(M, dtype=jnp.float32) * 1000,
    tau_s=jnp.array(tau_s),
    u_thr=jnp.ones(M, dtype=jnp.float32) * 20.0,
    c=jnp.ones(M, dtype=jnp.float32) * 0.0,
    delta_u=jnp.ones(M, dtype=jnp.float32) * 2.0,
    C_mem=jnp.ones(M, dtype=jnp.float32) * 250.0,
    w=jnp.array(w),
)

spikes = np.zeros((steps, M))
spikes[81, 0] = 1
spikes[801, 1] = 1

rec = integrate_state(params, staticparams, spikes, K, record=["u", "h", "y"])

example_u = [rec["u"][i - 100, 0, i] for i in range(100, 1000)]
plt.plot(Vm, label="nest")
plt.plot(rec["h"], "--", label="meso h")
plt.plot(example_u, "-.", label="meso u")
plt.legend()
plt.show()
