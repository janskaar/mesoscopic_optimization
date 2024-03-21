import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.signal import welch
import nest


simtime = 5000.0

neuron_status = {
    "C_m": 250.0,
    "tau_m": 20.0,
    "t_ref": 2.0,
    "E_L": 0.0,
    "V_reset": 0.0,
    "tau_syn_ex": 3.0,
    "tau_syn_in": 3.0,
    "V_m": 0.0,
    "V_th": 15.0,
    "I_e": 0.0,
}

NE = 800
NI = 200
N = np.array([NE, NI])
c = np.array([0.1, 0.1])
g = np.array([10.0, -50.0])
seed = 12345

nu_thr = (
    neuron_status["V_th"]
    * neuron_status["C_m"]
    / (neuron_status["tau_m"] * g[0] * neuron_status["tau_syn_ex"])
    * 1000.0
)
rate = nu_thr * 1.05
print(rate)

nest.ResetKernel()
nest.set(rng_seed=seed)
nest.print_time = True
nest.SetDefaults("iaf_psc_exp", neuron_status)
ex_pop = nest.Create("iaf_psc_exp", NE)
in_pop = nest.Create("iaf_psc_exp", NI)

noise = nest.Create("poisson_generator")
nest.SetStatus(noise, {"rate": rate})

nest.Connect(
    noise,
    ex_pop + in_pop,
    conn_spec={"rule": "all_to_all"},
    syn_spec={"synapse_model": "static_synapse", "weight": g[0]},
)

# nest.Connect(ex_pop, ex_pop + in_pop, conn_spec={"rule": "fixed_indegree",
#                                                  "indegree": int(0.1 * NE)},
#                                       syn_spec={"synapse_model": "static_synapse", "weight": g[0]})
#
# nest.Connect(in_pop, ex_pop + in_pop, conn_spec={"rule": "fixed_indegree",
#                                                 "indegree": int(0.1 * NI)},
#                                       syn_spec={"synapse_model": "static_synapse", "weight": g[1]})

# nest.Connect(ex_pop, ex_pop + in_pop, conn_spec={"rule": "all_to_all"},
#                                       syn_spec={"synapse_model": "static_synapse", "weight": g[0]*c[0]})
#
# nest.Connect(in_pop, ex_pop + in_pop, conn_spec={"rule": "all_to_all"},
#                                       syn_spec={"synapse_model": "static_synapse", "weight": g[1]*c[1]})

ex_spikerec = nest.Create("spike_recorder", NE)
in_spikerec = nest.Create("spike_recorder", NI)
nest.Connect(ex_pop, ex_spikerec, "one_to_one")
nest.Connect(in_pop, in_spikerec, "one_to_one")

mm = nest.Create("multimeter", NE)
nest.SetStatus(mm, {"record_from": ["V_m", "I_syn_ex", "I_syn_in"], "interval": 0.1})
nest.Connect(mm, ex_pop, "one_to_one")
nest.Simulate(simtime)

u = np.array([e["V_m"] for e in mm.events])
isyn = np.array([e["I_syn_ex"] for e in mm.events])


ex_spikes = nest.GetStatus(ex_spikerec, "events")
in_spikes = nest.GetStatus(in_spikerec, "events")
ex_spikes = np.concatenate([a["times"] for a in ex_spikes])
in_spikes = np.concatenate([a["times"] for a in in_spikes])
ex_isi = [np.ediff1d(a) for a in ex_spikes]
ex_isi_cat = np.concatenate(ex_isi)

bins = np.arange(0, simtime + 0.1, 0.1) - 0.001
ehist, _ = np.histogram(ex_spikes, bins=bins)
ihist, _ = np.histogram(in_spikes, bins=bins)
ehist = ehist.reshape((-1, 5)).sum(1)
ihist = ihist.reshape((-1, 5)).sum(1)

fs, psd = welch(ehist)
