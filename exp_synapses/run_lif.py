import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.signal import welch
import nest


simtime = 2000.

neuron_status = {"C_m": 250.,
                 "tau_m": 20.,
                 "t_ref": 2.,
                 "E_L": 0.,
                 "V_reset": 0.,
                 "tau_syn_ex": 3.,
                 "tau_syn_in": 3.,
                 "V_m": 0.,
                 "V_th": 15.,
                 "I_e": 0.
                }

NE = 4000
NI = 1000
N = np.array([NE, NI])
c = np.array([0.1, 0.1])
g = np.array([10., -50.])
seed = 12345

nu_thr = neuron_status["V_th"] * neuron_status["C_m"] / (neuron_status["tau_m"] * g[0] * neuron_status["tau_syn_ex"]) * 1000.
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

nest.Connect(noise, ex_pop + in_pop, conn_spec={"rule": "all_to_all"},
                                     syn_spec={"synapse_model": "static_synapse", "weight": g[0]})

nest.Connect(ex_pop, ex_pop + in_pop, conn_spec={"rule": "fixed_indegree",
                                                 "indegree": int(0.1 * NE)},
                                      syn_spec={"synapse_model": "static_synapse", "weight": g[0]})

nest.Connect(in_pop, ex_pop + in_pop, conn_spec={"rule": "fixed_indegree",
                                                "indegree": int(0.1 * NI)},
                                      syn_spec={"synapse_model": "static_synapse", "weight": g[1]})


ex_spikerec = nest.Create("spike_recorder", NE)
in_spikerec = nest.Create("spike_recorder", NI)
nest.Connect(ex_pop, ex_spikerec, "one_to_one")
nest.Connect(in_pop, in_spikerec, "one_to_one")

nest.Simulate(simtime)

ex_spikes = nest.GetStatus(ex_spikerec, "events")
in_spikes = nest.GetStatus(in_spikerec, "events")
ex_spikes = np.concatenate([a["times"] for a in ex_spikes])
in_spikes = np.concatenate([a["times"] for a in in_spikes])
ex_isi = [np.ediff1d(a) for a in ex_spikes]
ex_isi_cat = np.concatenate(ex_isi)

bins = np.arange(0, simtime+0.1, 0.1)- 0.001
ex_hist, _ = np.histogram(ex_spikes, bins=bins)
in_hist, _ = np.histogram(in_spikes, bins=bins)
ex_hist = ex_hist.reshape((-1, 5)).sum(1)
in_hist = in_hist.reshape((-1, 5)).sum(1)

hists = np.stack((ex_hist, in_hist)).T
#np.save("example_hists.npy", hists)






