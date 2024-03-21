import nest, random
import numpy as np
import matplotlib.pyplot as plt

nest.ResetKernel()
resolution = 0.1
nest.SetKernelStatus({'resolution': resolution})

simtime = 10500.
J = 40.
g = 5.5
NE = 8000
CE = int(NE * 0.1)
NI = 2000
CI = int(NI * 0.1)

neuron_params = {'V_m': 0.,
                 'V_th': 15.,
                 'C_m': 250.,
                 'g': 250. / 20., # g = C / tau
                 't_ref': 2.,
                 'V_reset': 0.,
                 'E_L': 0.,
                 'tau_syn': [2.],
                 'spike_dependent_threshold': False,
                 'after_spike_currents': False,
                 'adapting_threshold': False,
                 }


epop = nest.Create('glif_psc', NE,
                  params=neuron_params)

ipop = nest.Create('glif_psc', NI,
                  params=neuron_params)

nest.SetStatus(epop, [{'V_m': random.random() * 15} for _ in range(NE)])
nest.SetStatus(ipop, [{'V_m': random.random() * 15} for _ in range(NI)])

# dcg = nest.Create('dc_generator', params={'start':0.,
#                                           'stop':simtime,
#                                           'amplitude':250.})

noise = nest.Create('poisson_generator', params={'rate': 5000.})

mm = nest.Create('multimeter',
                 params={'interval': resolution,
                         'record_from': ['V_m', 'I', 'I_syn', 'threshold',
                                         'threshold_spike',
                                         'threshold_voltage',
                                         'ASCurrents_sum']})

esd = nest.Create('spike_detector')
isd = nest.Create('spike_detector')
nest.Connect(epop, esd)
nest.Connect(ipop, isd)

esyn_spec = {'weight': J, 'delay': 2., 'receptor_type': 1}
isyn_spec = {'weight': -g * J, 'delay': 2., 'receptor_type': 1}

nest.Connect(epop,
             epop,
             syn_spec=esyn_spec,
             conn_spec={'rule': 'fixed_indegree', 'indegree': CE})


nest.Connect(epop,
             ipop,
             syn_spec=esyn_spec,
             conn_spec={'rule': 'fixed_indegree', 'indegree': CE})

nest.Connect(ipop,
             ipop,
             syn_spec=isyn_spec,
             conn_spec={'rule': 'fixed_indegree', 'indegree': CI})

nest.Connect(ipop,
             epop,
             syn_spec=isyn_spec,
             conn_spec={'rule': 'fixed_indegree', 'indegree': CI})

nest.Connect(noise,
             epop,
             syn_spec=esyn_spec)

nest.Connect(noise,
             ipop,
             syn_spec=esyn_spec)

nest.Connect(mm, epop[:1])

nest.Simulate(simtime)

data = nest.GetStatus(mm)[0]['events']
senders = data['senders']

espike_data = nest.GetStatus(esd)[0]['events']
ispike_data = nest.GetStatus(isd)[0]['events']
espikes = espike_data['times']
ispikes = ispike_data['times']

bins = np.arange(0, simtime + 1, 0.1) - 0.05
ehist, _ = np.histogram(espikes, bins=bins)
ihist, _ = np.histogram(ispikes, bins=bins)
np.save('lif_hist.npy', np.stack((ehist, ihist)))
