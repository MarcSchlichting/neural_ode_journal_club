from neuron import h, gui
import matplotlib.pyplot as plt
import numpy as np

num_neurons = 10
times = []
voltages = []
for i in range(num_neurons):
# Set up a single neuron
    soma = h.Section(name="soma")
    soma.L = soma.diam = 10  # Set soma dimensions
    soma.insert("hh")
    soma.insert("pas")  # Insert Hodgkin-Huxley dynamics


    # Record time and voltage
    time = h.Vector()
    voltage = h.Vector()
    time.record(h._ref_t)  # Record simulation time
    voltage.record(soma(0.5)._ref_v)  # Record membrane potential at the center of the soma

    # Create a NetStim to generate a spike train
    netstim = h.NetStim()
    netstim.start = 0  # Start of the spike train (ms)
    netstim.number = 100  # Total number of spikes
    netstim.interval = 20  # Interval between spikes (ms)
    netstim.noise = 1  # Deterministic (0) or stochastic (1) spike train

    # Create a synapse on the soma
    syn = h.ExpSyn(soma(0.5))  # Exponential synapse at the center of soma
    syn.tau = 2.0  # Synaptic time constant (ms)
    syn.e = 0.0  # Reversal potential (mV)

    # Connect NetStim to the synapse
    netcon = h.NetCon(netstim, syn)
    netcon.weight[0] = 0.1  # Synaptic weight

    # Run the simulation
    h.tstop = 400  # Total simulation time (ms)
    h.run()
    
    times.append(time)
    voltages.append(voltage)

# Plot the results
np.savetxt("./data/hh_simulation_vm.txt",voltages)
np.savetxt("./data/hh_simulation_t.txt",times)

fig,axs = plt.subplots(num_neurons,1)
for i,(t,v) in enumerate(zip(times,voltages)):
    axs[i].plot(t, v)
# plt.axhline(-20, color="gray", linestyle="--", label="Spike Threshold")
# plt.title("Spike Train Simulation in a Single Neuron")
# plt.legend()
plt.grid()
plt.show()
