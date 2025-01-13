import numpy as np
from pyHH.src.pyhh.models import HHModel
from pyHH.src.pyhh.simulations import Simulation
import matplotlib.pyplot as plt

# customize a neuron model if desired
model = HHModel()
model.gNa = 100  # typically 120
model.gK = 5  # typically 36
model.EK = -35  # typically -12

# customize a stimulus waveform
stim = np.zeros(20000)
stim[7000:13000] = 50  # add a square pulse

# simulate the model cell using the custom waveform
sim = Simulation(model)
sim.Run(stimulusWaveform=stim, stepSizeMs=0.01)
plt.plot(sim.times,sim.Vm)
plt.show()

print("stop")