# READ ME

The python scripts/functions stored in this repository are a Key Resource for:

Rideaux, Storrs, Maiello & Welchman (2021) How multisensory neurons solve causal
inference. *Proceedings of the National Academy of Sciences*

This code can be used to train and test the MultiNet neural network, to
reproduce the results presented in the paper.

**The training image sequences required to train the network, in addition to
pre-trained networks and results files, can be downloaded from:

https://osf.io/7n8uj/**

### Instructions

#### Training MultiNet

You can use the pre-trained MultiNets included in the `results` folder, or you
can train new MultiNets.

The code used to train MultiNet include:
  (in-house libraries)
- `params.py` : defines core network + stimulus parameters
- `multiNet.py` : defines network architecture

  (training script)
- `train_network.py` : trains a MultiNet network

To train a new MultiNet, ensure the training image sequence files are stored in
a subfolder named "dataset", within the same folder as the scripts, and run the
`train_network.py` script.

This will initiate the training process. The time it will take to finish will
depend on the capabilities of your machine.

#### Testing MultiNet

If there are 5 valid instantiations of MultiNet in the `results` folder, either
the original pre-trained networks or a new ones you have trained, you can test
its estimates and properties.

The scripts used to test MultiNet include:
  (in house libraries)
- `params.py` : defines core network + stimulus parameters
- `multiNet.py` : defines network architecture
- `dot_stim_gen.py` : functions for generating dot motion stimuli

  (analysis scripts)
- `reweighting.py` : Figure 2b-c
- `decision-bias.py` : Figure 2e-g
- `congruency.py` : Figure 3
- `causal-inference.py` : Figure 4a
- `tuning.py` : Figure 4b-c
- `lesion.py` : Figure 4d

  (results visualization scripts)
- `reweighting_plot.py` : Figure 2b-c
- `decision-bias_plot.py` : Figure 2e-g
- `congruency_plot.py` : Figure 3 & 4a,
- `tuning.py_plot` : Figure 4b-c
- `lesion_plot.py` : Figure 4d

To test MultiNet, ensure that 5 valid network instantiations are stored in
a subfolder named "results", within the same folder as the scripts, and run an
analysis script. Once this is complete, run the corresponding visualisation
script.
