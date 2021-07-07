''' Script to measure MultiNet's estimates after artificially lesioning
congruent/opposite units, as implemented in Figure 4 Rideaux, Storrs,
Maiello and Welchman, Proceedings of the National Academy of Sciences, 2021

** There must be 5 valid "tuning" results saved in the 'results' folder prior to
running this visualization script. **

[DEPENDENCIES]
+ numpy==1.15.4
+ scipy
+ pickle
+ matplotlib

[EXAMPLE]
To run the script, please launch your terminal, move to the <MultiNet> folder
and run the following command:
>> python lesion_plot.py

'''
#  Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import pearsonr

# In-house libraries
import params

# Define parameters
nnParams = params.nnParams()
n_networks = 5
plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = .5

for n_idx in range(n_networks):
    data = np.load('results/lesion' + '[' + str(n_idx) + ']',
                    encoding='latin1', allow_pickle=True)
    if n_idx==0:
        vis = data['MST_resp_vis']
        vest = data['MST_resp_vest']
        combined = data['MST_resp_combined']
        wr = data['wreg']
        estimate = data['estimate']
    else:
        vis = np.concatenate([vis, data['MST_resp_vis']], axis=3)
        vest = np.concatenate([vest, data['MST_resp_vest']], axis=3)
        combined = np.concatenate(
            [combined, data['MST_resp_combined']], axis=4)
        wr = np.concatenate([wr, data['wreg']], axis=0)

# Threshold negative activations
vis[vis < 0] = 0
vest[vest < 0] = 0
combined[combined < 0] = 0

# Average activations across stimulus presentations
vis = vis.mean(axis=0)
vest = vest.mean(axis=0)

# Calculate congruency index
c_index = np.empty([vis.shape[2], 4])
pci = np.empty([vis.shape[2], 4])
for dim_idx in range(4):
    for u_idx in range(vis.shape[2]):
        c_index[u_idx, dim_idx], pci[u_idx, dim_idx] = pearsonr(
            vis[:, dim_idx, u_idx], vest[:, dim_idx, u_idx])

cong = (pci < .01) & (c_index > 0) # congruent units
oppo = (pci < .01) & (c_index < 0) # opposite units

# Define placeholders
est = np.empty([combined.shape[0], combined.shape[1], combined.shape[2], 2, 4])
estc = np.empty(
    [combined.shape[0], combined.shape[1], combined.shape[2], 2, 4])
esto = np.empty(
    [combined.shape[0], combined.shape[1], combined.shape[2], 2, 4])

# Calculate estimates from congruent/opposite unit subpopulations
for dim_idx in range(4):
    est[:, :, :, :, dim_idx] = np.dot(
        combined[:, :, :, dim_idx, :],
        wr[:, dim_idx::4])/(n_idx+1)
    estc[:, :, :, :, dim_idx] = np.dot(
        combined[:, :, :, dim_idx, cong[:, dim_idx]],
        wr[cong[:, dim_idx], dim_idx::4])/(n_idx+1)
    esto[:, :, :, :, dim_idx] = np.dot(
        combined[:, :, :, dim_idx, oppo[:, dim_idx]],
        wr[oppo[:, dim_idx], dim_idx::4])/(n_idx+1)

# Average estimates across stimulus repeats and velocity dims
est = est.mean(axis=4).mean(axis=0)
estc = estc.mean(axis=4).mean(axis=0)
esto = esto.mean(axis=4).mean(axis=0)

# Plot results
plt.subplot(441)
y = est[:, :, 0]
vlim = np.ceil(np.abs(y).max()*10)/10
plt.imshow(y, vmin=-vlim, vmax=vlim, origin='lower',
           cmap='RdBu', interpolation='bilinear')
plt.xticks([]), plt.yticks([]), plt.axis(
    'off'), plt.colorbar(ticks=[-vlim, 0, vlim])

plt.subplot(442)
y = estc[:, :, 0]
vlim = np.ceil(np.abs(y).max()*10)/10
plt.imshow(y, vmin=-vlim, vmax=vlim, origin='lower',
           cmap='RdBu', interpolation='bilinear')
plt.xticks([]), plt.yticks([]), plt.axis(
    'off'), plt.colorbar(ticks=[-vlim, 0, vlim])

plt.subplot(443)
y = esto[:, :, 0]
plt.imshow(y, vmin=-vlim, vmax=vlim, origin='lower',
           cmap='RdBu', interpolation='bilinear')
plt.xticks([]), plt.yticks([]), plt.axis(
    'off'), plt.colorbar(ticks=[-vlim, 0, vlim])

plt.subplot(444)
y1 = estc[:, :, 0]
y1 /= np.abs(y1.min()-y1.max())
y2 = esto[:, :, 0]
y2 /= np.abs(y2.min()-y2.max())
y = y2-y1
y /= np.abs(y).max()
vlim = np.ceil(np.abs(y).max()*10)/10
plt.imshow(y, vmin=-vlim, vmax=vlim, origin='lower',
           cmap='RdBu', interpolation='bilinear')
plt.xticks([]), plt.yticks([]), plt.axis(
    'off'), plt.colorbar(ticks=[-vlim, 0, vlim])

plt.subplot(445)
y = est[:, :, 1]
vlim = np.ceil(np.abs(y).max()*10)/10
plt.imshow(y, vmin=-vlim, vmax=vlim, origin='lower',
           cmap='RdBu', interpolation='bilinear')
plt.xticks([]), plt.yticks([]), plt.axis(
    'off'), plt.colorbar(ticks=[-vlim, 0, vlim])

plt.subplot(447)
y = esto[:, :, 1]
vlim = np.ceil(np.abs(y).max()*10)/10
plt.imshow(y, vmin=-vlim, vmax=vlim, origin='lower',
           cmap='RdBu', interpolation='bilinear')
plt.xticks([]), plt.yticks([]), plt.axis(
    'off'), plt.colorbar(ticks=[-vlim, 0, vlim])

plt.subplot(446)
y = estc[:, :, 1]
plt.imshow(y, vmin=-vlim, vmax=vlim, origin='lower',
           cmap='RdBu', interpolation='bilinear')
plt.xticks([]), plt.yticks([]), plt.axis(
    'off'), plt.colorbar(ticks=[-vlim, 0, vlim])

plt.subplot(448)
y1 = estc[:, :, 1]
y1 /= np.abs(y1.min()-y1.max())
y2 = esto[:, :, 1]
y2 /= np.abs(y2.min()-y2.max())
y = y2-y1
y /= np.abs(y).max()
vlim = np.ceil(np.abs(y).max()*10)/10
plt.imshow(y, vmin=-vlim, vmax=vlim, origin='lower',
           cmap='RdBu', interpolation='bilinear')
plt.xticks([]), plt.yticks([]), plt.axis(
    'off'), plt.colorbar(ticks=[-vlim, 0, vlim])

plt.show()
print('done.')
