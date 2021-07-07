''' Script to measure the tuning of all MultiNet's units in response to
different visual and vestibular inputs, as implemented in Figure 4 Rideaux,
Storrs, Maiello and Welchman, Proceedings of the National Academy of Sciences,
2021

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
>> python tuning_plot.py

'''
#  Helper libraries
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt

# In-house libraries
import params

def gauss(x, amp, x0, xsigma, offset):
    return np.abs(amp) * np.exp(-.5*((x-x0)/xsigma)**2) + offset


# Define parameters
nnParams = params.nnParams()
n_networks = 5
n_boot = 50000
plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = .5

#  Load results
for n_idx in range(n_networks):
    data = np.load('results/tuning'
                   + '[' + str(n_idx) + ']', encoding='latin1', allow_pickle=True)
    for k in data.keys(): exec("{0}=data[\'{0}\']".format(k))
    del data

    if n_idx==0:
        vis = MST_resp_vis
        vest = MST_resp_vest
        pref_vis = MST_pref_vis
        pref_vest = MST_pref_vest
        wreg = reg
        wbin = bin
    else:
        vis = np.concatenate([vis, MST_resp_vis], axis=0)
        vest = np.concatenate([vest, MST_resp_vest], axis=0)
        pref_vis = np.concatenate([pref_vis, MST_pref_vis], axis=0)
        pref_vest = np.concatenate([pref_vest, MST_pref_vest], axis=0)
        wreg = np.concatenate([wreg, reg], axis=0)
        wbin = np.concatenate([wbin, bin], axis=0)

# Remove negative activations
V1_resp[V1_resp < 0] = 0
MT_resp[MT_resp < 0] = 0
VST_resp[VST_resp < 0] = 0
MST_resp_vis[MST_resp_vis < 0] = 0
MST_resp_vest[MST_resp_vest < 0] = 0
nd = V1_resp.shape[1]

# Define placeholders
c_index = np.empty([vis.shape[0], 4]) # congruency index
pci = np.empty([vis.shape[0], 4]) # significance
pref_V = np.empty([vis.shape[0], 2, 4]) # preferred velocity
p0 = [1, 0, 4, 0] # initial guess parameters
bounds = [[-np.inf, -4, -np.inf, -np.inf], [np.inf, 4, np.inf, np.inf]]
for u_idx in range(vis.shape[0]):
    for i in range(2):
        try:
            tmp, _ = curve_fit(gauss, dx, vis[u_idx, :, :, 0].mean(
                axis=(1-i)), method='trf', p0=p0, bounds=bounds)
            pref_V[u_idx, 0, i] = tmp[1]
        except:
            pref_V[u_idx, 0, i] = np.nan
        try:
            tmp, _ = curve_fit(gauss, dx, vest[u_idx, :, :, 0].mean(
                axis=(1-i)), method='trf', p0=p0, bounds=bounds)
            pref_V[u_idx, 1, i] = tmp[1]
        except:
            pref_V[u_idx, 1, i] = np.nan
        try:
            tmp, _ = curve_fit(gauss, dx, vis[u_idx, :, :, 1].mean(
                axis=(1-i)), method='trf', p0=p0, bounds=bounds)
            pref_V[u_idx, 0, i+2] = tmp[1]
        except:
            pref_V[u_idx, 0, i+2] = np.nan
        try:
            tmp, _ = curve_fit(gauss, dx, vest[u_idx, :, :, 1].mean(
                axis=(1-i)), method='trf', p0=p0, bounds=bounds)
            pref_V[u_idx, 1, i+2] = tmp[1]
        except:
            pref_V[u_idx, 1, i+2] = np.nan
        c_index[u_idx, i], pci[u_idx, i] = pearsonr(vis[u_idx, :, :, 0].mean(axis=(1-i)),
                                                    vest[u_idx, :, :, 0].mean(axis=(1-i)))
        c_index[u_idx, i+2], pci[u_idx, i+2] = pearsonr(vis[u_idx, :, :, 1].mean(axis=(1-i)),
                                                        vest[u_idx, :, :, 1].mean(axis=(1-i)))

cong = (pci < .05) & (c_index > 0) # congruent units
oppo = (pci < .05) & (c_index < 0) # opposite units

# Plot results (weights between MSTd and regression units)
f0 = plt.figure()
ax = plt.subplot(441)
barWidth = 0.33
yc = np.empty(2)
yo = np.empty(2)
ycerr = np.empty([2, 2])
yoerr = np.empty([2, 2])
abs_reg = np.abs(wreg)
for i in range(2):
    yc[i] = abs_reg[:, i*4:(i+1)*4].flatten()[cong.flatten()].mean()
    yo[i] = abs_reg[:, i*4:(i+1)*4].flatten()[oppo.flatten()].mean()
    x = abs_reg[:, i*4:(i+1)*4].flatten()[cong.flatten()]
    ycerr[i, :] = np.percentile(
        x[np.random.randint(0, len(x)-1, [len(x), n_boot])].mean(axis=0), [5, 95])
    x = abs_reg[:, i*4:(i+1)*4].flatten()[oppo.flatten()]
    yoerr[i, :] = np.percentile(
        x[np.random.randint(0, len(x)-1, [len(x), n_boot])].mean(axis=0), [5, 95])
r1 = np.arange(len(yc))
r2 = [x + barWidth for x in r1]
ax.bar(r1, yc, color='dodgerblue', width=barWidth)
ax.bar(r2, yo, color='mediumvioletred', width=barWidth)
for i in range(2):
    ax.plot([r1[i], r1[i]], ycerr[i, :], '-k', linewidth=1)
    ax.plot([r2[i], r2[i]], yoerr[i, :], '-k', linewidth=1)
ax.set_xticks([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_bounds(0, .12)
ax.xaxis.set_tick_params(width=.5)
ax.yaxis.set_tick_params(width=.5)
ax.set_yticks([0., .04, .08, .12])
ax.set_ylim([ax.spines['left'].get_bounds()[
            0]-np.diff(ax.spines['left'].get_bounds())*.1, ax.spines['left'].get_bounds()[1]])

# Plot results (weights between MSTd and binary units)
ax = plt.subplot(445)
yc = wbin.flatten()[cong.flatten()].mean()
yo = wbin.flatten()[oppo.flatten()].mean()
x = wbin.flatten()[cong.flatten()]
ycerr = np.percentile(
    x[np.random.randint(0, len(x)-1, [len(x), n_boot])].mean(axis=0), [5, 95])
x = wbin.flatten()[oppo.flatten()]
yoerr = np.percentile(
    x[np.random.randint(0, len(x)-1, [len(x), n_boot])].mean(axis=0), [5, 95])
ax.plot([-.5, .5], [0, 0], '-k', linewidth=.5)
ax.bar(-.25, yc, color='dodgerblue', width=.5)
ax.bar(.25, yo, color='mediumvioletred', width=.5)
ax.plot([-.25, -.25], ycerr, '-k', linewidth=1)
ax.plot([.25, .25], yoerr, '-k', linewidth=1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_bounds(-.06, .06)
ax.xaxis.set_tick_params(width=.5)
ax.yaxis.set_tick_params(width=.5)
ax.set_xticks([])
ax.set_yticks([-.05, 0., .05])
plt.xlim(-.6, 2.4)

# Define placeholders
bregc = np.empty([len(speed), 2, 2])
brego = np.empty([len(speed), 2, 2])
bregc_ci = np.empty([len(speed), 2, 2, 2])
brego_ci = np.empty([len(speed), 2, 2, 2])

# Sort weights a function of preferred velocity
for i in range(len(speed)):
    for v in range(4):
        if v == 0:
            regc_vis = wreg[cong[:, v]*(pref_vis[:, v] == i), v::4]
            regc_vest = wreg[cong[:, v]*(pref_vest[:, v] == i), v::4]
            rego_vis = wreg[oppo[:, v]*(pref_vis[:, v] == i), v::4]
            rego_vest = wreg[oppo[:, v]*(pref_vest[:, v] == i), v::4]
            if i == 0:
                regc = wreg[cong[:, v], v::4]
                rego = wreg[oppo[:, v], v::4]
                pvc_vis = pref_V[cong[:, v], 0, v]
                pvc_vest = pref_V[cong[:, v], 1, v]
                pvo_vis = pref_V[oppo[:, v], 0, v]
                pvo_vest = pref_V[oppo[:, v], 1, v]
        else:
            regc_vis = np.concatenate(
                [regc_vis, wreg[cong[:, v]*(pref_vis[:, v] == i), v::4]], axis=0)
            regc_vest = np.concatenate(
                [regc_vest, wreg[cong[:, v]*(pref_vest[:, v] == i), v::4]], axis=0)
            rego_vis = np.concatenate(
                [rego_vis, wreg[oppo[:, v]*(pref_vis[:, v] == i), v::4]], axis=0)
            rego_vest = np.concatenate(
                [rego_vest, wreg[oppo[:, v]*(pref_vest[:, v] == i), v::4]], axis=0)
            if i == 0:
                regc = np.concatenate([regc, wreg[cong[:, v], v::4]], axis=0)
                rego = np.concatenate([rego, wreg[oppo[:, v], v::4]], axis=0)
                pvc_vis = np.concatenate(
                    [pvc_vis, pref_V[cong[:, v], 0, v]], axis=0)
                pvc_vest = np.concatenate(
                    [pvc_vest, pref_V[cong[:, v], 1, v]], axis=0)
                pvo_vis = np.concatenate(
                    [pvo_vis, pref_V[oppo[:, v], 0, v]], axis=0)
                pvo_vest = np.concatenate(
                    [pvo_vest, pref_V[oppo[:, v], 1, v]], axis=0)
    bregc[i, :, 0] = regc_vis.mean(axis=0)
    bregc[i, :, 1] = regc_vest.mean(axis=0)
    brego[i, :, 0] = rego_vis.mean(axis=0)
    brego[i, :, 1] = rego_vest.mean(axis=0)
    # Calculate bootstrapped confidence intervals
    for j in range(2):
        x = regc_vis[:, j]
        try:
            bregc_ci[i, :, j, 0] = np.percentile(
                x[np.random.randint(0, len(x)-1, [len(x), n_boot])].mean(axis=0), [5, 95])
        except:
            print('insufficient data points to calculate confidence intervals.')
        x = regc_vest[:, j]
        try:
            bregc_ci[i, :, j, 1] = np.percentile(
                x[np.random.randint(0, len(x)-1, [len(x), n_boot])].mean(axis=0), [5, 95])
        except:
            print('insufficient data points to calculate confidence intervals.')
        x = rego_vis[:, j]
        try:
            brego_ci[i, :, j, 0] = np.percentile(
                x[np.random.randint(0, len(x)-1, [len(x), n_boot])].mean(axis=0), [5, 95])
        except:
            print('insufficient data points to calculate confidence intervals.')
        x = rego_vest[:, j]
        try:
            brego_ci[i, :, j, 1] = np.percentile(
                x[np.random.randint(0, len(x)-1, [len(x), n_boot])].mean(axis=0), [5, 95])
        except:
            print('insufficient data points to calculate confidence intervals.')

# Plot scatterplot of weights
cmap = plt.cm.get_cmap('PiYG')
ax = plt.subplot(443)
ax.axis('equal')
col = regc[:, 0]
col /= np.abs(np.abs(wreg).max())
plt.scatter(pvc_vis, pvc_vest, c=cmap(col/2+.5),
            alpha=.33, edgecolor='none', s=5) # plot data
# Clean up plot
plt.ylim([-4, 4])
plt.xlim([-4, 4])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_bounds(ax.get_ylim())
ax.spines['bottom'].set_bounds(ax.get_xlim())
ax.set_xlim([ax.spines['bottom'].get_bounds()[
            0]-np.diff(ax.spines['bottom'].get_bounds())*.1, ax.spines['bottom'].get_bounds()[1]])
ax.set_ylim([ax.spines['left'].get_bounds()[
            0]-np.diff(ax.spines['left'].get_bounds())*.1, ax.spines['left'].get_bounds()[1]])
ax.xaxis.set_tick_params(width=.5)
ax.yaxis.set_tick_params(width=.5)
ax.set_yticks([-4, 0, 4])
ax.set_xticks([-4, 0, 4])

ax = plt.subplot(444)
ax.axis('equal')
col = regc[:, 1]
col /= np.abs(np.abs(wreg).max())
plt.scatter(pvc_vis, pvc_vest, c=cmap(col/2+.5),
            alpha=.33, edgecolor='none', s=5) # plot data
# Clean up plot
plt.ylim([-4, 4])
plt.xlim([-4, 4])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_bounds(ax.get_ylim())
ax.spines['bottom'].set_bounds(ax.get_xlim())
ax.set_xlim([ax.spines['bottom'].get_bounds()[
            0]-np.diff(ax.spines['bottom'].get_bounds())*.1, ax.spines['bottom'].get_bounds()[1]])
ax.set_ylim([ax.spines['left'].get_bounds()[
            0]-np.diff(ax.spines['left'].get_bounds())*.1, ax.spines['left'].get_bounds()[1]])
ax.xaxis.set_tick_params(width=.5)
ax.yaxis.set_tick_params(width=.5)
ax.set_yticks([-4, 0, 4])
ax.set_xticks([-4, 0, 4])
ax = plt.subplot(447)
ax.axis('equal')
col = rego[:, 0]
col /= np.abs(np.abs(wreg).max())
plt.scatter(pvo_vis, pvo_vest, c=cmap(col/2+.5),
            alpha=.33, edgecolor='none', s=5) # plot data
# Clean up plot
plt.ylim([-4, 4])
plt.xlim([-4, 4])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_bounds(ax.get_ylim())
ax.spines['bottom'].set_bounds(ax.get_xlim())
ax.set_xlim([ax.spines['bottom'].get_bounds()[
            0]-np.diff(ax.spines['bottom'].get_bounds())*.1, ax.spines['bottom'].get_bounds()[1]])
ax.set_ylim([ax.spines['left'].get_bounds()[
            0]-np.diff(ax.spines['left'].get_bounds())*.1, ax.spines['left'].get_bounds()[1]])
ax.xaxis.set_tick_params(width=.5)
ax.yaxis.set_tick_params(width=.5)
ax.set_yticks([-4, 0, 4])
ax.set_xticks([-4, 0, 4])

ax = plt.subplot(448)
ax.axis('equal')
col = rego[:, 1]
col /= np.abs(np.abs(wreg).max())
plt.scatter(pvo_vis, pvo_vest, c=cmap(col/2+.5),
            alpha=.33, edgecolor='none', s=5) # plot data
# Clean up plot
plt.ylim([-4, 4])
plt.xlim([-4, 4])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_bounds(ax.get_ylim())
ax.spines['bottom'].set_bounds(ax.get_xlim())
ax.set_xlim([ax.spines['bottom'].get_bounds()[
            0]-np.diff(ax.spines['bottom'].get_bounds())*.1, ax.spines['bottom'].get_bounds()[1]])
ax.set_ylim([ax.spines['left'].get_bounds()[
            0]-np.diff(ax.spines['left'].get_bounds())*.1, ax.spines['left'].get_bounds()[1]])
ax.xaxis.set_tick_params(width=.5)
ax.yaxis.set_tick_params(width=.5)
ax.set_yticks([-4, 0, 4])
ax.set_xticks([-4, 0, 4])

# Plot binned weights
ax = plt.subplot(449)
ax.plot([-4, 4], [0, 0], '-k', linewidth=.5)
ax.plot(speed, bregc[:, 0, :], linewidth=1)
for i in range(len(speed)):
    for ii in range(2):
        ax.plot([speed[i], speed[i]], bregc_ci[i, :, 0, ii],
                'C'+str(ii), linewidth=1) # plot data
# Clean up plot
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_bounds(-4, 4)
ax.spines['left'].set_bounds(-.15, .15)
ax.set_xticks([-4, -2, 0, 2, 4])
ax.set_yticks([-.1, 0, .1])
ax.set_xlim([ax.spines['bottom'].get_bounds()[
            0]-np.diff(ax.spines['bottom'].get_bounds())*.1, ax.spines['bottom'].get_bounds()[1]])
ax.set_ylim([ax.spines['left'].get_bounds()[
            0]-np.diff(ax.spines['left'].get_bounds())*.1, ax.spines['left'].get_bounds()[1]])
ax.xaxis.set_tick_params(width=.5)
ax.yaxis.set_tick_params(width=.5)

ax = plt.subplot(4, 4, 10)
ax.plot([-4, 4], [0, 0], '-k', linewidth=.5)
ax.plot(speed, bregc[:, 1, :], linewidth=1)
for i in range(len(speed)):
    for ii in range(2):
        ax.plot([speed[i], speed[i]], bregc_ci[i, :, 1, ii],
                'C'+str(ii), linewidth=1) # plot data
# Clean up plot
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_bounds(-4, 4)
ax.spines['left'].set_bounds(-.15, .15)
ax.set_xticks([-4, -2, 0, 2, 4])
ax.set_yticks([-.1, 0, .1])
ax.set_xlim([ax.spines['bottom'].get_bounds()[
            0]-np.diff(ax.spines['bottom'].get_bounds())*.1, ax.spines['bottom'].get_bounds()[1]])
ax.set_ylim([ax.spines['left'].get_bounds()[
            0]-np.diff(ax.spines['left'].get_bounds())*.1, ax.spines['left'].get_bounds()[1]])
ax.xaxis.set_tick_params(width=.5)
ax.yaxis.set_tick_params(width=.5)

ax = plt.subplot(4, 4, 13)
ax.plot([-4, 4], [0, 0], '-k', linewidth=.5)
ax.plot(speed, brego[:, 0, :], linewidth=1)
for i in range(len(speed)):
    for ii in range(2):
        ax.plot([speed[i], speed[i]], brego_ci[i, :, 0, ii],
                'C'+str(ii), linewidth=1) # plot data
# Clean up plot
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_bounds(-4, 4)
ax.spines['left'].set_bounds(-.15, .15)
ax.xaxis.set_tick_params(width=.5)
ax.yaxis.set_tick_params(width=.5)
ax.set_xticks([-4, -2, 0, 2, 4])
ax.set_yticks([-.1, 0, .1])
ax.set_xlim([ax.spines['bottom'].get_bounds()[
            0]-np.diff(ax.spines['bottom'].get_bounds())*.1, ax.spines['bottom'].get_bounds()[1]])
ax.set_ylim([ax.spines['left'].get_bounds()[
            0]-np.diff(ax.spines['left'].get_bounds())*.1, ax.spines['left'].get_bounds()[1]])
ax.xaxis.set_tick_params(width=.5)
ax.yaxis.set_tick_params(width=.5)

ax = plt.subplot(4, 4, 14)
ax.plot([-4, 4], [0, 0], '-k', linewidth=.5)
ax.plot(speed, brego[:, 1, :], linewidth=1)
for i in range(len(speed)):
    for ii in range(2):
        ax.plot([speed[i], speed[i]], brego_ci[i, :, 1, ii],
                'C'+str(ii), linewidth=1) # plot data
# Clean up plot
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_bounds(-4, 4)
ax.spines['left'].set_bounds(-.15, .15)
ax.set_xticks([-4, -2, 0, 2, 4])
ax.set_yticks([-.1, 0, .1])
ax.set_xlim([ax.spines['bottom'].get_bounds()[
            0]-np.diff(ax.spines['bottom'].get_bounds())*.1, ax.spines['bottom'].get_bounds()[1]])
ax.set_ylim([ax.spines['left'].get_bounds()[
            0]-np.diff(ax.spines['left'].get_bounds())*.1, ax.spines['left'].get_bounds()[1]])
ax.xaxis.set_tick_params(width=.5)
ax.yaxis.set_tick_params(width=.5)

plt.show()
print('done.')
