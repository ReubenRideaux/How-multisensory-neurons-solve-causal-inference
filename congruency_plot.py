''' Script to plot the congruency results of MSTd units in the network, as
implemented in Figure 3 Rideaux, Storrs, Maiello and Welchman, Proceedings of
the National Academy of Sciences, 2021

** There must be a 5 valid "congruency" and "causal-inference" results saved in
the 'results' folder prior to running this visualization script. **

[DEPENDENCIES]
+ numpy
+ scipy
+ sklearn
+ matplotlib
+ pickle

[EXAMPLE]
To run the script, please launch your terminal, move to the <MultiNet> folder
and run the following command:
>> python congruency_plot.py

'''

#  Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import pearsonr
from sklearn.metrics import auc
from scipy.optimize import curve_fit

# In-house libraries
import params

def sigmoid(x, mu, sigma):
    y = 1/(1 + np.exp(-(1/sigma)*(x-mu)))
    return(y)

def sinewave(x, phase, amp, offset):
    y = amp*np.sin(x+phase)+offset
    return(y)

plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = .5

# Define parameters
nnParams = params.nnParams()
n_networks = 5
n_boot = 50000
n_crit = 40

# Load cogruency results
for n_idx in range(n_networks):
    # Load previously saved results
    print('Loading results/' + 'congruency' +  '[' + str(n_idx) + ']')
    data = np.load('results/' + 'congruency' + '[' + str(n_idx) + ']',
        encoding='latin1', allow_pickle=True)
    for k in data.keys(): exec("{0}=data[\'{0}\']".format(k))
    del data

    if n_idx==0:
        nd = dx.shape[1]
        vis = MST_resp_vis
        vest = MST_resp_vest
        combined = MST_resp_combined
    else:
        vis = np.concatenate([vis, MST_resp_vis], axis=3)
        vest = np.concatenate([vest, MST_resp_vest], axis=3)
        combined = np.concatenate([combined, MST_resp_combined], axis=3)
    direction *= 180/np.pi
    nMST = vis.shape[3]

# Calculate ROC curves
ROC_vis = np.empty([nd//2, n_crit, 2, nMST])
ROC_vest = np.empty([nd//2, n_crit, 2, nMST])
ROC_combined = np.empty([nd//2, n_crit, 2, nMST])
AUC = np.empty([nd//2, 3, nMST])
for u_idx in range(nMST):
    for d_idx in range(nd//2):
        for c_idx in range(n_crit):
            crit = np.linspace(vis[:, 1, :, u_idx].min(),
                               vis[:, 1, :, u_idx].max(), n_crit)
            ROC_vis[d_idx, c_idx, :, u_idx] = np.mean(
                vis[:, 1, [d_idx, -(d_idx+1)], u_idx] > crit[c_idx], axis=0)
            crit = np.linspace(vest[:, 1, :, u_idx].min(),
                               vest[:, 1, :, u_idx].max(), n_crit)
            ROC_vest[d_idx, c_idx, :, u_idx] = np.mean(
                vest[:, 1, [d_idx, -(d_idx+1)], u_idx] > crit[c_idx], axis=0)
            crit = np.linspace(combined[:, 1, :, u_idx].min(
            ), combined[:, 1, :, u_idx].max(), n_crit)
            ROC_combined[d_idx, c_idx, :, u_idx] = np.mean(
                combined[:, 1, [d_idx, -(d_idx+1)], u_idx] > crit[c_idx], axis=0)
        AUC[d_idx, 0, u_idx] = auc(ROC_vis[d_idx, :, 1, u_idx].transpose(
        ), ROC_vis[d_idx, :, 0, u_idx].transpose())
        AUC[d_idx, 1, u_idx] = auc(ROC_vest[d_idx, :, 1, u_idx].transpose(
        ), ROC_vest[d_idx, :, 0, u_idx].transpose())
        AUC[d_idx, 2, u_idx] = auc(ROC_combined[d_idx, :, 1, u_idx].transpose(
        ), ROC_combined[d_idx, :, 0, u_idx].transpose())
AUC[AUC < .5] = 1-AUC[AUC < .5]
AUC = np.concatenate([1-AUC, np.flip(AUC, axis=0)], axis=0)

# Fit sigmoid to performance
initial_guess = [0, 10]  # [mu,sigma]
popt = np.empty([4, 2, nMST])
for u_idx in range(nMST):
    for c_idx in range(3):
        try:
            popt[c_idx, :, u_idx], _ = curve_fit(
                sigmoid, direction[1, :], AUC[:, c_idx, u_idx], method='lm', p0=initial_guess)
        except:
            print('fitting failed...')
    popt[3, 1, u_idx] = np.sqrt(
        (popt[0, 1, u_idx]**2*popt[1, 1, u_idx]**2)/(popt[0, 1, u_idx]**2+popt[1, 1, u_idx]**2))

# Remove negative activation values
vis[vis < 0] = 0
vest[vest < 0] = 0
combined[combined < 0] = 0

# Define placeholders
r_vis = np.empty(nMST)
p_vis = np.empty(nMST)
r_vest = np.empty(nMST)
p_vest = np.empty(nMST)
c_index = np.empty(nMST)
pci = np.empty(nMST)

# Calculate congrueny (correlation) values
for u_idx in range(nMST):
    r_vis[u_idx], p_vis[u_idx] = pearsonr(
        direction[1, :], vis[:, 1, :, u_idx].mean(axis=0))
    r_vest[u_idx], p_vest[u_idx] = pearsonr(
        direction[1, :], vest[:, 1, :, u_idx].mean(axis=0))
c_index = r_vest*r_vis # congruency index
pci = (p_vest < .05)*(p_vis < .05) # significance
sen = popt[:, 1, :] # sensitivity
s_index = popt[2, 1, :]/popt[3, 1, :] # predicted sensitivity
cong = np.where((pci == 1) & (c_index > 0))[0] # congruent units
oppo = np.where((pci == 1) & (c_index < 0))[0] # opposite units
unclassed = np.where(pci == 0)[0] # unclassed units

# Plot results (sensitivity scatter plot)
f0 = plt.figure()
ax = plt.subplot(441)
ax.scatter(c_index[unclassed], s_index[unclassed],
           color='white', edgecolor='k', s=18, alpha=.33)
ax.scatter(c_index[oppo], s_index[oppo],
           color='mediumvioletred', edgecolor='k', s=18, alpha=.33)
ax.scatter(c_index[cong], s_index[cong], color='dodgerblue',
           edgecolor='k', s=18, alpha=.33)
ax.set_yscale("log")
ax.plot([-1, 1], [1, 1], '--k')
plt.ylim([.4, s_index[oppo].max()])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_bounds(.6, s_index[oppo].max())
ax.spines['bottom'].set_bounds(-1, 1)
ax.set_xlim([ax.spines['bottom'].get_bounds()[
            0]-np.diff(ax.spines['bottom'].get_bounds())*.1, ax.spines['bottom'].get_bounds()[1]])
ax.set_ylim([ax.spines['left'].get_bounds()[
            0]-np.diff(ax.spines['left'].get_bounds())*.1, ax.spines['left'].get_bounds()[1]])
ax.xaxis.set_tick_params(width=.5)
ax.yaxis.set_tick_params(width=.5)
ax.set_yticks([.6, .7, .8, .9, 1, 2, 3, 4, 5, 6, 7, 8, 9,
               10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

# Fit sinewave function to responses to determine preferred direction
pref_dir = np.empty([nMST, 2, 3])
cov = np.empty([nMST, 2, 3, 3])
for u_idx in range(nMST):
    pref_dir[u_idx, 0, :], cov[u_idx, 0, :, :] = curve_fit(
        sinewave, direction[0, :]*np.pi/180, vis[:, 0, :, u_idx].mean(axis=0), method='lm', p0=[0, 1, 0])
    pref_dir[u_idx, 1, :], cov[u_idx, 1, :, :] = curve_fit(
        sinewave, direction[0, :]*np.pi/180, vest[:, 0, :, u_idx].mean(axis=0), method='lm', p0=[0, 1, 0])
pref_dir[:, :, 0] *= np.sign(pref_dir[:, :, 1])

# Remove units with low direction selectivity
crit_vis = np.median(np.abs(pref_dir[:, 0, 1]))
crit_vest = np.median(np.abs(pref_dir[:, 1, 1]))
inc = (np.abs(pref_dir[:, 0, 1]) > crit_vis) * \
       (np.abs(pref_dir[:, 1, 1]) > crit_vest)
A = np.diff(pref_dir[inc, :, 0]*180/np.pi)
A = (A + 180) % 360 - 180

# Plot results (delta preferred direction histogram)
ax = plt.subplot(443)
ax.hist(np.abs(A), 10)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_bounds(0,75)
ax.spines['bottom'].set_bounds(0, 180)
ax.set_xlim([ax.spines['bottom'].get_bounds()[
            0]-np.diff(ax.spines['bottom'].get_bounds())*.1, ax.spines['bottom'].get_bounds()[1]])
ax.set_ylim([ax.spines['left'].get_bounds()[
            0]-np.diff(ax.spines['left'].get_bounds())*.1, ax.spines['left'].get_bounds()[1]])
ax.xaxis.set_tick_params(width=.5)
ax.yaxis.set_tick_params(width=.5)
ax.set_xticks([0, 90, 180])

# Plot results (sensitivity bar plots)
pci = pci.flatten()
c_index = c_index.flatten()
ax = plt.subplot(446)
yerr = np.empty([2,4])
for i in range(4):
    y = sen[i, oppo]
    yerr[:,i] = np.percentile(
        y[np.random.randint(0, len(y)-1, [len(y), n_boot])].mean(axis=0), [5, 95])
ax.bar(np.arange(4), np.nanmean(sen[:, oppo], axis=1), yerr=np.abs(yerr-np.nanmean(sen[:, oppo], axis=1)))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_bounds(0, plt.ylim()[1])
ax.set_ylim([ax.spines['left'].get_bounds()[
            0]-np.diff(ax.spines['left'].get_bounds())*.1, ax.spines['left'].get_bounds()[1]])
ax.yaxis.set_tick_params(width=.5)
ax.set_xticks([])

ax = plt.subplot(448)
for i in range(4):
    y = sen[i, cong]
    yerr[:,i] = np.percentile(
        y[np.random.randint(0, len(y)-1, [len(y), n_boot])].mean(axis=0), [5, 95])
ax.bar(np.arange(4), np.nanmean(sen[:, cong], axis=1), yerr=np.abs(yerr-np.nanmean(sen[:, cong], axis=1)))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_bounds(0, plt.ylim()[1])
ax.set_ylim([ax.spines['left'].get_bounds()[
            0]-np.diff(ax.spines['left'].get_bounds())*.1, ax.spines['left'].get_bounds()[1]])
ax.yaxis.set_tick_params(width=.5)
ax.set_xticks([])

# Define wrapped x axis
wrap_idx = np.arange(direction.shape[1]+1) % direction.shape[1]
wrap_dir = direction[0, wrap_idx]
wrap_dir[-1] *= -1

# Calculate sum prediction
sum_pred = vis+vest

# Plot results (single unit activation)
u_idx = 587 # may need to update this index for new network instantiations
ax = plt.subplot(449)
ax.plot(wrap_dir, vis[:, 0, wrap_idx, u_idx].mean(axis=0),
        'o-', markersize=4, linewidth=.5, markeredgecolor='w')
ax.plot(wrap_dir, vest[:, 0, wrap_idx, u_idx].mean(axis=0),
        'o-', markersize=4, linewidth=.5, markeredgecolor='w')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_bounds(0, plt.ylim()[1])
ax.spines['bottom'].set_bounds(-180, 180)
ax.set_xlim([ax.spines['bottom'].get_bounds()[
            0]-np.diff(ax.spines['bottom'].get_bounds())*.1, ax.spines['bottom'].get_bounds()[1]])
ax.set_ylim([ax.spines['left'].get_bounds()[
            0]-np.diff(ax.spines['left'].get_bounds())*.1, ax.spines['left'].get_bounds()[1]])
ax.xaxis.set_tick_params(width=.5)
ax.yaxis.set_tick_params(width=.5)
ax.set_xticks([-180, 0, 180])

u_idx = 215 # may need to update this index for new network instantiations
ax = plt.subplot(4, 4, 11)
ax.plot(wrap_dir, vis[:, 0, wrap_idx, u_idx].mean(axis=0),
        'o-', markersize=4, linewidth=.5, markeredgecolor='w')
ax.plot(wrap_dir, vest[:, 0, wrap_idx, u_idx].mean(axis=0),
        'o-', markersize=4, linewidth=.5, markeredgecolor='w')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_bounds(0, plt.ylim()[1])
ax.spines['bottom'].set_bounds(-180, 180)
ax.set_xlim([ax.spines['bottom'].get_bounds()[
            0]-np.diff(ax.spines['bottom'].get_bounds())*.1, ax.spines['bottom'].get_bounds()[1]])
ax.set_ylim([ax.spines['left'].get_bounds()[
            0]-np.diff(ax.spines['left'].get_bounds())*.1, ax.spines['left'].get_bounds()[1]])
ax.xaxis.set_tick_params(width=.5)
ax.yaxis.set_tick_params(width=.5)
ax.set_xticks([-180, 0, 180])
ax = plt.subplot(4, 4, 14)
ax.plot(direction[1, :], vis[:, 1, :, u_idx].mean(axis=0),
        'o-', markersize=4, linewidth=.5, markeredgecolor='w')
ax.plot(direction[1, :], vest[:, 1, :, u_idx].mean(axis=0),
        'o-', markersize=4, linewidth=.5, markeredgecolor='w')
ax.plot(direction[1, :], combined[:, 1, :, u_idx].mean(axis=0),
        'o-', markersize=4, linewidth=.5, markeredgecolor='w')
ax.plot(direction[1, :], sum_pred[:, 1, :, u_idx].mean(axis=0),
        'o-', markersize=4, linewidth=.5, markeredgecolor='w')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim([0, plt.ylim()[1]])
ax.spines['left'].set_bounds(0, plt.ylim()[1])
ax.spines['bottom'].set_bounds(direction[1, :].min(), direction[1, :].max())
ax.set_xlim([ax.spines['bottom'].get_bounds()[
            0]-np.diff(ax.spines['bottom'].get_bounds())*.1, ax.spines['bottom'].get_bounds()[1]])
ax.set_ylim([ax.spines['left'].get_bounds()[
            0]-np.diff(ax.spines['left'].get_bounds())*.1, ax.spines['left'].get_bounds()[1]])
ax.xaxis.set_tick_params(width=.5)
ax.yaxis.set_tick_params(width=.5)

# Load causal-inference results
for n_idx in range(n_networks):
    data = np.load('results/' + 'causal-inference'
                   + '[' + str(n_idx) + ']', encoding='latin1', allow_pickle=True)
    if n_idx == 0:
        vis = data['MST_resp_vis']
        vest = data['MST_resp_vest']
        combined = data['MST_resp_combined']
    else:
        vis = np.concatenate([vis, data['MST_resp_vis']], axis=2)
        vest = np.concatenate([vest, data['MST_resp_vest']], axis=2)
        combined = np.concatenate(
            [combined, data['MST_resp_combined']], axis=3)

# Average across stimulus repeats
combined = combined.mean(axis=0)
vis = vis.mean(axis=0)
vest = vest.mean(axis=0)

# Threshold negative activity values
combined[combined < 0] = 0
vis[vis < 0] = 0
vest[vest < 0] = 0

# Calculate activity range
vmin = (np.nanmean(combined[:, :, cong], axis=2)
        - np.nanmean(combined[:, :, oppo], axis=2)).min()
vmax = (np.nanmean(combined[:, :, cong], axis=2)
        - np.nanmean(combined[:, :, oppo], axis=2)).max()
vmm = np.abs([vmin, vmax]).max()

# Plot results
plt.subplot(4,4,16)
plt.imshow(np.nanmean(combined[:, :, oppo], axis=2)-np.nanmean(combined[:, :, cong],
                                                                  axis=2), origin='lower', vmin=-vmm, vmax=vmm, cmap='RdBu', interpolation='bilinear')
plt.axis('equal')
plt.axis('off')
plt.colorbar()

plt.show()

print('done.')
