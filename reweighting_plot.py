''' Script to test the cue-reliability based reweighting of the MultiNet, as
implemented in Figure 2 Rideaux, Storrs, Maiello and Welchman, Proceedings of
the National Academy of Sciences, 2021

** There must be 5 "reweighting" results saved in the 'results' folder prior
to running this visualization script. **

[DEPENDENCIES]
+ numpy==1.15.4
+ scipy
+ pickle
+ matplotlib

[EXAMPLE]
To run the script, please launch your terminal, move to the <MultiNet> folder
and run the following command:
>> python reweighting_plot.py

'''

#  Helper libraries
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# In-house libraries
import params
from dot_stim_gen import cart2pol


def sigmoid(x, mu, sigma):
    y = 1/(1 + np.exp(-(1/sigma)*(x-mu)))
    return(y)

# Define parameters
nnParams = params.nnParams()
sParams = params.sParams()
n_networks = 5

# Load results
for n_idx in range(n_networks):
    print('Loading results/' + 'reweighting' + '[' + str(n_idx) + ']')
    data = np.load('results/' + 'reweighting' + '[' + str(n_idx) + ']',
                   encoding='latin1', allow_pickle=True)
    for k in data.keys():exec("{0}=data[\'{0}\']".format(k))
    del data

    if n_idx==0:
        est_vis = np.empty(np.insert(vis.shape, 4, n_networks))
        est_vest = np.empty(np.insert(vest.shape, 4, n_networks))
        est_combined = np.empty(np.insert(combined.shape, 5, n_networks))
    est_vis[:, :, :, :, n_idx] = vis
    est_vest[:, :, :, :, n_idx] = vest
    est_combined[:, :, :, :, :, n_idx] = combined

# Plot single-cue results
colours = ['k', 'C2', 'C1']
popt = np.empty([5, 2])  # fit parameters placeholder
initial_guess = [0, 10]
maxfev = 5000
x = np.linspace(direction.min(), direction.max(), 100)*180/np.pi
plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = .5
ax = plt.subplot(441)
y = np.mean(cart2pol(est_vest[:, 0, :, 0, :], est_vest[:, 0, :, 2, :])[
            1] < np.pi/2, axis=0).mean(axis=1)  # binarize estimates
popt[0, :], _ = curve_fit(sigmoid, direction*180/np.pi,
                          y, method='lm', p0=initial_guess, maxfev=maxfev)  # fit sigmoid
ax.plot(x, sigmoid(x, *popt[0, :]), '-',
        color=(217/255, 139/255, 110/255), linewidth=.5) # plot model fit
ax.plot(direction*180/np.pi, y, 'o', markerfacecolor=(217/255,
                                                      139/255, 110/255), markeredgecolor='w', markersize=4)  # plot data
y = np.mean(cart2pol(est_vis[:, 0, :, 0, :], est_vis[:, 0, :, 2, :])[
            1] < np.pi/2, axis=0).mean(axis=1)  # binarize estimates
popt[1, :], _ = curve_fit(sigmoid, direction*180/np.pi,
                          y, method='lm', p0=initial_guess, maxfev=maxfev)  # fit sigmoid
ax.plot(x, sigmoid(x, *popt[1, :]), '-',
        color=(104/255, 135/255, 180/255), linewidth=.5) # plot model fit
ax.plot(direction*180/np.pi, y, 's', markerfacecolor=(104/255,
                                                      135/255, 180/255), markeredgecolor='w', markersize=4) # plot data
y = np.mean(cart2pol(est_vis[:, 1, :, 0, :], est_vis[:, 1, :, 2, :])[
            1] < np.pi/2, axis=0).mean(axis=1)  # binarize estimates
popt[2, :], _ = curve_fit(sigmoid, direction*180/np.pi,
                          y, method='lm', p0=initial_guess, maxfev=maxfev)  # fit sigmoid
ax.plot(x, sigmoid(x, *popt[2, :]), '--',
        color=(176/255, 208/255, 254/255), linewidth=.5) # plot model fit
ax.plot(direction*180/np.pi, y, 'v', markerfacecolor=(176/255,
                                                      208/255, 254/255), markeredgecolor='w', markersize=4) # plot data
# Clean up plot
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_bounds(0, 1)
ax.spines['bottom'].set_bounds(x.min(), x.max())
ax.set_xlim([ax.spines['bottom'].get_bounds()[
            0]-np.diff(ax.spines['bottom'].get_bounds())*.1, ax.spines['bottom'].get_bounds()[1]])
ax.set_ylim([ax.spines['left'].get_bounds()[
            0]-np.diff(ax.spines['left'].get_bounds())*.1, ax.spines['left'].get_bounds()[1]])
ax.xaxis.set_tick_params(width=.5)
ax.yaxis.set_tick_params(width=.5)
ax.set_xticks([-10, 0, 10])

y = np.mean(cart2pol(est_combined[:, 0, 0, :, 0, :], est_combined[:, 0, 0, :, 2, :])[
            1] < np.pi/2, axis=0).mean(axis=1)  # binarize estimates
popt[3, :], _ = curve_fit(sigmoid, direction*180/np.pi,
                          y, method='lm', p0=initial_guess, maxfev=maxfev)  # fit sigmoid
y = np.mean(cart2pol(est_combined[:, 1, 0, :, 0, :], est_combined[:, 1, 0, :, 2, :])[
            1] < np.pi/2, axis=0).mean(axis=1)  # binarize estimates
popt[4, :], _ = curve_fit(sigmoid, direction*180/np.pi,
                          y, method='lm', p0=initial_guess, maxfev=maxfev)  # fit sigmoid
# calculate predicted sensitivity
pred_low = np.sqrt((popt[0, 1]**2*popt[2, 1]**2)/(popt[0, 1]**2+popt[2, 1]**2))
pred_high = np.sqrt((popt[0, 1]**2*popt[1, 1]**2)
                    / (popt[0, 1]**2+popt[1, 1]**2))  # calculate predicted sensitivity

# Plot sensitivity results
ax = plt.subplot(449)
ax.plot([0, 1], [popt[0, 1], popt[0, 1]]/popt[0, 1],
        '-o',color='#D98B6E', markersize=6)
ax.plot([0, 1], [popt[2, 1], popt[1, 1]]/popt[0, 1],
        '-o',color='#6887B4', markersize=6)
ax.plot([0, 1], [pred_low, pred_high]/popt[0, 1],
        '--o',color='#2B854D', markerfacecolor='w', markersize=6)
ax.plot([0, 1], [popt[4, 1], popt[3, 1]]/popt[0, 1],
        '-o',color='#2B854D', markersize=6)

# Clean up plot
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_bounds(ax.get_ylim())
ax.spines['bottom'].set_bounds(0, 1)
ax.set_xlim([ax.spines['bottom'].get_bounds()[
            0]-np.diff(ax.spines['bottom'].get_bounds())*.1, ax.spines['bottom'].get_bounds()[1]])
ax.set_ylim([ax.spines['left'].get_bounds()[
            0]-np.diff(ax.spines['left'].get_bounds())*.1, ax.spines['left'].get_bounds()[1]])
ax.xaxis.set_tick_params(width=.5)
ax.yaxis.set_tick_params(width=.5)
ax.set_xticks([])

# Plot combined cue results (high reliability vis)
ax = plt.subplot(443)
for i in range(3):
    y = np.mean(cart2pol(est_combined[:, 0, i, :, 0, :], est_combined[:, 0, i, :, 2, :])[
                1] < np.pi/2, axis=0).mean(axis=1)  # binarize estimates
    popt, _ = curve_fit(sigmoid, direction*180/np.pi, y,
                        method='lm', p0=initial_guess, maxfev=maxfev)  # fit sigmoid
    ax.plot(x, sigmoid(x, *popt), '-'+colours[i], linewidth=.5) # plot model fit
    ax.plot(direction*180/np.pi, y, 'o',
            markerfacecolor=colours[i], markeredgecolor='w', markersize=4) # plot data
# Clean up plot
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_bounds(0, 1)
ax.spines['bottom'].set_bounds(x.min(), x.max())
ax.set_xlim([ax.spines['bottom'].get_bounds()[
            0]-np.diff(ax.spines['bottom'].get_bounds())*.1, ax.spines['bottom'].get_bounds()[1]])
ax.set_ylim([ax.spines['left'].get_bounds()[
            0]-np.diff(ax.spines['left'].get_bounds())*.1, ax.spines['left'].get_bounds()[1]])
ax.xaxis.set_tick_params(width=.5)
ax.yaxis.set_tick_params(width=.5)
ax.set_xticks([-10, 0, 10])

# Plot combined cue results (low reliability vis)
ax = plt.subplot(4, 4, 11)
for i in range(3):
    y = np.mean(cart2pol(est_combined[:, 1, i, :, 0, :], est_combined[:, 1, i, :, 2, :])[
                1] < np.pi/2, axis=0).mean(axis=1)  # binarize estimates
    popt, _ = curve_fit(sigmoid, direction*180/np.pi, y,
                        method='lm', p0=initial_guess, maxfev=maxfev)  # fit sigmoid
    ax.plot(x, sigmoid(x, *popt), '-'+colours[i], linewidth=.5) # plot model fit
    ax.plot(direction*180/np.pi, y, 'o',
            markerfacecolor=colours[i], markeredgecolor='w', markersize=4) # plot data
# Clean up plot
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_bounds(0, 1)
ax.spines['bottom'].set_bounds(x.min(), x.max())
ax.set_xlim([ax.spines['bottom'].get_bounds()[
            0]-np.diff(ax.spines['bottom'].get_bounds())*.1, ax.spines['bottom'].get_bounds()[1]])
ax.set_ylim([ax.spines['left'].get_bounds()[
            0]-np.diff(ax.spines['left'].get_bounds())*.1, ax.spines['left'].get_bounds()[1]])
ax.xaxis.set_tick_params(width=.5)
ax.yaxis.set_tick_params(width=.5)
ax.set_xticks([-10, 0, 10])

plt.show()

print('done.')
