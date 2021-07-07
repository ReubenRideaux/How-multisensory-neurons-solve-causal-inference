''' Script to MultiNet's estimates around the point of cue separation, as
implemented in Figure 2 Rideaux, Storrs, Maiello and Welchman, Proceedings of
the National Academy of Sciences, 2021

** There must be 5 valid "decision-bias" results saved in the 'results' folder
prior to running this visualization script. **

[DEPENDENCIES]
+ numpy
+ pickle
+ matplotlib

[EXAMPLE]
To run the script, please launch your terminal, move to the <MultiNet> folder
and run the following command:
>> python decision-bias_plot.py

'''
#  Helper libraries
import numpy as np
import pickle
import matplotlib.pyplot as plt

# In-house libraries
import params

# Define parameters
n_boot = 50000
n_networks = 5
plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = .5
window_len = 6 # rolling average window size
window = np.ones(window_len, 'd')/window_len
x_vest = np.linspace(-8, 8, 32)  # vestibular x dim

# Load results
for n_idx in range(n_networks):
    data = np.load('results/decision-bias' + '[' + str(n_idx) + ']',
                    encoding='latin1', allow_pickle=True)
    if n_idx==0:
        in_v = np.empty(np.insert(data['input_vest'].shape, 3, n_networks))
        out_r = np.empty(np.insert(data['out_r'].shape, 2, n_networks))
        out_b = np.empty(np.insert(data['out_b'].shape, 2, n_networks))
    in_v[:,:,:,n_idx] = data['input_vest']
    out_r[:,:,n_idx] = data['out_r']
    out_b[:,:,n_idx] = data['out_b']

# Sanity check that the network decided to combine cues ~%50 of the time.
print((out_b[:, 0, :] >= .5).mean(axis=0).mean(axis=0))

# Plot data (regression estimate histogram; combine cues)
ax = plt.subplot(441)
y1 = out_r[:, 0, :].flatten()[out_b[:, 0, :].flatten() < .5] # decision to combine
y2 = out_r[:, 0, :].flatten()[out_b[:, 0, :].flatten() > .5] # decision to separate
bins = np.linspace(out_r[:, 0, :].flatten().min(),
                   out_r[:, 0, :].flatten().max(), 32, endpoint=False) # histogram bins
hv = ax.hist(y1, bins=bins, alpha=.5, weights=np.ones(len(y1)) / len(y1)) # plot data
Y = np.convolve(window, hv[0], mode='same') # compute rolling average
ax.plot(hv[1][:-1]+np.diff(hv[1])/2, Y, 'C0', linewidth=.5) # plot rolling average
hv = ax.hist(y2, bins=bins, alpha=.5, weights=np.ones(len(y2)) / len(y2)) # plot data
Y = np.convolve(window, hv[0], mode='same')# compute rolling average
ax.plot(hv[1][:-1]+np.diff(hv[1])/2, Y, 'C1', linewidth=.5) # plot rolling average
# Clean up plot
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_bounds(0, .15)
ax.spines['bottom'].set_bounds(
    out_r[:, 0, :].flatten().min(), out_r[:, 0, :].flatten().max())
ax.set_xticks([0, 1, 2, 3])
ax.set_xlim([ax.spines['bottom'].get_bounds()[
            0]-np.diff(ax.spines['bottom'].get_bounds())*.1, ax.spines['bottom'].get_bounds()[1]])
ax.set_ylim([ax.spines['left'].get_bounds()[
            0]-np.diff(ax.spines['left'].get_bounds())*.1, ax.spines['left'].get_bounds()[1]])
ax.xaxis.set_tick_params(width=.5)
ax.yaxis.set_tick_params(width=.5)

# Plot data (bar plot)
ax = plt.subplot(4, 8, 12)
# Compute means and bootstrapped confidence intervals
y1err = np.percentile(y1[np.random.randint(0, len(y1)-1, [len(y1), n_boot])].mean(axis=0), [5, 95])
ax.bar(-.25, y1.mean(), color='C0', width=.5)
ax.plot([-.25,-.25],y1err)
y2err = np.percentile(y2[np.random.randint(0, len(y2)-1, [len(y2), n_boot])].mean(axis=0), [5, 95])
ax.bar(.25, y2.mean(), color='C1', width=.5)
ax.plot([.25,.25],y2err)
# Clean up plot
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_bounds(0, 2)
ax.spines['bottom'].set_bounds(-.5, .5)
ax.set_xlim([ax.spines['bottom'].get_bounds()[
            0]-np.diff(ax.spines['bottom'].get_bounds())*.1, ax.spines['bottom'].get_bounds()[1]])
ax.set_xticks([])
ax.set_yticks([0, 1, 2])

# Plot data (regression estimate histogram; separate cues)
ax = plt.subplot(443)
y1 = out_r[:, 4, :].flatten()[out_b[:, 0, :].flatten() < .5] # decision to combine
y2 = out_r[:, 4, :].flatten()[out_b[:, 0, :].flatten() > .5] # decision to separate
bins = np.linspace(out_r[:, 4, :].flatten().min(),
                   out_r[:, 4, :].flatten().max(), 32, endpoint=False) # histogram bins
hv = ax.hist(y1, bins=bins, alpha=.5, weights=np.ones(len(y1)) / len(y1)) # plot data
Y = np.convolve(window, hv[0], mode='same') # compute rolling average
ax.plot(hv[1][:-1]+np.diff(hv[1])/2, Y, 'C0', linewidth=.5) # plot rolling average
hv = ax.hist(y2, bins=bins, alpha=.5, weights=np.ones(len(y2)) / len(y2)) # plot data
Y = np.convolve(window, hv[0], mode='same') # compute rolling average
ax.plot(hv[1][:-1]+np.diff(hv[1])/2, Y, 'C1', linewidth=.5) # plot rolling average
# Clean up plot
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_bounds(0, .15)
ax.spines['bottom'].set_bounds(
    out_r[:, 4, :].flatten().min(), out_r[:, 4, :].flatten().max())
ax.set_xticks([0, 2, 4, 6])
ax.set_xlim([ax.spines['bottom'].get_bounds()[
            0]-np.diff(ax.spines['bottom'].get_bounds())*.1, ax.spines['bottom'].get_bounds()[1]])
ax.set_ylim([ax.spines['left'].get_bounds()[
            0]-np.diff(ax.spines['left'].get_bounds())*.1, ax.spines['left'].get_bounds()[1]])
ax.xaxis.set_tick_params(width=.5)
ax.yaxis.set_tick_params(width=.5)

# Plot data (vesibular signals)
ax = plt.subplot(449)
# Define placeholders
y1err = np.empty([2,in_v.shape[1]])
y2err = np.empty([2,in_v.shape[1]])
y1m = np.empty(in_v.shape[1])
y2m = np.empty(in_v.shape[1])
# Compute means and bootstrapped confidence intervals
for i in range(in_v.shape[1]):
    y1 = in_v[:, i, 0, :].flatten()[out_b[:, 0, :].flatten() < .5]
    y1m[i] = y1.mean()
    y1err[:,i] = np.percentile(y1[np.random.randint(0, len(y1)-1, [len(y1), n_boot])].mean(axis=0), [5, 95])
    y2 = in_v[:, i, 0, :].flatten()[out_b[:, 0, :].flatten() > .5]
    y2m[i] = y2.mean()
    y2err[:,i] = np.percentile(y2[np.random.randint(0, len(y2)-1, [len(y2), n_boot])].mean(axis=0), [5, 95])
ax.plot(x_vest, y1m) # plot data
ax.fill_between(x_vest, y1err[0,:], y1err[1,:], 'C0', alpha=.3) # plot CIs
ax.plot(x_vest, y2m)# plot data
ax.fill_between(x_vest, y2err[0,:], y2err[1,:], 'C1', alpha=.3) # plot CIs
# Clean up plot
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_bounds(ax.get_ylim())
ax.spines['bottom'].set_bounds(x_vest.min(), x_vest.max())
ax.set_xlim([ax.spines['bottom'].get_bounds()[
            0]-np.diff(ax.spines['bottom'].get_bounds())*.1, ax.spines['bottom'].get_bounds()[1]])
ax.set_ylim([ax.spines['left'].get_bounds()[
            0]-np.diff(ax.spines['left'].get_bounds())*.1, ax.spines['left'].get_bounds()[1]])
ax.xaxis.set_tick_params(width=.5)
ax.yaxis.set_tick_params(width=.5)

plt.show()

print('done.')
