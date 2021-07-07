''' Script to MultiNet's estimates around the point of cue separation, as
implemented in Figure 2 Rideaux, Storrs, Maiello and Welchman, Proceedings of
the National Academy of Sciences, 2021

** There must be 5 valid networks saved in the 'results' folder prior to running
this analysis. **

[DEPENDENCIES]
+ tensorflow==1.12.0
+ numpy
+ pickle

[EXAMPLE]
To run the script, please launch your terminal, move to the <MultiNet> folder
and run the following command:
>> python decision-bias.py

'''
#  Helper libraries
import tensorflow as tf
import sys
import numpy as np
import pickle

# In-house libraries
from dot_stim_gen import init_dots, move_dots, draw_dots
import params
import multiNet


def gauss(x, x0, xsigma):
    return np.exp(-.5*((x-x0)/xsigma)**2)


# Define parameters
nnParams = params.nnParams()
sParams = params.sParams()
n_networks = 5

# Define experimental parameters
n_images = 2560  # number of signals per condition
scale = 4  # multisampling factor
dot_size = 12  # size of the dots
imSize = int(sParams['imgHeight']*scale)  # image size
dot_size *= scale
sigma = dot_size/4  # dot sigma
n_dots = 20  # number of dots per image sequence
xsigma = 2  # vestibular signal sigma
x_vest = np.linspace(-8, 8, 32)  # vestibular x dim

# Define dot matrix
dot_mat = np.empty([dot_size, dot_size])
rbegin = -np.round(dot_size/2)
cbegin = -np.round(dot_size/2)
for r in range(dot_size):
    for c in range(dot_size):
        dot_mat[r, c] = np.exp(-(((rbegin+r)/sigma)
                                 ** 2 + ((cbegin+c)/sigma)**2)/2)
# Define image dimensions
border = np.round(dot_size+imSize*(np.sqrt(2)-1)).astype(int)
imRad = np.array(.5*imSize+border).astype(int)
srcRect = np.array([0, 0, dot_size, dot_size])
crx = np.round(0.5*(srcRect[0]+srcRect[2]))
cry = np.round(0.5*(srcRect[1]+srcRect[3]))

# Define motion vectors
speed = 2.3
dx = speed*scale
dz = 0.0
dy = 0.0
dr = 0.0

# Define input placeholders
input_vis = np.empty([n_images, imSize//scale, imSize
                      // scale, sParams['nTimePoints']])

# Define network
network = multiNet.nn(drop_rate=0.0)

for n_idx in range(n_networks):
    # Load trained network
    network.load_weights(nnParams['saveDir'] + '[' + str(n_idx) + ']' + '.h5')

    count = 0
    for im_idx in range(n_images):
        # Provide progress report
        sys.stdout.write(
            '\rTesting network instatiation #%d, measuring network estimates... %d%% ' % (n_idx, count/n_images*100))
        sys.stdout.write('\033[K')
        count += 1
        # Generate visual (dot motion) signals
        for t in range(sParams['nTimePoints']):
            if t == 0:
                py, px, dot_pol = init_dots(dot_size, imRad, n_dots)
            else:
                py, px = move_dots(dot_size, imRad, border,
                                   py, px, dx, dy, dz, dr)
            input_vis[im_idx, :, :, t] = draw_dots(
                py, px, cry, crx, imRad, n_dots, dot_mat, imSize, border, scale, dot_pol)
    # Generate vestibular signals
    vest_label = np.tile(np.array([[0, 0, 0, 0]]), [n_images, 1])
    input_vest = gauss(x_vest.reshape(-1, 1), vest_label.flatten(), xsigma)
    input_vest = input_vest.reshape(
        len(x_vest), vest_label.shape[0], vest_label.shape[1])
    input_vest = np.swapaxes(input_vest, 0, 1)
    input_vest += np.random.normal(0, .8, input_vest.shape)
    # Record estimates
    out_r, out_b = network.predict([input_vis, input_vest])

    # Save results
    data = {'input_vest': input_vest,
            'out_r': out_r,
            'out_b': out_b,
            'speed': speed,
            'dx': dx/scale,
            'dy': dy/scale,
            'dz': dz/scale,
            'dr': dr,
            'dot_size': dot_size/scale,
            'n_dots': n_dots,
            'scale': scale,
            'xsigma': xsigma}
    outfile = open('results/decision-bias' + '[' + str(n_idx) + ']', 'wb')
    pickle.dump(data, outfile)
    outfile.close()

print('done.')
