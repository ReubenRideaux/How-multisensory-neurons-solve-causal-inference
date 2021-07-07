''' Script to measure the activity of  MultiNet's congruent and opposite MSTd
units in response to a range of cue combinations, as implemented in Figure 4a
Rideaux, Storrs, Maiello and Welchman, Proceedings of the National Academy of
Sciences, 2021

** There must be 5 valid networks saved in the 'results' folder prior to running
this analysis. **

[DEPENDENCIES]
+ tensorflow==1.12.0
+ numpy
+ pickle

[EXAMPLE]
To run the script, please launch your terminal, move to the <MultiNet> folder
and run the following command:
>> python causal-inference.py

'''

# Helper libraries
import tensorflow as tf
import sys
import numpy as np
import pickle

# In-house libraries
import params
import multiNet
from dot_stim_gen import pol2cart, init_dots, move_dots, draw_dots


def gauss(x, x0, xsigma):
    return np.exp(-.5*((x-x0)/xsigma)**2)


# Define parameters
nnParams = params.nnParams()
sParams = params.sParams()
n_networks = 5

# Define experimental parameters
n_images = 64  # number of signals per condition
scale = 4  # multisampling factor
nd = 13  # number of speeds to test
dot_size = 6  # size of the dots
imSize = int(sParams['imgHeight']*scale)  # image size
dot_size *= scale
sigma = dot_size/4  # dot sigma
n_dots = 100  # number of dots per image sequence
xsigma = 4  # vestibular signal sigma
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
speed = 4.0
direction = np.linspace(-np.pi/2, np.pi/2, nd)
[dz, dx] = pol2cart(speed, direction)
dx *= scale
dy = dx*0
dz *= scale
dr = dx*0

# Define input/output placeholders
input_vis = np.empty([n_images, imSize//scale, imSize
                      // scale, sParams['nTimePoints']])
MST_resp_vis = np.empty([n_images, nd, nnParams['nMST']])
MST_resp_vest = np.empty([n_images, nd, nnParams['nMST']])
MST_resp_combined = np.empty([n_images, nd, nd, nnParams['nMST']])

# Define network
network = multiNet.nn(drop_rate=0.0)

for n_idx in range(n_networks):
    # Load trained network
    network.load_weights(nnParams['saveDir'] + '[' + str(n_idx) + ']' + '.h5')

    # Extracts the outputs V1, MT, & MST layers
    layer_outputs = [layer.output for layer in network.layers[:]]
    layer_outputs.pop(3)
    layer_outputs.pop(0)

    # Creates a model that will return these outputs, given the model input
    activation_model = tf.keras.models.Model(
        inputs=network.input, outputs=layer_outputs)

    count = 0
    for vis_d_idx in range(nd):
        # Generate visual (dot motion) signals
        for im_idx in range(n_images):
            for t in range(sParams['nTimePoints']):
                if t == 0:
                    py, px, dot_pol = init_dots(dot_size, imRad, n_dots)
                else:
                    py, px = move_dots(dot_size, imRad, border, py, px,
                                       dx[vis_d_idx], dy[vis_d_idx], dz[vis_d_idx], dr[vis_d_idx])
                input_vis[im_idx, :, :, t] = draw_dots(
                    py, px, cry, crx, imRad, n_dots, dot_mat, imSize, border, scale, dot_pol)
        for vest_d_idx in range(nd):
            # Provide progress report
            sys.stdout.write(
                '\rTesting network instatiation #%d, measuring unit actiation... %d%% ' % (n_idx, count/(nd**2)*100))
            sys.stdout.write('\033[K')
            count += 1

            # Generate vestibular signals
            vest_label = np.tile(np.array([[dx[vest_d_idx]/scale,
                                            dy[vest_d_idx]/scale,
                                            dz[vest_d_idx]/scale,
                                            dr[vest_d_idx]*(4/.125)]
                                           ]), [n_images, 1])
            input_vest = gauss(x_vest.reshape(-1, 1),
                               vest_label.flatten(), xsigma)
            input_vest = input_vest.reshape(
                len(x_vest), vest_label.shape[0], vest_label.shape[1])
            input_vest = np.swapaxes(input_vest, 0, 1)

            # Record activations
            if vest_d_idx == 0:
                MST_resp_vis[:, vis_d_idx, :] = activation_model.predict(
                    [input_vis, input_vest*0])[9]  # single-cue visual
            if vis_d_idx == 0:
                MST_resp_vest[:, vest_d_idx, :] = activation_model.predict(
                    [input_vis*0, input_vest])[9]  # single-cue vestibular
            MST_resp_combined[:, vis_d_idx, vest_d_idx, :] = activation_model.predict(
                [input_vis, input_vest])[9]  # combined cue

     # Save results
    data = {'MST_resp_vis': MST_resp_vis,
            'MST_resp_vest': MST_resp_vest,
            'MST_resp_combined': MST_resp_combined,
            'speed': speed,
            'direction': direction,
            'dx': dx/scale,
            'dy': dy/scale,
            'dz': dz/scale,
            'dr': dr,
            'dot_size': dot_size/scale,
            'n_dots': n_dots,
            'scale': scale,
            'xsigma': xsigma}
    outfile = open('results/causal-inference' + '[' + str(n_idx) + ']', 'wb')
    pickle.dump(data, outfile)
    outfile.close()

print('done.')
