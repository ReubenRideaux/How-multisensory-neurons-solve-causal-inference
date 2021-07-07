''' Script to test the congruency of MSTd units in the network, as implemented
in Figure 3 Rideaux, Storrs, Maiello and Welchman, Proceedings of the National
Academy of Sciences, 2021

** There must be 5 valid networks saved in the 'results' folder prior to running
this analysis. **

[DEPENDENCIES]
+ tensorflow==1.12.0
+ numpy
+ scipy
+ pickle

[EXAMPLE]
To run the script, please launch your terminal, move to the <MultiNet> folder
and run the following command:
>> python congruency.py

'''
#  Helper libraries
import tensorflow as tf
import sys
import numpy as np
import pickle

# In-house libraries
import params
import multiNet
from dot_stim_gen import pol2cart, init_dots, move_dots, draw_dots

# Define 1D gaussian function
def gauss(x, x0, xsigma):
   return np.exp(-.5*((x-x0)/xsigma)**2)

# Load parameters
nnParams = params.nnParams()
sParams = params.sParams()
n_networks = 5

# Define experimental parameters
n_images = 64  # number of signals per condition
scale = 4  # multisampling factor
nd = 10  # number of directions to test
dot_size = 6  # size of the dots
imSize = int(sParams['imgHeight']*scale)  # image size
dot_size *= scale
sigma = dot_size/4  # dot sigma
n_dots = 100  # number of dots per image sequence
speed = 4 # dot speed (pixels/frame)
xsigma = 4 # vestibular signal sigma
x_vest = np.linspace(-8, 8, 32) # vestibular x dim

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
lin_direction = np.linspace(-np.pi, np.pi, nd, endpoint=False)
log_direction = np.logspace(np.log10(.01), np.log10(.2), nd//2, base=10)*np.pi
log_direction = np.concatenate(
    [np.flip(-log_direction), log_direction], axis=0)
direction = np.stack((lin_direction, log_direction))
[dz, dx] = pol2cart(speed, direction)
dx *= scale
dy = dz*0
dz *= scale
dr = dz*0

# Define network
network = multiNet.nn(drop_rate=0.0)

# Iterate over multiple network instantiations
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

    # Define input/output placeholders
    input_vis = np.empty([n_images, imSize//scale, imSize
                          // scale, sParams['nTimePoints']])
    MST_resp_vis = np.empty([n_images, 2, nd, nnParams['nMST']])
    MST_resp_vest = np.empty([n_images, 2, nd, nnParams['nMST']])
    MST_resp_combined = np.empty([n_images, 2, nd, nnParams['nMST']])

    count = 0
    for c_idx in range(2):
        for d_idx in range(nd):
            # Provide progress report
            sys.stdout.write(
                '\rClassifying network instatiation #%d, classifying MST unit congruency... %d%% ' % (n_idx,count/(nd*2)*100))
            sys.stdout.write('\033[K')
            count += 1
            for im_idx in range(n_images):
                # Generate visual (dot motion) signals
                for t in range(sParams['nTimePoints']):
                    if t == 0:
                        py, px, dot_pol = init_dots(dot_size, imRad, n_dots)
                    else:
                        py, px = move_dots(
                            dot_size, imRad, border, py, px, dx[c_idx, d_idx], dy[c_idx, d_idx], dz[c_idx, d_idx], dr[c_idx, d_idx])
                    input_vis[im_idx, :, :, t] = draw_dots(
                        py, px, cry, crx, imRad, n_dots, dot_mat, imSize, border, scale, dot_pol)

            # Generate vestibular signals
            vest_label = np.tile(np.array([[dx[c_idx, d_idx]/scale,
                                            dy[c_idx, d_idx]/scale,
                                            dz[c_idx, d_idx]/scale,
                                            dr[c_idx, d_idx]*(4/.125)]
                                           ]), [n_images, 1])
            input_vest = gauss(x_vest.reshape(-1, 1),
                               vest_label.flatten(), xsigma)
            input_vest = input_vest.reshape(
                len(x_vest), vest_label.shape[0], vest_label.shape[1])
            input_vest = np.swapaxes(input_vest, 0, 1)

            if c_idx == 1:
                input_vest += np.random.normal(0, .2, input_vest.shape)

            # Save network activations
            MST_resp_vis[:, c_idx, d_idx, :] = activation_model.predict(
                [input_vis, input_vest*0])[9]
            MST_resp_vest[:, c_idx, d_idx, :] = activation_model.predict(
                [input_vis*0, input_vest])[9]
            MST_resp_combined[:, c_idx, d_idx, :] = activation_model.predict(
                [input_vis, input_vest])[9]

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
    outfile = open('results/congruency' + '[' + str(n_idx) + ']', 'wb')
    pickle.dump(data, outfile)
    outfile.close()

print('done.')
