''' Script to measure the tuning of all MultiNet's units in response to
different visual and vestibular inputs, as implemented in Figure 4 Rideaux,
Storrs, Maiello and Welchman, Proceedings of the National Academy of Sciences,
2021

** There must be 5 valid networks saved in the 'results' folder prior to running
this analysis. **

[DEPENDENCIES]
+ tensorflow==1.12.0
+ numpy
+ pickle

[EXAMPLE]
To run the script, please launch your terminal, move to the <MultiNet> folder
and run the following command:
>> python tuning.py

'''

#  Helper libraries
import tensorflow as tf
import sys
import numpy as np
import pickle

# In-house libraries
from dot_stim_gen import init_dots, move_dots, draw_dots
import multiNet
import params


def gauss(x, x0, xsigma):
    return np.exp(-.5*((x-x0)/xsigma)**2)


# Define parameters
nnParams = params.nnParams()
sParams = params.sParams()
n_networks = 5

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

    # Define experimental parameters
    n_images = 64  # number of signals per condition
    scale = 4  # multisampling factor
    ns = 7  # number of speeds to test
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
    speed = np.linspace(-4, 4, ns)
    dx = speed*scale
    dy = speed*scale
    dz = speed*scale
    dr = speed/4*.125

    # Define input/output placeholders
    input_vis = np.zeros(
        [n_images, imSize//scale, imSize//scale, sParams['nTimePoints']])
    V1_resp = np.zeros([nnParams['nV1'], ns, ns, 2])
    V1_pref = np.zeros([nnParams['nV1'], 4])
    MT_resp = np.zeros([nnParams['nMT'], ns, ns, 2])
    MT_pref = np.zeros([nnParams['nMT'], 4])
    VST_resp = np.zeros([nnParams['nVST'], ns, ns, 2])
    VST_pref = np.zeros([nnParams['nVST'], 4])
    MST_resp_vis = np.zeros([nnParams['nMST'], ns, ns, 2])
    MST_pref_vis = np.zeros([nnParams['nMST'], 4])
    MST_resp_vest = np.zeros([nnParams['nMST'], ns, ns, 2])
    MST_pref_vest = np.zeros([nnParams['nMST'], 4])
    pred = np.zeros([n_images, 2, ns, ns])

    count = 0
    for dim_idx in range(2):
        for xz_idx in range(ns):
            for yr_idx in range(ns):
                # Provide progress report
                sys.stdout.write(
                    '\rTesting network instatiation #%d, establishing unit tuning... %d%% ' % (n_idx, count/(2*ns**2)*100))
                sys.stdout.write('\033[K')
                count += 1
                # Generate visual (dot motion) signals
                for im_idx in range(n_images):
                    for t in range(sParams['nTimePoints']):
                        if t == 0:
                            py, px, dot_pol = init_dots(
                                dot_size, imRad, n_dots)
                        else:
                            py, px = move_dots(dot_size, imRad, border, py, px, dx[xz_idx]*(
                                dim_idx == 0), dy[yr_idx]*(dim_idx == 0), dz[xz_idx]*dim_idx, dr[yr_idx]*dim_idx)
                        input_vis[im_idx, :, :, t] = draw_dots(
                            py, px, cry, crx, imRad, n_dots, dot_mat, imSize, border, scale, dot_pol)

                # Generate vestibular signals
                vest_label = np.array([[dx[xz_idx]*(dim_idx == 0)/scale,
                                        dy[yr_idx]*(dim_idx == 0)/scale,
                                        dz[xz_idx]*dim_idx/scale,
                                        dr[yr_idx]*dim_idx*(4/.125)]])
                input_vest = gauss(x_vest.reshape(-1, 1),
                                   vest_label.flatten(), xsigma)
                input_vest = input_vest.reshape(
                    len(x_vest), vest_label.shape[0], vest_label.shape[1])
                input_vest = np.swapaxes(input_vest, 0, 1)
                input_vest = np.tile(input_vest, [input_vis.shape[0], 1, 1])

                # Save visual activations
                activations = activation_model.predict(
                    [input_vis, input_vest*0])
                V1_resp[:, xz_idx, yr_idx, dim_idx] = np.sum(
                    np.sum(np.mean(activations[0], axis=0), axis=0), axis=0)
                MT_resp[:, xz_idx, yr_idx, dim_idx] = np.mean(
                    activations[4], axis=0)
                MST_resp_vis[:, xz_idx, yr_idx, dim_idx] = np.mean(
                    activations[9], axis=0)

                # Save vestibular activations
                activations = activation_model.predict(
                    [input_vis*0, input_vest])
                VST_resp[:, xz_idx, yr_idx,
                         dim_idx] = activations[5].mean(axis=0)
                MST_resp_vest[:, xz_idx, yr_idx,
                              dim_idx] = activations[9].mean(axis=0)

    # Store preferred velocity
    for dim_idx in range(2):
        for u_idx in range(nnParams['nV1']):
            V1_pref[u_idx, dim_idx*2:dim_idx*2+2] = np.unravel_index(np.argmax(
                V1_resp[u_idx, :, :, dim_idx], axis=None), np.shape(V1_resp[u_idx, :, :, dim_idx]))
        for u_idx in range(nnParams['nMT']):
           MT_pref[u_idx, dim_idx*2:dim_idx*2+2] = np.unravel_index(np.argmax(
               MT_resp[u_idx, :, :, dim_idx], axis=None), np.shape(MT_resp[u_idx, :, :, dim_idx]))
        for u_idx in range(nnParams['nVST']):
           VST_pref[u_idx, dim_idx*2:dim_idx*2+2] = np.unravel_index(np.argmax(
               VST_resp[u_idx, :, :, dim_idx], axis=None), np.shape(VST_resp[u_idx, :, :, dim_idx]))
        for u_idx in range(nnParams['nMST']):
           MST_pref_vis[u_idx, dim_idx*2:dim_idx*2+2] = np.unravel_index(np.argmax(
               MST_resp_vis[u_idx, :, :, dim_idx], axis=None), np.shape(MST_resp_vis[u_idx, :, :, dim_idx]))
        for u_idx in range(nnParams['nMST']):
           MST_pref_vest[u_idx, dim_idx*2:dim_idx*2+2] = np.unravel_index(np.argmax(
               MST_resp_vest[u_idx, :, :, dim_idx], axis=None), np.shape(MST_resp_vest[u_idx, :, :, dim_idx]))
    V1_pref = V1_pref.astype(int)
    MT_pref = MT_pref.astype(int)
    VST_pref = VST_pref.astype(int)
    MST_pref_vis = MST_pref_vis.astype(int)
    MST_pref_vest = MST_pref_vest.astype(int)

    data = {'V1_pref': V1_pref,
            'V1_resp': V1_resp,
            'MT_pref': MT_pref,
            'MT_resp': MT_resp,
            'VST_pref': VST_pref,
            'VST_resp': VST_resp,
            'MST_pref_vis': MST_pref_vis,
            'MST_resp_vis': MST_resp_vis,
            'MST_pref_vest': MST_pref_vest,
            'MST_resp_vest': MST_resp_vest,
            'dx': dx/scale,
            'dy': dy/scale,
            'dz': dz/scale,
            'dr': dr,
            'dot_size': dot_size/scale,
            'n_dots': n_dots,
            'scale': scale,
            'speed': speed,
            'reg': network.layers[13].get_weights()[0],
            'bin': network.layers[14].get_weights()[0]}

    outfile = open('results/tuning' + '[' + str(n_idx) + ']', 'wb')
    pickle.dump(data, outfile)
    outfile.close()

print('done.')
