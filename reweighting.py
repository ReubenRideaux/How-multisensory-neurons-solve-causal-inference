''' Script to test the cue-reliability based reweighting of the MultiNet, as
implemented in Figure 2 Rideaux, Storrs, Maiello and Welchman, Proceedings of
the National Academy of Sciences, 2021

** There must be 5 valid networks saved in the 'results' folder prior to running
this analysis. **

[DEPENDENCIES]
+ tensorflow==1.12.0
+ numpy==1.15.4
+ pickle

[EXAMPLE]
To run the script, please launch your terminal, move to the <MultiNet> folder
and run the following command:
>> python reweighting.py

'''

#  Helper libraries
import sys
import tensorflow as tf
import numpy as np
import pickle

# In-house libraries
from dot_stim_gen import pol2cart, init_dots, move_dots, draw_dots
import multiNet
import params


def gauss(x, x0, xsigma):
    return np.exp(-.5*((x-x0)/xsigma)**2)


# Define parameters
nnParams = params.nnParams()
sParams = params.sParams()
n_networks = 5

# Define experimental parameters
n_images = 128  # number of signals per condition
scale = 4  # multisampling factor
nd = 7  # number of directions to test
dot_size = 6  # size of the dots
imSize = int(sParams['imgHeight']*scale)  # image size
dot_size *= scale
sigma = dot_size/4  # dot sigma
n_dots = 100  # number of dots per image sequence
speed = 4  # dot speed (pixels/frame)
xsigma = 6  # vestibular signal sigma
x_vest = np.linspace(-8, 8, 32)  # vestibular x dim
noise = np.round([.01*n_dots, .6*n_dots]).astype('int')  # visual signal noise

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
direction = np.logspace(np.log10(.01), np.log10(.08), (nd-1)//2, base=10)*np.pi
direction = np.concatenate([np.flip(-direction), direction], axis=0)
direction = np.insert(direction, 3, 0)
[dz, dx] = pol2cart(speed, direction)
[dzc, dxc] = pol2cart(speed, [0., .05*np.pi, -0.05*np.pi])
dzc -= speed

dx *= scale
dy = dz*0
dz *= scale
dr = dz*0

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

    # Define input/output placeholders
    input_vis = np.empty(
        [n_images, imSize//scale, imSize//scale, sParams['nTimePoints']])
    combined = np.empty([n_images, 2, 3, nd, 4])
    vis = np.empty([n_images, 2, nd, 4])
    vest = np.empty([n_images, 2, nd, 4])

    count = 0
    for r_idx in range(2):
        for c_idx in range(3):
            for d_idx in range(nd):
                # Provide progress report
                sys.stdout.write(
                    '\rTesting network instatiation #%d, simulating cue conflict... %d%% ' % (n_idx, count/(nd*2*3)*100))
                sys.stdout.write('\033[K')
                count += 1
                # Generate visual (dot motion) signals
                for im_idx in range(n_images):
                    DX = np.tile(dx[d_idx]-dxc[c_idx]*scale, n_dots)
                    DX[:noise[r_idx]] = np.random.uniform(-4, 4, noise[r_idx])
                    DY = np.tile(dy[d_idx], n_dots)
                    DY[:noise[r_idx]] = np.random.uniform(-4, 4, noise[r_idx])
                    DZ = np.tile(dz[d_idx]-dzc[c_idx]*scale, n_dots)
                    DZ[:noise[r_idx]] = 0
                    for t in range(sParams['nTimePoints']):
                        if t == 0:
                            py, px, dot_pol = init_dots(
                                dot_size, imRad, n_dots)
                        else:
                            py, px = move_dots(
                                dot_size, imRad, border, py, px, DX, DY, DZ, dr[d_idx])
                        input_vis[im_idx, :, :, t] = draw_dots(
                            py, px, cry, crx, imRad, n_dots, dot_mat, imSize, border, scale, dot_pol)

                # Generate vestibular signals
                vest_label = np.tile(np.array([[dx[d_idx]/scale+dxc[c_idx],
                                                dy[d_idx]/scale,
                                                dz[d_idx]/scale+dzc[c_idx],
                                                dr[d_idx]*(4/.125)]
                                               ]), [n_images, 1])
                input_vest = gauss(x_vest.reshape(-1, 1),
                                   vest_label.flatten(), xsigma)
                input_vest = input_vest.reshape(
                    len(x_vest), vest_label.shape[0], vest_label.shape[1])
                input_vest = np.swapaxes(input_vest, 0, 1)
                input_vest += np.random.normal(0, .3, input_vest.shape)

                combined[:, r_idx, c_idx, d_idx, :] = network.predict(
                    [input_vis, input_vest])[0][:, :4]
                if c_idx == 0:
                    vis[:, r_idx, d_idx, :] = network.predict(
                        [input_vis, input_vest*0])[0][:, :4]
                    vest[:, r_idx, d_idx, :] = network.predict(
                        [input_vis*0, input_vest])[0][:, :4]
    # Save results
    data = {'combined': combined,
            'vis': vis,
            'vest': vest,
            'noise': noise,
            'xsigma': xsigma,
            'speed': speed,
            'direction': direction,
            'dx': dx/scale,
            'dy': dy/scale,
            'dz': dz/scale,
            'dr': dr,
            'dxc': dxc,
            'dzc': dzc,
            'dot_size': dot_size/scale,
            'n_dots': n_dots,
            'scale': scale,
            'xsigma': xsigma}
    outfile = open('results/reweighting' + '[' + str(n_idx) + ']', 'wb')
    pickle.dump(data, outfile)
    outfile.close()

print('done.')
