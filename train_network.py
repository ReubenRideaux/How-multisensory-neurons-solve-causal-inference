''' Script to train a multisensory convolutional neural network
(i.e., MultiNet) as implemented in Rideaux, Storrs, Maiello and Welchman,
Proceedings of the National Academy of Sciences, 2021

** The training image files (training_image_sequences.1-8)
must be placed into the 'dataset' folder prior to running this script. **

[DEPENDENCIES]
+ tensorflow==1.12.0
+ numpy
+ pickle
+ gzip
+ sklearn

[EXAMPLE]
To run the script, please launch your terminal, move to the <MultiNet> folder
and run the following command:
>> python train_network.py

'''

# Helper libraries
import tensorflow as tf
import numpy as np
import pickle, gzip
from sklearn.model_selection import train_test_split

# In-house libraries
import params
import multiNet

# Define 1D gaussian function
def gauss(x, x0, xsigma):
   return np.exp(-.5*((x-x0)/xsigma)**2)

# Load parameters
tParams = params.tParams()
sParams = params.sParams()
nnParams = params.nnParams()

# Load visual dataset
data = {'images': np.empty([sParams['nImages'],
                            sParams['imgHeight'],
                            sParams['imgHeight'],
                            sParams['nTimePoints']]),
        'labels': np.empty([sParams['nImages'], 4])}
n_files = 8
for i in range(n_files):
    print('Loading file #'+str(i))
    infile = gzip.GzipFile(sParams['img_dir'] + str(i), 'rb')
    obj = infile.read()
    temp_storage = pickle.loads(obj)
    infile.close()
    data['images'][i*(sParams['nImages']//n_files):(i+1)*
        (sParams['nImages']//n_files),:,:,:] = temp_storage['images']
    data['labels'][i*(sParams['nImages']//n_files):(i+1)*
        (sParams['nImages']//n_files),:] = temp_storage['labels']
del temp_storage

# Normalize pixel values to between -1:1
data['images'] = data['images'].astype('float32') / 255. * 2. - 1.

# Shuffle and split visual signals
vis_data_train, vis_data_test, vis_label_train, vis_label_test = train_test_split(
    data['images'], data['labels'], test_size=0.25, random_state=42)

# Vestibular x dim
x = np.linspace(-8, 8, 32)

# Generate vestibular signal sigma
xsigma_train = np.random.uniform(1, 8, np.prod(vis_label_train.shape))
xsigma_test = np.random.uniform(1, 8, np.prod(vis_label_test.shape))

# Generate vestibular signal mean
vest_label_train = np.random.uniform(-4, 4, np.prod(vis_label_train.shape))
vest_label_test = np.random.uniform(-4, 4, np.prod(vis_label_test.shape))
vest_label_train = vest_label_train.flatten()
vest_label_test = vest_label_test.flatten()

# Generate vestibular signals
vest_data_train = gauss(x.reshape(-1, 1), vest_label_train, xsigma_train)
vest_data_test = gauss(x.reshape(-1, 1), vest_label_test, xsigma_test)

# Reshape vestibular signals
vest_data_train = vest_data_train.reshape(
    len(x), vis_label_train.shape[0], vis_label_train.shape[1])
vest_data_test = vest_data_test.reshape(
    len(x), vis_label_test.shape[0], vis_label_test.shape[1])
vest_data_train = np.swapaxes(vest_data_train, 0, 1)
vest_data_test = np.swapaxes(vest_data_test, 0, 1)
vest_label_train = vest_label_train.reshape(
    vis_label_train.shape[0], vis_label_train.shape[1])
vest_label_test = vest_label_test.reshape(
    vis_label_test.shape[0], vis_label_test.shape[1])

## Add noise to vestibular signal
vest_data_train += np.random.normal(0, .3, vest_data_train.shape)
vest_data_test += np.random.normal(0, .3, vest_data_test.shape)

# Define label placeholders []
label_train = np.empty([vis_label_train.shape[0], 12])
label_test = np.empty([vis_label_test.shape[0], 12])

# Average
label_train[:, :4] = (vis_label_train+vest_label_train)/2
label_test[:, :4] = (vis_label_test+vest_label_test)/2

# Difference
label_train[:, 4:8] = vis_label_train-vest_label_train
label_test[:, 4:8] = vis_label_test-vest_label_test

# Binary causal inference
label_train[:, 8:] = np.abs(
    vis_label_train-vest_label_train) < np.median(np.abs(vis_label_train-vest_label_train))
label_test[:, 8:] = np.abs(
    vis_label_test-vest_label_test) < np.median(np.abs(vis_label_test-vest_label_test))

# Define network
network = multiNet.nn(drop_rate=tParams['drop_rate'])

# Train network
network.fit([vis_data_train, vest_data_train],
            [label_train[:, :8], label_train[:, 8:]],
            epochs=tParams['epochs'],
            batch_size=tParams['batch_size'],
            shuffle=True,
            verbose=1,
            validation_data=([vis_data_test, vest_data_test],
                             [label_test[:, :8], label_test[:, 8:]]))

# Save network
network.save(nnParams['saveDir'] + '.h5')

print('done.')
