"""Build the neural net graph.

Args:
drop_rate: an input scalar between 0-1

Returns:
the network

[DEPENDENCIES]
+ tensorflow==1.12.0

During training drop_rate > 0.0. During testing drop_rate == 0. If
drop_rate == 0 the activations are separated from the layers to allow
interogation of activity before and after non-linear activation is applied."""

import tensorflow as tf
import params

def custom_loss_function(y_true,y_pred):
    loss = y_true-y_pred
    loss = loss * [1.0,1.0,1.0,1.0,0.5,0.5,0.5,0.5]
    loss = tf.keras.backend.square(loss)
    loss = tf.keras.backend.mean(loss,axis=1)
    return loss

def nn(drop_rate):
    sParams = params.sParams()
    nnParams = params.nnParams()
    if drop_rate>0.0:
        activation = 'relu'
    else:
        activation = 'linear'
    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.001)

    input_vis = tf.keras.layers.Input(shape=(sParams['imgHeight'],sParams['imgHeight'],sParams['nTimePoints'],))

    input_vest = tf.keras.layers.Input(shape=(32,4,))

    vis = tf.keras.layers.Conv2D(filters=nnParams['nV1'],
                                       kernel_size=(nnParams['rfdims'],nnParams['rfdims']),
                                       strides=(1,1),
                                       padding=nnParams['tfConvPadding'],
                                       activation=activation,
                                       input_shape=(sParams['imgHeight'],sParams['imgHeight'],sParams['nTimePoints']),
                                       kernel_initializer=kernel_initializer,
                                       name='V1')(input_vis)
    if drop_rate>0.0:
        vis = tf.keras.layers.Dropout(drop_rate)(vis)
    else:
        vis = tf.keras.layers.Activation('relu')(vis)
    vis = tf.keras.layers.Flatten()(vis)
    vis = tf.keras.layers.Dense(nnParams['nMT'],
                                      kernel_initializer=kernel_initializer,
                                      activation=activation,
                                      name='MT')(vis)
    if drop_rate>0.0:
        vis = tf.keras.layers.Dropout(drop_rate)(vis)
    else:
        vis = tf.keras.layers.Activation('relu')(vis)
    vis = tf.keras.models.Model(inputs=input_vis,outputs=vis)

    vest = tf.keras.layers.Flatten()(input_vest)
    vest = tf.keras.layers.Dense(nnParams['nVST'],
                                      kernel_initializer=kernel_initializer,
                                      activation=activation,
                                      name='VST')(vest)
    if drop_rate>0.0:
        vest = tf.keras.layers.Dropout(drop_rate)(vest)
    else:
        vest = tf.keras.layers.Activation('relu')(vest)
    vest = tf.keras.models.Model(inputs=input_vest,outputs=vest)

    combined = tf.keras.layers.concatenate([vis.output,vest.output])
    combined = tf.keras.layers.Dense(nnParams['nMST'],
                                  kernel_initializer=kernel_initializer,
                                  activation=activation,
                                  name='MST')(combined)
    if drop_rate>0.0:
        combined = tf.keras.layers.Dropout(drop_rate)(combined)
    else:
        combined = tf.keras.layers.Activation('relu')(combined)
    combined_reg = tf.keras.layers.Dense(8,
                                         kernel_initializer=kernel_initializer,
                                         name='regression_output')(combined)
    combined_bin = tf.keras.layers.Dense(4,
                                         activation='sigmoid',
                                         name='binary_output')(combined)
    losses = {'regression_output': custom_loss_function,
        	  'binary_output': 'binary_crossentropy'}
    loss_weights = {'regression_output': 1.0, 'binary_output': 0.2}
    network = tf.keras.models.Model(inputs=[vis.input,vest.input],outputs=[combined_reg,combined_bin])

    network.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                    loss=losses,
                    loss_weights=loss_weights,
                    metrics=['accuracy'])

    return network
