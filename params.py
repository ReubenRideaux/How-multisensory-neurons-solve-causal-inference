"""Parameters used to define the network architechture, training protocol, and
image sequence propoerties."""

def nnParams():
    """Load architecture parameters of the neural network."""
    modelParams = {'label': 'MultiNet',
                   'rfdims': 12,
                   'nV1': 64,
                   'nMT': 64,
                   'nVST': 12,
                   'nMST': 128,
                   'tfConvPadding': 'VALID'}
    modelParams['saveDir'] = 'results/' + modelParams['label']
    return modelParams

def tParams():
    """Load training parameters for the neural network."""
    trainParams = {'epochs': 50,
                   'batch_size': 128,
                   'drop_rate': 0.5,
                   'resultsDir': 'results/'}
    return trainParams

def sParams():
    """Load stimulus parameters."""
    stimParams = {'imgHeight': 64,
                  'nTimePoints': 6,
                  'nImages': 64000,
                  'img_dir': 'dataset/training_image_sequences.'}
    return stimParams
