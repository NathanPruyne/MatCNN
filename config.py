"""
Determines Network Training Configurations. The training process uses this
module to obtain hyperparameters from versions defined for each dataset.
This version must be specified upon training, evaluating, and applying the
network. This module must be updated each time the user would like to train
on a new dataset, or with new hyperparameters.
(This is most easily accomplished in a text or code editor, and not from
within the command line.)
"""
import sys


# Generates the default configuration for a given dataset with the following
# hyperparameters. These default values will later be updated by the version
# specified in the get_config function below.
def gen_default(dataset, n_class, size, batch_size=4, lr=1e-4,
                epoch=40):
    default = {
        'root': './data/' + dataset,
        'n_class': n_class,
        'size': size,
        'batch_size': batch_size,
        'optimizer': 'Adam',
        'lr': lr,
        'epoch': epoch,
        'aug': False,
        'shuffle': False,
        'patience': epoch,
        'balance': False
    }
    return default


# Master dictionary whose keys correspond to the dataset on which the NN is to be
# trained. Each entry contains a sub-dictionary on versions, each of which defines
# the set of hyperparameters to be used in training. Custom sets of hyperparameters
# can be created by editing versions to change any of the desired entries in the
# default set listed above. Notice there are no longer entries for the mean and
# standard deviation, which is now handled entirely by the average.py script.
config = {
    'uhcs': {
        'default': gen_default('uhcs', n_class=4, size=(484, 645)),
        'v1': {'model': 'pixelnet'},
        'v2': {'model': 'unet'},
        'v3': {'model': 'segnet', 'optimizer': 'SGD', 'lr': 0.01},
        'v4': {'model': 'pixelnet', 'aug': True},
        'v5': {'model': 'unet', 'aug': True},
        'v6': {'model': 'segnet', 'aug': True, 'optimizer': 'SGD', 'lr': 0.01}
    },
    'tomography': {
        'default': gen_default('tomography', n_class=2, size=(852, 852)),
        'v1': {'model': 'pixelnet'},
        'v2': {'model': 'unet', 'epoch':1},
        'v3': {'model': 'segnet', 'optimizer': 'SGD', 'lr': 0.01},
        'v4': {'model': 'pixelnet', 'aug': True, 'balance': True},
        'v5': {'model': 'unet', 'aug': True, 'balance': True},
        'v6': {'model': 'segnet', 'aug': True, 'optimizer': 'SGD', 'lr': 0.01, 'balance': True},
	    'v7': {'model': 'unet', 'epoch': 4, 'lr':0.001, 'aug': True},
	    'v8': {'model': 'segnet', 'epoch': 50, 'aug': True, 'lr': 0.001, 'balance': True} 
    },
    'CatSpine': {
        'default': gen_default('CatSpine', n_class=2, size=(2300, 1920)),
        'v1': {'model': 'segnet', 'batch_size': 1, 'optimizer': 'SGD', 'lr': 0.001, 'epoch': 20, 'aug': True,
               'shuffle': True, 'patience': 3, 'balance': True},
        'v2': {'model': 'segnet', 'batch_size': 1, 'optimizer': 'SGD', 'lr': 0.001, 'epoch': 30, 'aug': True,
               'shuffle': True, 'patience': 3, 'balance': True},
        'v3': {'model': 'segnet', 'batch_size': 1, 'optimizer': 'SGD', 'lr': 0.001, 'epoch': 40, 'aug': True,
               'shuffle': True, 'patience': 3, 'balance': True},
        'v4': {'model': 'segnet', 'batch_size': 1, 'optimizer': 'SGD', 'lr': 0.001, 'epoch': 50, 'aug': True,
               'shuffle': True, 'patience': 3, 'balance': True}
    },
}


# The get_config function takes the default set of hyperparameters given by the gen_default
# function, and updates, or creates new, entries given the version specified.
def get_config(dataset, version):
    try:
        args = config[dataset]['default'].copy()
    except KeyError:
        print('dataset %s does not exist' % dataset)
        sys.exit(1)
    try:
        args.update(config[dataset][version])
    except KeyError:
        print('version %s is not defined' % version)
    args['name'] = dataset + '_' + version
    return args
