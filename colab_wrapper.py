from types import SimpleNamespace
import json
import os
import glob

import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from apply import apply
from cropping import crop
import image2npy
from main import main
import config
from models import model_mappings

def define_config():
    with open('configs.json') as fp:
        config.config = json.load(fp)

def save_config(config_obj):
    with open('configs.json', 'w') as fp:
        json.dump(config_obj, fp, indent=2)

def run_main(dataset, mode, version, save=False, gpu=False, test_folder="test", overlay=False):
    torch.cuda.empty_cache()
    args = SimpleNamespace(dataset=dataset, mode=mode, version=version, save=save, gpu=gpu, test_folder=test_folder, overlay=overlay)
    define_config()
    main(args)

def run_apply(dataset, version, test_set, gpu=False):
    torch.cuda.empty_cache()
    args = SimpleNamespace(dataset=dataset, version=version, test_set=test_set, gpu=gpu)
    define_config()
    apply(args)

def run_crop(dataset, pixel_num, train_num, val_num, rename=False):
    args = SimpleNamespace(dataset=dataset, pixel_num=pixel_num, train_num=train_num, val_num=val_num, rename=rename)
    crop(args)

def run_image2npy(dataset, subdirectory):
    args = SimpleNamespace(dataset=dataset, subdirectory=subdirectory)
    image2npy.convert(args)

def warn_batch_size(size):
    if size > 1:
        print('WARNING: Batch sizes of greater than 1 are likely to cause out of memory errors on Colab!')
        ans = input('Would you like to continue? ')
        if not (ans.lower() == 'yes' or ans.lower() == 'ye' or ans.lower() == 'y'):
            return False
        return True
    return True

def add_dataset(dataset, n_class, size, batch_size=1, lr=1e-4, epoch=40):
    define_config()
    config_dict = config.config
    config_dict[dataset] = {'default': config.gen_default(dataset, n_class, size, batch_size, lr, epoch)}
    if not warn_batch_size(batch_size):
        return False
    save_config(config_dict)
    print("Added new dataset " + dataset)
    return True

valid_config_settings = ['model', 'n_class', 'root', 'size', 'batch_size', 'shuffle', 'balance', 'optimizer', 'lr', 'patience', 'epoch', 'aug']

def add_version(dataset, version_num=None, **kwargs):
    define_config()
    config_dict = config.config
    is_model = False
    for key in kwargs.keys():
        if key == 'model':
            if kwargs['model'] not in model_mappings:
                print("Invalid model type: " + kwargs['model'])
                return False
            is_model = True
        if key == 'batch_size':
            if not warn_batch_size(kwargs['batch_size']):
                return False
        if key not in valid_config_settings:
            print("Invalid configuration setting: " + key)
            return False
    if not is_model:
        print("Must include a model type in config settings!")
        return False
    try:
        dataset_dict = config_dict[dataset]
    except KeyError:
        print('dataset %s does not exist' % dataset)
        return False
    if not version_num:
        next_num = 1
        while ('v' + str(next_num)) in dataset_dict.keys():
            next_num += 1
        version_num = next_num
    elif ('v' + str(version_num)) in dataset_dict.keys():
        response = input("WARNING: This will overwrite " + dataset + " v" + str(version_num) + ", continue? ")
        if not (response.lower() == 'yes' or response.lower() == 'ye' or response.lower() == 'y'):
            return False
    dataset_dict['v' + str(version_num)] = kwargs
    save_config(config_dict)
    print("Saved version v" + str(version_num) + " of dataset " + dataset)
    return True


def remove_version(dataset, version_num=None):
    define_config()
    config_dict = config.config
    try:
        dataset_dict = config_dict[dataset]
    except KeyError:
        print('dataset %s does not exist' % dataset)
        return False
    if version_num:
        if ('v' + str(version_num)) not in dataset_dict.keys():
            print ('version v' + str(version_num) + ' does not exist')
            return False
    else:
        versions = []
        for key in dataset_dict.keys():
            if key[0] == 'v':
                try:
                    versions.append(int(key[1:]))
                except ValueError:
                    pass
        versions.sort()
        version_num = versions[-1]
    response = input("Continue deleting " + dataset + " v" + str(version_num) + "? ")
    if not (response.lower() == 'yes' or response.lower() == 'ye' or response.lower() == 'y'):
        return False
    dataset_dict.pop('v' + str(version_num))
    save_config(config_dict)
    print("Deleted version v" + str(version_num) + " of dataset " + dataset)
    return True

def display_images(folder):
    if not os.path.isdir(folder):
        print("Image directory not found!")
        return False
    for image_file in glob.glob(folder + "/*.png"):
        img = mpimg.imread(image_file)
        plt.figure()
        plt.title(os.path.basename(image_file))
        plt.imshow(img)
    return True

def display_overlays(dataset, version, test_folder='test'):
    define_config()
    config_dict = config.config
    try:
        dataset_dict = config_dict[dataset]
    except KeyError:
        print('dataset %s does not exist' % dataset)
        return False
    try:
        model = dataset_dict[version]
    except KeyError:
        print('version %s does not exist' % version)
        return False
    directory_path = dataset_dict['default']['root'] + '/' + test_folder + '/overlays/' + version + '_' + model['model']
    if not os.path.isdir(directory_path) or len(glob.glob(directory_path + "/*.png")) == 0:
        print("Overlays not found! Perhaps they haven't made yet?")
        return False
    return display_images(directory_path)

def display_predictions(dataset, version, test_set):
    define_config()
    config_dict = config.config
    try:
        dataset_dict = config_dict[dataset]
    except KeyError:
        print('dataset %s does not exist' % dataset)
        return False
    try:
        model = dataset_dict[version]
    except KeyError:
        print('version %s does not exist' % version)
        return False
    directory_path = 'data/' + test_set + '_predictions/' + dataset + '_' + version + '_' + model['model']
    if not os.path.isdir(directory_path) or len(glob.glob(directory_path + "/*.png")) == 0:
        print("Predictions not found! Perhaps they haven't made yet?")
        return False
    return display_images(directory_path)

def display_plot(dataset, version):
    define_config()
    config_dict = config.config
    try:
        dataset_dict = config_dict[dataset]
    except KeyError:
        print('dataset %s does not exist' % dataset)
        return False
    try:
        model = dataset_dict[version]
    except KeyError:
        print('version %s does not exist' % version)
        return False
    path = 'plots/' + dataset + '_' + version + '_' + model['model'] + '.png'
    if not os.path.isfile(path):
        print("Plot not found! Perhaps it hasn't made yet?")
        return False
    img = mpimg.imread(path)
    plt.figure()
    plt.title(os.path.basename(path))
    plt.imshow(img)
    return True
