from types import SimpleNamespace
import json
import os
import glob

import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

from apply import apply
from cropping import crop
import image2npy
from main import main
import config
from models import model_mappings
from evaluate import evaluate

import plotly.express as px
import plotly.graph_objects as go

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

def run_evaluate(dataset, version, save=False, gpu=False, test_folder="test", overlay=False, neat=False):
    torch.cuda.empty_cache()
    args = SimpleNamespace(dataset=dataset, version=version, save=save, gpu=gpu, test_folder=test_folder, overlay=overlay, neat=neat)
    define_config()
    evaluate(args)

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

valid_config_settings = ['model', 'n_class', 'root', 'size', 'batch_size', 'shuffle', 'balance', 'optimizer', 'lr', 'patience', 'epoch', 'aug', 'transforms']

def add_version(dataset, version=None, **kwargs):
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
        if key == 'transforms' and type(kwargs['transforms']) != list:
            print("Transforms must be a list of transformations")
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
    if not version:
        next_num = 1
        while ('v' + str(next_num)) in dataset_dict.keys():
            next_num += 1
        version = next_num
    elif type(version) == int:
        version = 'v' + str(version)
    if version in dataset_dict.keys():
        response = input("WARNING: This will overwrite " + dataset + " " + str(version) + ", continue? ")
        if not (response.lower() == 'yes' or response.lower() == 'ye' or response.lower() == 'y'):
            return False
    dataset_dict[str(version)] = kwargs
    save_config(config_dict)
    print("Saved version " + str(version) + " of dataset " + dataset)
    return True


def remove_version(dataset, version=None):
    define_config()
    config_dict = config.config
    try:
        dataset_dict = config_dict[dataset]
    except KeyError:
        print('dataset %s does not exist' % dataset)
        return False
    if version:
        if type(version) == int:
            version = 'v' + str(version)
        if (str(version)) not in dataset_dict.keys():
            print ('version ' + str(version) + ' does not exist')
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
        version = versions[-1]
    response = input("Continue deleting " + dataset + " " + str(version) + "? ")
    if not (response.lower() == 'yes' or response.lower() == 'ye' or response.lower() == 'y'):
        return False
    dataset_dict.pop(str(version))
    save_config(config_dict)
    print("Deleted version " + str(version) + " of dataset " + dataset)
    return True

def display_images(folder, disp_all):
    if not os.path.isdir(folder):
        print("Image directory not found!")
        return False
    imgs = glob.glob(folder + "/*.png")
    figs = []
    for image_file in imgs:
        img = mpimg.imread(image_file)
        figs.append(px.imshow(img, title=os.path.basename(image_file)))
    if disp_all:
        for fig in figs:
          fig.show()
    else:
        full = go.Figure()
        for fig in figs:
            trace = fig.data[0]
            trace.visible = False
            full.add_trace(trace)
        
        full.data[0].visible = True
        full.update_layout(title=go.layout.Title(text=os.path.basename(imgs[0])))
        
        steps = []

        for i in range(len(full.data)):
            image_name = os.path.basename(imgs[i])
            step = dict(
                method='update',
                args=[{"visible": [False] * len(full.data)},
                      {"title": image_name}],
                label=image_name
            )
            step["args"][0]["visible"][i] = True
            steps.append(step)
        
        sliders = [dict(
            active=0,
            currentvalue={'visible': False},
            steps=steps
        )]

        full.update_layout(sliders=sliders)

        full.show()

    return True

def display_overlays(dataset, version, test_folder='test', disp_all=False):
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
    return display_images(directory_path, disp_all)

def display_metrics(dataset, version, test_folder='test', decimal_places=4):
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
    path = dataset_dict['default']['root'] + '/' + test_folder + '/predictions/' + version + '_' + model['model'] + '_Metrics.csv'
    if not os.path.isfile(path):
        print("Metrics not found! Perhaps they haven't been made yet?")
        return False
    metrics = pd.read_csv(path)
    metrics = metrics.rename(columns={"Unnamed: 0": "Image"})
    mean = metrics.mean()
    mean['Image'] = 'Average'
    metrics = metrics.append(mean, ignore_index=True)
    metrics = metrics.round(decimal_places)
    return metrics

def display_predictions(dataset, version, test_set, disp_all=False):
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
    return display_images(directory_path, disp_all)

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
    plt.figure(figsize=(24, 24))
    plt.title(os.path.basename(path))
    plt.imshow(img)
    return True

table_relevant = ['model', 'batch_size', 'shuffle', 'balance', 'optimizer', 'lr', 'patience', 'epoch', 'aug', 'transforms']

def show_versions(dataset):
    define_config()
    config_dict = config.config
    try:
        dataset_dict = config_dict[dataset]
    except KeyError:
        print('dataset %s does not exist' % dataset)
        return False
    default_vals = dataset_dict.pop('default')
    for version in dataset_dict.keys():
        for key in table_relevant:
            if key not in dataset_dict[version].keys() and key in default_vals.keys():
                dataset_dict[version][key] = default_vals[key]
    print(pd.DataFrame.from_dict(dataset_dict, orient='index').fillna("").sort_index().to_string())