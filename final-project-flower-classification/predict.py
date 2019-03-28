#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# DEVELOPER: Aimee Ukasick
# DATE CREATED: 23 May 2018                                  
# PURPOSE: Predict flower name from an image
#
# Use argparse Expected Call with <> indicating expected user input:
#      python check_images.py --dir <directory with images> --arch <model>
#             --dogfile <file that contains dognames>
#   Example call:
#    python predict.py input checkpoint
##

# Import statements
import argparse
import collections
import json
from time import time
import torch
from torch import cuda
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
from PIL import Image
import helper


# Main program function defined below
def main():
    # read in command line args
    print("Starting...")
    # Measures total program runtime by collecting start time
    start_time = time()

    print("Parse Command Line Arguments")
    in_args = parse_input_args()

    # print("in_args.gpu ", in_args.gpu)
    use_gpu = (in_args.gpu == 'True')
    # print("use_gpu", use_gpu)
    is_cuda = torch.cuda.is_available()

    # load model from arch specified on command line (or default)
    print("Loading Pre-Trained Model From Checkpoint ", in_args.checkpoint)
    model = load_model_from_checkpoint(in_args.checkpoint)

    if use_gpu:
        if is_cuda:
            # Move model parameters to the GPU
            print("Using GPU")
            model.cuda()
    else:
        print("Using CPU")
        model.cpu()

    # predict flower class of image
    print("Predicting Class of Flower Image")
    flower_probability, class_id_list = predict(in_args.image_path,
                                                model, in_args.top_k,
                                                is_cuda, use_gpu)

    print("Print Class Name and Probability")
    print_flower_names_and_probs(flower_probability, class_id_list,
                                 in_args.category_names)

    # Measure total program runtime by collecting end time
    end_time = time()

    # Computes overall runtime in seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time / 3600))) + ":" + str(
              int((tot_time % 3600) / 60)) + ":"
          + str(int((tot_time % 3600) % 60)))


def parse_input_args():
    # Create parser
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str,
                        default='checkpoint.pth.tar',
                        help='path and name of checkpoint; ex: '
                             'checkpoint.pth.tar if the file is in the same '
                             'directory as predict.py')
    parser.add_argument('--top_k', type=int, default=3,
                        help='top K most likely classes; default: 3')
    parser.add_argument('--category_names', type=str,
                        default='cat_to_name.json',
                        help='File path to JSON mapping of categories to real '
                             'names; default: cat_to_name.json')
    parser.add_argument('--gpu', type=str, default='True',
                        help='use GPU for predicting image class; choices: '
                             'True or False; default: True')
    parser.add_argument('--image_path', type=str,
                        default='flowers/test/101/image_07949.jpg',
                        help='path to flower image; default: '
                             'flowers/test/101/image_07949.jpg')

    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_filepath):
    # only using for prediction so no need to create/load optimizer
    checkpoint = torch.load(checkpoint_filepath)
    hidden_units = checkpoint['hidden_units']
    arch = checkpoint['arch']
    # from train.py arch choices are alexnet and the default densenet
    if arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        # need to load the classifier we used when we saved the model
        # or loading the state dict will throw an error
        model.classifier = helper.create_alexnet_classifier(hidden_units)
    else:
        model = models.densenet121(pretrained=True)
        model.classifier = helper.create_densenet_classifier(hidden_units)

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    # do we need to do this since we aren't using it for training?
    for param in model.parameters():
        param.requires_grad = False

    return model


def process_image(image_path):
    print("image_path ", image_path)
    # Process a PIL image for use in a PyTorch model
    raw_image = Image.open(image_path)

    # preserve aspect ratios with thumbnail_size max resolution
    thumbnail_size = 256, 256
    raw_image.thumbnail(thumbnail_size, Image.ANTIALIAS)
    width, height = raw_image.size

    # crop  box – The crop rectangle, as a (left, upper, right, lower)-tuple.
    # 224 x 224
    crop_width = 224
    crop_height = 224
    left = (width - crop_width) / 2
    upper = (height - crop_height) / 2
    right = (width + crop_width) / 2
    lower = (height + crop_height) / 2
    raw_image_crop = raw_image.crop((left, upper, right, lower))

    # normalize the color channels
    # dtype = float64 is a double - do not use because torch then coverts
    # the array to a DoubleTensor
    # and we want a FloatTensor
    # https://github.com/pytorch/pytorch/issues/541
    np_raw_image = np.array(raw_image_crop, dtype=np.dtype(np.float32))
    norm_mean = np.array([0.485, 0.456, 0.406])
    norm_std_dev = np.array([0.229, 0.224, 0.225])
    np_raw_image = ((np_raw_image / 255) - norm_mean) / norm_std_dev

    transposed_image = np_raw_image.transpose(2, 0, 1)
    # transpose turns all the float32 values back to float64!
    # torch views float64 as a double and thus creates a DoubleTensor when
    # torch.from_numpy(transposed_image)
    tensor_image = torch.FloatTensor(transposed_image)
    return tensor_image


def predict(image_path, model, topk, is_cuda, use_gpu):
    # process image, move the tensor to cuda, then pass it to the model
    image_tensor = process_image(image_path)
    if use_gpu and is_cuda:
        # print("using cuda")
        image_tensor = image_tensor.cuda()
        model.cuda()

    # print("image_torch size after processing {}".format(image_tensor.size()))
    # unsqueeze to solve Expected 4D tensor as input, got 3D tensor instead
    # error
    # or use image_tensor.unsqueeze_(0) but clearer for me to do if this way:
    image_tensor_unsqueezed = image_tensor.unsqueeze(0)

    # set the model in inference mode
    model.eval()

    # Calculate the class probabilities (softmax) for img
    # returns a Variable. A Variable wraps a Tensor. It supports nearly all
    # the APIs defined by a Tensor.
    # Variable containing [torch.cuda.FloatTensor of size 1x102 (GPU 0)]
    # output_variable_tensor = model.forward(Variable(
    # image_tensor_unsqueezed, volatile=True))
    with torch.no_grad():
        output_variable_tensor = model.forward(
            Variable(image_tensor_unsqueezed))
    # print("type(output_variable_tensor): {}".format(output_variable_tensor))

    # obtain the probabilities
    probabilities = torch.exp(output_variable_tensor)
    #print("type(probabilities): {}".format(type(probabilities)))

    # find the K largest values using topk(k)
    # A tuple of (values, indices) is returned,
    # where the indices are the indices of the elements in the original
    # input tensor.
    # the elements in the tuple are of type Variable
    # values contains a FloatTensor
    # indices contains a LongTensor
    flower_probability, topk_indices = torch.topk(probabilities, topk)
    # print("type(flower_probability): {}".format(type(flower_probability)))
    # print("flower_probability: {}".format(flower_probability))
    # print("type(topk_indices): {}".format(type(topk_indices)))
    # print("topk_indices: {}".format(topk_indices))

    topk_data_long_tensor = topk_indices.data
    # print("type(topk_data_long_tensor): {}".format(type(
    # topk_data_long_tensor)))
    topk_data_list = topk_data_long_tensor.tolist()[0]  # <class 'list'>
    # print("type(topk_data_list): {}".format(type(topk_data_list)))
    # print("topk_data_list items \n\n", topk_data_list)

    # create idx_to_classID dict from dict model.class_to_idx (reverse it)
    # model.class_to_idx key = str, value = int
    idx_to_class_id = dict()
    for key, value in model.class_to_idx.items():
        idx_to_class_id.setdefault(value, list()).append(key)

    # create list of classIDs
    # for each index in topk_data_list, get the correct class id
    # idx_to_classID key = int, value = str
    class_id_list = list()
    for item in topk_data_list:
        class_id = idx_to_class_id.get(item)[0]
        class_id_list.append(class_id)

    # print("class_ids_list \n\n", class_ids_list)

    return flower_probability, class_id_list


def print_flower_names_and_probs(flower_probability, class_id_list,
                                 category_names_file):
    # load cat_to_names json file
    cat_to_name = load_cat_to_names(category_names_file)
    # get class names from class_ids
    # cat_to_name type: <class 'collections.OrderedDict'>; class_id_list is
    # a list of strings
    class_names = list()
    for class_id in class_id_list:
        name = cat_to_name.get(class_id)
        # print("name = " + name)
        class_names.append(name)

    # convert the data from FloatTensor to a plain old python list
    probs = flower_probability.data.tolist()[0]

    for index in range(len(class_names)):
        print("Probability that the image is of class {} is {}.".format(
            class_names[index], probs[index]))


def load_cat_to_names(category_names_file):
    with open(category_names_file, 'r') as f:
        cat_to_name = json.load(f)

    cat_to_name = collections.OrderedDict(sorted(cat_to_name.items()))
    return cat_to_name


# runs the program
if __name__ == "__main__":
    main()
