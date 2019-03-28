#!/usr/bin/env python3
# -*- coding: utf-8 -*-                                                                            
# DEVELOPER: Aimee Ukasick
# DATE CREATED: 23 May 2018                                  
# PURPOSE: train a model
#
##

# Import statements
import argparse
from time import time
import datetime
import torchvision.models as models
from torch.autograd import Variable

from collections import OrderedDict
import torch
from torch import nn
from torch import cuda
from torch import optim

from torchvision import datasets, transforms, models

import helper


# Main program function defined below
def main():
    # read in command line args
    print("Starting...")
    # Measures total program runtime by collecting start time
    start_time = time()
    print("Parse Command Line Arguments")
    in_args = parse_input_args()
    use_gpu = (in_args.gpu == 'True')
    is_cuda = torch.cuda.is_available()

    # load model from arch specified on command line (or default)
    print(" Loading Pre-Trained Model")
    arch = in_args.arch
    hidden_units = in_args.hidden_units
    print(" arch {} and hidden_units {}".format(arch, hidden_units))
    model = load_model_and_classifier(arch, hidden_units)

    if use_gpu and is_cuda:
        # Move model parameters to the GPU
        print("Using GPU")
        model.cuda()
    else:
        print("Using CPU")
        model.cpu()

    # define common variables used in train and validate functions
    criterion = nn.NLLLoss()
    learning_rate = in_args.learning_rate
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    num_epochs = in_args.epochs
    steps = 0
    running_loss = 0
    print_every = 40

    print("Create Transforms")
    train_test_valid_transforms = create_transforms()
    print("Create Image Data Sets")
    train_test_valid_datasets = create_image_datasets(
        train_test_valid_transforms, in_args)
    print("Create Data Loaders")
    data_loaders = create_data_loaders(train_test_valid_datasets)

    # train model
    print("Train Model")
    train_model(model, data_loaders, criterion, optimizer,
                num_epochs, steps, running_loss, print_every, use_gpu, is_cuda)

    # validate model
    print("Validate Model")
    validate_model(model, data_loaders, criterion, use_gpu, is_cuda)

    # save checkpoint
    print("Save Checkpoint")
    # save mapping of classes to indices obtained from image datasets
    class_to_idx = train_test_valid_datasets['train_data'].class_to_idx
    save_checkpoint(model, num_epochs, optimizer, in_args, class_to_idx)

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
    parser.add_argument('data_directory', type=str, 
                        help='path to top level image folder that contains subdirectories test, train, valid')
    parser.add_argument('--save_dir', type=str, 
                        help='directory to save checkpoint if other than current directory')
    # The training script allows users to choose from at least two different
    # architectures available from torchvision.models
    parser.add_argument('--arch', type=str, default='densenet',
                        help='torchvision.models model; choices: densenet, '
                             'alexnet; default: densenet')
    parser.add_argument('--learning_rate', type=float, default='0.001',                        
                        help='learning rate for the optimizer; default 0.001')
    parser.add_argument('--epochs', type=int, default='20',                        
                        help='number of epochs to use to train model; default: 20')
    # The `hidden_units` option is supposed to set the number of hidden units
    # in the **classifier**.
    parser.add_argument('--hidden_units', type=int, default='512',
                        help='hidden units; default 512 to match default '
                             'arch of densenet')
    parser.add_argument('--gpu', type=str, default='True',
                        help='use GPU for predicting image class; choices: '
                             'True or False; default: True')

    return parser.parse_args()


def load_model_and_classifier(arch, hidden_units):
    # parameters must be frozen *before* the classifier is added

    if arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        freeze_model_parms(model)
        model.classifier = create_classifier_for_alexnet(hidden_units)
    else:
        model = models.densenet121(pretrained=True)
        freeze_model_parms(model)
        model.classifier = create_classifier_for_densenet(hidden_units)

    return model


def freeze_model_parms(model):
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False


# To make things simpler just use a two layer architecture
#  with hidden_units as the intermediate value
def create_classifier_for_densenet(hidden_units):
    # (classifier): Linear(in_features=1024, out_features=1000, bias=True)
    if hidden_units >= 1024:
        raise ValueError(create_error_message("Densenet", 1024, hidden_units))

    if hidden_units <= 102:
        raise ValueError(create_error_message("Densenet", 1024, hidden_units))

    return helper.create_densenet_classifier(hidden_units)


def create_error_message(arch, input_num, hidden_units):
    msg = "{}: --hidden_units cannot be greater than or equal to model input of {}" \
          ". Also, --hidden_units must be greater than output of 102. You entered {}".format(arch, input_num, hidden_units)
    return msg


def create_classifier_for_alexnet(hidden_units):
    # (fc): Linear(in_features=512, out_features=1000, bias=True)
    if hidden_units >= 9216:
        raise ValueError(create_error_message("Alexnet", 9216, hidden_units))

    if hidden_units <= 102:
        raise ValueError(create_error_message("Alexnet", 9216, hidden_units))

    return helper.create_alexnet_classifier(hidden_units)


def train_model(model, data_loaders, criterion, optimizer,
                num_epochs, steps, running_loss, print_every, use_gpu,
                is_cuda):
    # Train the classifier layers using backpropagation using the
    # pre-trained network to get the features

    for epoch in range(num_epochs):
        # model in training mode, dropout is on
        model.train()

        for inputs, labels in iter(data_loaders['trainloader']):
            steps += 1

            optimizer.zero_grad()

            inputs, labels = Variable(inputs), Variable(labels)
            # Move input and label tensors to the GPU
            if use_gpu and is_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]

            if steps % print_every == 0:
                # model in inference mode, dropout is off
                model.eval()

                accuracy = 0
                test_loss = 0

                for ii, (inputs, labels) in enumerate(data_loaders[
                                                          'validloader']):
                    # move input and label tensors to the GPU
                    if use_gpu and is_cuda:
                        inputs = inputs.cuda()
                        labels = labels.cuda()

                    inputs = Variable(inputs)
                    labels = Variable(labels)

                    output = model.forward(inputs)
                    test_loss += criterion(output, labels).data[0]

                    # calculate the accuracy
                    # model's output is LogSoftmax, take exponential to get
                    # the probabilities
                    probabilities = torch.exp(output).data
                    # class with highest probability is our predicted class,
                    #  compare with true label
                    equality = (labels.data == probabilities.max(1)[1])
                    # accuracy is the number of correct predictions divided
                    # by all the predictions, so just take the mean
                    accuracy += equality.type_as(torch.FloatTensor()).mean()

                print("Epoch: {}/{}.. ".format(epoch + 1, num_epochs),
                      "Training Loss: {:.3f}.. ".format(
                          running_loss / print_every),
                      "Valid Loss: {:.3f}.. ".format(
                          test_loss / len(data_loaders['validloader'])),
                      "Valid Accuracy: {:.3f}".format(
                          accuracy / len(data_loaders['validloader'])))

                running_loss = 0

                # make sure dropout is on for training
                model.train()


def validate_model(model, data_loaders, criterion, use_gpu, is_cuda):
    model.eval()
    accuracy = 0
    test_loss = 0
    for ii, (inputs, labels) in enumerate(data_loaders['testloader']):
        # move input and label tensors to the GPU
        if use_gpu and is_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()

        inputs = Variable(inputs)
        labels = Variable(labels)

        output = model.forward(inputs)
        test_loss += criterion(output, labels).data[0]

        # calculate the accuracy
        # model’s output is LogSoftmax, take exponential to get the
        # probabilities
        probabilities = torch.exp(output).data
        # class with highest probability is our predicted class, compare
        # with true label
        equality = (labels.data == probabilities.max(1)[1])
        # accuracy is the number of correct predictions divided by all the
        # predictions, so just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

        print("Test Loss: {:.3f}.. ".format(
            test_loss / len(data_loaders['testloader'])),
              "Test Accuracy: {:.3f}".format(
                  accuracy / len(data_loaders['testloader'])))


def create_transforms():
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(
                                               [0.485, 0.456, 0.406],
                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                              [0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(
                                               [0.485, 0.456, 0.406],
                                               [0.229, 0.224, 0.225])])

    train_test_valid_transforms = {"train_transforms": train_transforms,
                                   "test_transforms": test_transforms,
                                   "valid_transforms": valid_transforms}

    return train_test_valid_transforms


def create_image_datasets(train_test_valid_transforms, in_args):
    data_dir = in_args.data_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir,
                                      transform=train_test_valid_transforms[
                                          'train_transforms'])
    test_data = datasets.ImageFolder(test_dir,
                                     transform=train_test_valid_transforms[
                                         'test_transforms'])
    valid_data = datasets.ImageFolder(valid_dir,
                                      transform=train_test_valid_transforms[
                                          'valid_transforms'])

    train_test_valid_datasets = {"train_data": train_data,
                                 "test_data": test_data,
                                 "valid_data": valid_data}

    return train_test_valid_datasets


def create_data_loaders(train_test_valid_datasets):
    # Using the image datasets and the transforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_test_valid_datasets[
                                                  'train_data'], batch_size=64,
                                              shuffle=True)
    testloader = torch.utils.data.DataLoader(train_test_valid_datasets[
                                                 'test_data'], batch_size=32)
    validloader = torch.utils.data.DataLoader(train_test_valid_datasets[
                                                  'valid_data'], batch_size=32)

    data_loaders = {"trainloader": trainloader,
                    "testloader": testloader,
                    "validloader": validloader}
    return data_loaders


def save_checkpoint(model, epochs, optimizer, in_args, class_to_idx):
    now = datetime.datetime.now()
    time_str = now.strftime('%Y-%m-%dT%H:%M:%S') + ('-%02d' % (
            now.microsecond / 10000))
    filename = in_args.arch + "-checkpoint." + time_str + ".pth.tar"
    if (in_args.save_dir is not None) and (len(in_args.save_dir) > 0):
        filename = filename + "/" + filename

    checkpoint = {
        'state_dict': model.state_dict(),
        'epochs': epochs + 1,
        'optimizer': optimizer.state_dict,
        'class_to_idx': class_to_idx,
        'arch': in_args.arch,
        'hidden_units': in_args.hidden_units
    }

    torch.save(checkpoint, filename)


# Call to main function to run the program
if __name__ == "__main__":
    main()
