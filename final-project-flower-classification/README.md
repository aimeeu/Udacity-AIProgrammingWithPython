# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

The flowers dataset may be obtained from http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html
This version is not the same as used in this project. The dataset was broken into test, train, and valid folders. Under each folder, there are 102 subfolders with images.

Flowers.tar.gz is stored in aimeeu's private file storage.

Train.py uses a pretrained model, either densenet or alexnet, to train the model and then save a checkpoint.
Predict.py uses the saved checkpoint to predict the image class of a flower.
