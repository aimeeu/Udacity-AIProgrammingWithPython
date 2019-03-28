#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# DEVELOPER: Aimee Ukasick
# DATE CREATED: 25 May 2018
# PURPOSE: common functions for train.py and predict.py

##

from collections import OrderedDict
from torch import nn


def create_densenet_classifier(hidden_units):
    classifier = nn.Sequential(OrderedDict([
        ('linear1', nn.Linear(1024, hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.5)),
        ('linear2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    return classifier


def create_alexnet_classifier(hidden_units):
    classifier = nn.Sequential(OrderedDict([
        ('linear1', nn.Linear(9216, hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.5)),
        ('linear2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    return classifier
