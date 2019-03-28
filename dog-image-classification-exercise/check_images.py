#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND/intropylab-classifying-images/check_images.py
#
# DONE: 0. Fill in your information in the programming header below
# PROGRAMMER: Aimee Ukasick
# DATE CREATED: 11 April 2018
# REVISED DATE:             <=(Date Revised - if any)
# PURPOSE: Check images & report results: read them in, predict their
#          content (classifier), compare prediction to actual value labels
#          and output results
#
# Use argparse Expected Call with <> indicating expected user input:
#      python check_images.py --dir <directory with images> --arch <model>
#             --dogfile <file that contains dognames>
#   Example call:
#    python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
##

# Imports python modules
import argparse
# Imports time() and sleep() functions from time module
from time import time, sleep
from os import listdir

# Imports classifier function for using CNN to classify images
from classifier import classifier

# Main program function defined below


def main():
    # DONE: 1. Define start_time to measure total program runtime by
    # collecting start time
    start_time = time()

    # DONE: 2. Define get_input_args() function to create & retrieve command
    # line arguments
    print("***** calling get_input_args *****")
    in_args = get_input_args()
    #print_command_line_args(in_args)

    # DONE: 3. Define get_pet_labels() function to create pet image labels by
    # creating a dictionary with key=filename and value=file label to be used
    # to check the accuracy of the classifier function
    print("***** calling get_pet_labels *****")
    petlabels_dict = get_pet_labels(in_args.dir)
    # print("*****  print_petlabels_dict(petlabels_dict) *****")
    # print_petlabels_dict(petlabels_dict)


    # DONE: 4. Define classify_images() function to create the classifier
    # labels with the classifier function using in_arg.arch, comparing the
    # labels, and creating a dictionary of results (result_dic)
    # print("***** calling classify_images *****")
    # result_dic = classify_images(in_args.dir, petlabels_dict, in_args.arch)
    # print("***** printing my code classify_images result_dic *****")
    # print_result_dic(result_dic)
    print("***** calling classify_images_udacity *****")
    result_dic = classify_images_udacity(in_args.dir, petlabels_dict,
                                         in_args.arch)
    # print("***** printing classify_images_udacity result_dic *****")
    # print_result_dic(result_dic)

    # DONE: 5. Define adjust_results4_isadog() function to adjust the results
    # dictionary(result_dic) to determine if classifier correctly classified
    # images as 'a dog' or 'not a dog'. This demonstrates if the model can
    # correctly classify dog images as dogs (regardless of breed)
    # print("***** calling adjust_results4_isadog *****")
    # adjust_results4_isadog(result_dic, in_args.dogfile)
    # print("***** printing my adjust_results4_isadog *****")
    # print_adjust_results4_isadog(result_dic)
    print("***** calling adjust_results4_isadog_udacity *****")
    adjust_results4_isadog_udacity(result_dic, in_args.dogfile)
    # print("***** printing my adjust_results4_isadog_udacity *****")
    # print_adjust_results4_isadog(result_dic)

    # DONE: 6. Define calculates_results_stats() function to calculate
    # results of run and puts statistics in a results statistics
    # dictionary (results_stats_dic)
    print("***** calculates_results_stats *****")
    results_stats_dic = calculates_results_stats(result_dic)
    print("***** check_results_stats *****")
    check_results_stats(results_stats_dic, result_dic)

    # DONE: 7. Define print_results() function to print summary results,
    # incorrect classifications of dogs and breeds if requested.
    print_results(result_dic, results_stats_dic, in_args.arch,
                  True, True)

    # DONE: 1. Define end_time to measure total program runtime
    # by collecting end time
    end_time = time()

    # DONE: 1. Define tot_time to computes overall runtime in
    # seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time
    hours = int((tot_time / 3600))
    minutes = int(((tot_time % 3600) / 60))
    seconds = int(((tot_time % 3600) % 60))
    print("\n** Total Elapsed Runtime:", str(hours) +
          ":" + str(minutes) + ":" + str(seconds))


# TODO: 2.-to-7. Define all the function below. Notice that the input
# paramaters and return values have been left in the function's docstrings.
# This is to provide guidance for acheiving a solution similar to the
# instructor provided solution. Feel free to ignore this guidance as long as
# you are able to acheive the desired outcomes with this lab.

def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object.
     3 command line arguements are created:
       dir - Path to the pet image files(default- 'pet_images/')
       arch - CNN model architecture to use for image classification(default-
              pick any of the following vgg, alexnet, resnet)
       dogfile - Text file that contains all labels associated to dogs(default-
                'dognames.txt'
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object
    """
    parser = argparse.ArgumentParser()

    # arg 1 - path to folder with default
    parser.add_argument('--dir', type=str, default='pet_images/',
                        help='path to the folder that contains the images; default is pet_images')

    # arg 2 - CNN model architecture to use for image classification
    parser.add_argument('--arch', type=str, default='vgg',
                        help='CNN model to use for image classification; default is vgg')

    # arg 3 - file that contains all labels associated to dogs
    parser.add_argument('--dogfile', type=str, default='dognames.txt',
                        help='file that contains all labels associated to dogs;default is dognames.txt')

    # Assigns variable in_args to parse_args()
    in_args = parser.parse_args()

    return in_args


def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels based upon the filenames of the image
    files. Reads in pet filenames and extracts the pet image labels from the
    filenames and returns these label as petlabel_dic. This is used to check the accuracy of the image classifier model.
    The pet image labels are in all lower letters, have a single space separating each word in the multi-word pet labels, and that they correctly represent the filenames.

    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by pretrained CNN models (string)
    Returns:
     petlabels_dic - Dictionary storing image filename (as key) and Pet Image
                     Labels (as value)
    """
    filename_list = listdir(image_dir)
    #print("\nPrints 10 filenames from folder ", image_dir)
    # for idx in range(0, 10, 1):
    #print("%2d file: %-25s" % (idx + 1, filename_list[idx]))

    petlabels_dic = dict()
    for filename in filename_list:
        if filename not in petlabels_dic:
            # d['mynewkey'] = 'mynewvalue'
            name = filename.split(".")[0]
            name = name.replace("_", " ").lower()
            final_name = ''.join(
                char for char in name if not char.isdigit()).rstrip(" ")
            petlabels_dic[filename] = final_name
        else:
            print("WARNING: ", filename, " already exists in dictionary!")

    #udacity solution
    # in_files = listdir(image_dir)
    # petlabels_dic2 = dict()
    # for idx in range(0, len(in_files), 1):
    #     if in_files[idx][0] != ".": #only for Mac
    #         image_name = in_files[idx].split("_")
    #         pet_label = ""
    #         for word in image_name:
    #             if word.isalpha():
    #                 pet_label += word.lower() + " "
    #
    #         pet_label = pet_label.strip()
    #
    #         if in_files[idx] not in petlabels_dic2:
    #             petlabels_dic2[in_files[idx]] = pet_label
    #
    #         else:
    #             print("Warning: Duplicate files exist in directory",
    #                   in_files[idx])
    #
    # print("\n PRINTING petlabels_dic2")
    # print_petlabels_dict(petlabels_dic2)

    return petlabels_dic


def classify_images(images_dir, petlabel_dic, model):
    """
    Creates classifier labels with classifier function, compares labels, and 
    creates a dictionary containing both labels and comparison of them to be
    returned.
     PLEASE NOTE: This function uses the classifier() function defined in 
     classifier.py within this function. The proper use of this function is
     in test_classifier.py Please refer to this program prior to using the 
     classifier() function to classify images in this function. 
     Parameters: 
      images_dir - The (full) path to the folder of images that are to be
                   classified by pretrained CNN models (string)
      petlabel_dic - Dictionary that contains the pet image(true) labels
                     that classify what's in the image, where its key is the
                     pet image filename & its value is pet image label where
                     label is lowercase with space between each word in label 
      model - pre-trained CNN whose architecture is indicated by this
      parameter,
              values must be: resnet alexnet vgg (string)
     Returns:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)   where 1 = match between pet image and 
                    classifier labels and 0 = no match between labels
    """
    results_dic = {}

    for filename in petlabel_dic.keys():
        pet_label = petlabel_dic[filename]
        path = images_dir + "/" + filename
        classifier_label = classifier(path, model)
        classifier_label = classifier_label.lower()
        # remove leading and trailing whitespaces
        classifier_label = classifier_label.strip()
        found_index = classifier_label.find(pet_label)
        # if found, make sure the pet_label is a whole standalone word within
        # the classifier_label and not part of another word
        # example: cat can be part of polecat, which is a skunk, and that
        # would result in incorrect classification
        is_whole_word_match = 0
        #if found_index >= 0:
            # remove whitespace after comma
        # c_label = classifier_label.replace(", ", ",")
            # create list from classifier_label
        # label_list = c_label.split(",")
        #  if pet_label in label_list:
        #       is_whole_word_match = 1

        if found_index >= 0:
            conda = found_index == 0 and len(pet_label) == len(
                classifier_label)
            condb = found_index == 0 or classifier_label[
                found_index - 1] == " "
            condc = found_index + len(pet_label) == len(classifier_label)
            condd = classifier_label[found_index + len(pet_label):
                                     found_index + len(pet_label) + 1] in (
                    ",", " ")
            if conda or (condb and (condc or condd)):
                is_whole_word_match = 1

        value_list = [pet_label, classifier_label, is_whole_word_match]
        if pet_label not in results_dic:
            results_dic[pet_label] = value_list

    return results_dic


def classify_images_udacity(images_dir, petlabel_dic, model):
    """
    Creates classifier labels with classifier function, compares labels, and
    creates a dictionary containing both labels and comparison of them to be
    returned.
     PLEASE NOTE: This function uses the classifier() function defined in
     classifier.py within this function. The proper use of this function is
     in test_classifier.py Please refer to this program prior to using the
     classifier() function to classify images in this function.
     Parameters:
      images_dir - The (full) path to the folder of images that are to be
                   classified by pretrained CNN models (string)
      petlabel_dic - Dictionary that contains the pet image(true) labels
                     that classify what's in the image, where its key is the
                     pet image filename & its value is pet image label where
                     label is lowercase with space between each word in label
      model - pre-trained CNN whose architecture is indicated by this
      parameter,
              values must be: resnet alexnet vgg (string)
     Returns:
      results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)   where 1 = match between pet image and
                    classifier labels and 0 = no match between labels
    """
    results_dic = dict()
    for key in petlabel_dic:
        model_label = classifier(images_dir+key, model)
        model_label = model_label.lower()
        model_label = model_label.strip()
        # defines truth as pet image label and tries to find it using find()
        # string function to find it within classifier_label(model_label)
        truth = petlabel_dic[key]
        found = model_label.find(truth)
        if found >= 0:
            conda = found == 0 and len(truth) == len(
                model_label)
            condb = found == 0 or model_label[
                found - 1] == " "
            condc = found + len(truth) == len(model_label)
            condd = model_label[found + len(truth):
                                     found + len(truth) + 1] in (
                    ",", " ")
            if conda or (condb and (condc or condd)):
                if key not in results_dic:
                    results_dic[key] = [truth, model_label, 1]
        # found within a word/term not a label existing on its own
        else:
            if key not in results_dic:
                results_dic[key] = [truth, model_label, 0]

    return results_dic


def adjust_results4_isadog(results_dic, dogsfilename):
    """
    Adjusts the results dictionary to determine if classifier correctly 
    classified images 'as a dog' or 'not a dog' especially when not a match. 
    Demonstrates if model architecture correctly classifies dog images even if
    it gets dog breed wrong (not a match).
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifier labels and 0 = no match between labels
                    --- where idx 3 & idx 4 are added by this function ---
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
     dogsfile - A text file that contains names of all dogs from ImageNet 
                1000 labels (used by classifier model) and dog names from
                the pet image files. This file has one dog name per line
                dog names are all in lowercase with spaces separating the 
                distinct words of the dog name. This file should have been
                passed in as a command line argument. (string - indicates 
                text file's name)
    Returns:
           None - results_dic is mutable data type so no return needed.
    """
    # a match between classifier label and pet label match dogsfilename entry
    # if pet label in dogsfilename list, idx 3 = 1
    # if classifier label in dogsfilename list, idx 4 is 1
    dogsname_dic = dict()
    try:
        with open(dogsfilename) as f:
            for line in f:
                line = line.rstrip()
                if line not in dogsname_dic:
                    dogsname_dic[line] = 1
                else:
                    print("WARNING: duplicate dog name: " + line)

        print("dogsname_dic length = ", len(dogsname_dic))
    except BaseException as be:
        print("*****  ERROR *****")
        print(be)

    for filename in results_dic:
        #pet label image IS of dog/found in dognames_dic
        pet_label = results_dic[filename][0]
        classifier_label = results_dic[filename][1]
        if pet_label in dogsname_dic:
            if classifier_label in dogsname_dic:
                #if classifier_label in dognames_dic, extend by 1, 1
                results_dic[filename].extend((1, 1))
            else:
                #classifier is not a dog; extend by 1.0
                results_dic[filename].extend((1, 0))
        else:
            if classifier_label in dogsname_dic:
                results_dic[filename].extend((0, 1))
            else:
                results_dic[filename].extend((0, 0))


def adjust_results4_isadog_udacity(results_dic, dogsfile):
    """
    Adjusts the results dictionary to determine if classifier correctly
    classified images 'as a dog' or 'not a dog' especially when not a match.
    Demonstrates if model architecture correctly classifies dog images even if
    it gets dog breed wrong (not a match).
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and
                            classifier labels and 0 = no match between labels
                    --- where idx 3 & idx 4 are added by this function ---
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and
                            0 = pet Image 'is-NOT-a' dog.
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image
                            'as-a' dog and 0 = Classifier classifies image
                            'as-NOT-a' dog.
     dogsfile - A text file that contains names of all dogs from ImageNet
                1000 labels (used by classifier model) and dog names from
                the pet image files. This file has one dog name per line
                dog names are all in lowercase with spaces separating the
                distinct words of the dog name. This file should have been
                passed in as a command line argument. (string - indicates
                text file's name)
    Returns:
           None - results_dic is mutable data type so no return needed.
    """

    dognames_dic = dict()
    with open(dogsfile, "r") as infile:
        line = infile.readline()
        while line != "":
            line = line.rstrip()
            if line not in dognames_dic:
                dognames_dic[line] = 1
            else:
                print("Warning: duplicate dognames", line)

            line = infile.readline()

    for key in results_dic:
        if results_dic[key][0] in dognames_dic:
            if results_dic[key][1] in dognames_dic:
                results_dic[key].extend((1, 1))
            else:
                results_dic[key].extend((1, 0))
        else:
            if results_dic[key][1] in dognames_dic:
                results_dic[key].extend((0, 1))
            else:
                results_dic[key].extend((0, 0))


def calculates_results_stats(results_dic):
    """
    Calculates statistics of the results of the run using classifier's model 
    architecture on classifying images. Then puts the results statistics in a 
    dictionary (results_stats) so that it's returned for printing as to help
    the user to determine the 'best' model for classifying images. Note that 
    the statistics calculated as the results are either percentages or counts.
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
    Returns:
     results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's 
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value 
    """
    #{'Beagle_01141.jpg': ['beagle', 'walker hound, walker foxhound', 0, 1, 1]}

    # key = statistic's name (e.g. n_correct_dogs, pct_correct_dogs, n_correct_breed, pct_correct_breed)
    # value = statistic's value (e.g. 30, 100%, 24, 80%)
    # example_dictionary = {'n_correct_dogs': 30, 'pct_correct_dogs': 100.0, 'n_correct_breed': 24, 'pct_correct_breed': 80.0}

    results_stats = dict()
    # sets all counters to initial values of zero so they can be incremented
    # while processing through the images in results_dic
    results_stats['n_dogs_img'] = 0
    results_stats['n_match'] = 0
    results_stats['n_correct_dogs'] = 0
    results_stats['n_correct_notdogs'] = 0
    results_stats['n_correct_breed'] = 0

    for key in results_dic:
        # labels match exactly
        if results_dic[key][2] == 1:
            results_stats['n_match'] += 1

        # pet image label is a dog AND labels match - counts correct breed
        if sum(results_dic[key][2:]) == 3:
            results_stats['n_correct_breed'] += 1

        # pet image label is a dog - counts num dog images
        if results_dic[key][3] == 1:
            results_stats['n_dogs_img'] += 1

            # classifier classifies image as Dog (& pet image is a dog)
            # counts number of correct dog classifications
            if results_dic[key][4] == 1:
                results_stats['n_correct_dogs'] += 1

        # pet image label is NOT a dog
        else:
            # classifier classifies image as NOT a Dog
            #  (& pet image is NOT a dog)
            # counts number of correct dog classifications
            if results_dic[key][4] == 0:
                results_stats['n_correct_notdogs'] += 1

    # calc num total images
    results_stats['n_images'] = len(results_dic)

    # calc num of not-a-dog images using images & dog images counts
    results_stats['n_notdogs_img'] = (results_stats['n_images'] -
                                      results_stats['n_dogs_img'])

    # calc % correct matches
    results_stats['pct_match'] = (results_stats['n_match'] /
                                  results_stats['n_images']) * 100.0

    # calc % correct matches
    results_stats['pct_correct_dogs'] = (results_stats['n_correct_dogs'] /
                                         results_stats['n_dogs_img']) * 100.0

    # calc % correct breed of dog
    results_stats['pct_correct_breed'] = (results_stats['n_correct_breed'] /
                                          results_stats['n_dogs_img']) * 100.0

    # calc % correct not-a-dog images
    # uses conditional statement for when no 'not a dog' images were submitted
    if results_stats['n_notdogs_img'] > 0:
        results_stats['pct_correct_notdogs'] = (results_stats[
                                                    'n_correct_notdogs'] /
                                                results_stats['n_notdogs_img']) *100.0
    else:
        results_stats['pct_correct_notdogs'] = 0.0

    return results_stats


def print_results(results_dic, results_stats, model,
                  print_incorrect_dogs = False, print_incorrect_breed = False):
    """
    Prints summary results on the classification and then prints incorrectly 
    classified dogs and incorrectly classified dog breeds if user indicates 
    they want those printouts (use non-default values)
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifier labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
      results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's 
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value 
      model - pretrained CNN whose architecture is indicated by this parameter,
              values must be: resnet alexnet vgg (string)
      print_incorrect_dogs - True prints incorrectly classified dog images and 
                             False doesn't print anything(default) (bool)  
      print_incorrect_breed - True prints incorrectly classified dog breeds and 
                              False doesn't print anything(default) (bool) 
    Returns:
           None - simply printing results.
    """

    # OLD STRING FORMAT see following link:
    # https://docs.python.org/2/library/stdtypes.html#string-formatting
    # NEW STRING FORMAT see following link:
    # https://docs.python.org/3/library/string.html#format-string-syntax
    print("/nResults Summary for Model Architecture: ", model.upper())
    print("%20s: %3d" % ("N Images", results_stats['n_images']))
    print("%20s: %3d" % ("N Dog Images", results_stats['n_dogs_img']))
    print("%20s: %3d" % ("N Not-Dog Images", results_stats['n_notdogs_img']))

    # prints summary stats on model run
    print(" ")
    for key in results_stats:
        if key[0] == 'p':
            print("%20s: %5.1f" % (key, results_stats[key]))

    if (print_incorrect_dogs and
            ((results_stats['n_correct_dogs'] +
              results_stats['n_correct_notdogs'])
             != results_stats['n_images'])):
        print("\nINCORRECT Dog/NOT Dog Assignments:")
        for key in results_dic:
            if sum(results_dic[key][3:]) == 1:
                print("Real: {0} Classifier: {1}".format(
                    results_dic[key][0], results_dic[key][1]))

    if (print_incorrect_breed and
            (results_stats['n_correct_dogs'] != results_stats[
                'n_correct_breed'])):
        print("\nINCORRECT Dog Breed Assignment:")
        for key in results_dic:
            if sum(results_dic[key][3:]) == 2 and results_dic[key][2] == 0:
                print("Real: {0} Classifier: {1}".format(
                    results_dic[key][0], results_dic[key][1]))


def print_result_dic(result_dic):
    # temp code to print out result_dic
    print("\nprint_result_dic")
    print("\nMATCH:")
    n_match = 0
    n_notmatch = 0

    for key in result_dic:
        if result_dic[key][2] == 1:
            n_match += 1
            print("Pet Label: %-26s    Classifier Label: %-30s" % (result_dic[
                                                                 key][0],
                                                        result_dic[key][1]))

    print("\nNOT A MATCH:")

    for key in result_dic:
        if result_dic[key][2] == 0:
            n_notmatch += 1
            print("Pet Label: %-26s    Classifier Label: %-30s" % (result_dic[
                                                                  key][0],
                                                        result_dic[key][1]))

    print("\n# Total Images:", n_match + n_notmatch, "# Matches:", n_match,
          " # NOT MATCH:", n_notmatch)


def print_petlabels_dict(petlabels_dict):
    print("petlabels_dict has ", len(petlabels_dict), " key-value pairs. ")
    prnt = 0
    for key in petlabels_dict:
        print("{} key: {} ; value: {}".format((prnt+1), key, petlabels_dict[key]))
        prnt += 1


def print_command_line_args(in_args):
    print("arg1 --dir: ", in_args.dir, "; arg2 --arch: ", in_args.arch,
          "; arg3 --dogfile: ", in_args.dogfile)


def print_adjust_results4_isadog(result_dic):
    match = 0
    nomatch = 0
    print("\nMATCH:")
    for key in result_dic:
        if result_dic[key][2] == 1:
            match += 1
            print("Pet Label: %-26s    Classifier Label: %-30s PetLabelDog: "
                  "%1d ClassLabelDog: %1d" % (result_dic[key][0],
                                              result_dic[key][1],
                                              result_dic[key][3],
                                              result_dic[key][4]))

    print("\nNOT A MATCH:")
    for key in result_dic:
        if result_dic[key][2] == 0:
            nomatch += 1
            print("Pet Label: %-26s    Classifier Label: %-30s PetLabelDog: "
                  "%1d ClassLabelDog: %1d" % (result_dic[key][0],
                                              result_dic[key][1],
                                              result_dic[key][3],
                                              result_dic[key][4]))

    print("\n# Total Images:", match + nomatch, "# Matches:", match,
          " # NOT MATCH:", nomatch)


def check_results_stats(results_stats, result_dic):
    n_images = len(result_dic)
    n_pet_dog = 0
    n_class_cdog = 0
    n_class_cnotd = 0
    n_match_breed = 0
    for key in result_dic:
        if result_dic[key][2] == 1:
            if result_dic[key][3] == 1:
                n_pet_dog += 1
                if result_dic[key][4] == 1:
                    n_class_cdog += 1
                    n_match_breed += 1
            else:
                if result_dic[key][4] == 0:
                    n_class_cnotd += 1
        else:
            if result_dic[key][3] == 1:
                n_pet_dog += 1
                if result_dic[key][4] == 0:
                    n_class_cnotd += 1

    n_pet_notd = n_images - n_pet_dog
    pct_corr_dog = (n_class_cdog / n_pet_dog)*100
    pct_corr_notdog = (n_class_cnotd / n_pet_notd)*100
    pct_corr_breed = (n_match_breed / n_pet_dog)*100

    print("\n ** Function's Stats:")
    print("N images: %2d N Dog Images: %2d N Not Dog Images: %2d \nPct Corr "
          "dog: %5.1f Pct Correct not-a-dog: %5.1f Pct Correct Breed: %5.1f"
          % (results_stats['n_images'], results_stats['n_dogs_img'],
             results_stats['n_notdogs_img'], results_stats['pct_correct_dogs'],
             results_stats['pct_correct_notdogs'], results_stats['pct_correct_breed']))
    print("\n ** Check Stats:")
    print(
        "N images: %2d N Dog Images: %2d N Not Dog Images: %2d  \nPet Corr "
        "dog: %5.lf Pct Correct not-a-dog: %5.1f Pct Correct Breed: %5.1f"
        % (n_images, n_pet_dog, n_pet_notd, pct_corr_dog, pct_corr_notdog,
           pct_corr_breed))


# Call to main function to run the program
if __name__ == "__main__":
    main()
