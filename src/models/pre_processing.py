import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.append('../')
from utils import paragraphs_encoding, get_paragraphs_from_articles

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import statistics
from collections import Counter
import random
import json

def do_undersampling(train_paragraphs, train_labels, indexes_to_remove):
    indexes_to_remove.sort(reverse=True)

    for index in indexes_to_remove:
        del train_paragraphs[index]
        del train_labels[index]

    return train_paragraphs, train_labels

def select_indexes_to_remove(dict_density_ranges, undersample_value):
    indexes_to_remove = []

    for key, list_values in dict_density_ranges.items():
        print(key, len(list_values))
    
    for key, list_values in dict_density_ranges.items():
        number_elems_remove = 0
        if len(list_values) > undersample_value:
            number_elems_remove = len(list_values) - undersample_value
            elems_remove = random.sample(list_values, number_elems_remove)
            indexes_to_remove.extend(elems_remove)

    return indexes_to_remove


def density_ranges(values, low, high, bins):
    dict_ranges = {}
    step = (high - low) / bins

    curr_low = low
    curr_high = curr_low + step

    while (curr_high <= high):
        dict_ranges[(curr_low, curr_high)] = []
        curr_low = round(curr_low + step, 1)
        curr_high = round(curr_high + step, 1)

    for idx, val in enumerate(values):
        for pair in dict_ranges:
            if pair[1] != high:
                if val >= pair[0] and val < pair[1]:
                    curr_list = dict_ranges[pair]
                    curr_list.append(idx)
                    dict_ranges[pair] = curr_list
            else:
                if val >= pair[0] and val <= pair[1]:
                    curr_list = dict_ranges[pair]
                    curr_list.append(idx)
                    dict_ranges[pair] = curr_list
                    
    total_index = 0
    for elem in dict_ranges:
        total_index = total_index + len(dict_ranges[elem])

    if total_index != len(values):
        print("Error in building density ranges dictionary.")
        sys.exit()

    return dict_ranges


def pre_processing_set(args, iteration):

    if iteration != None:
        data_path = args.data_path + "iteration_" + str(iteration) + "/"
    else:
        data_path = args.data_path

    # Read train_paragraphs
    with open(data_path+'train_paragraphs.json') as file:
        train_paragraphs = json.load(file)

    # Read train_labels
    with open(data_path+'train_labels.json') as file:
        train_labels = json.load(file)

    # Read validation_paragraphs
    with open(data_path+'validation_paragraphs.json') as file:
        validation_paragraphs = json.load(file)

    # Read validation_labels
    with open(data_path+'validation_labels.json') as file:
        validation_labels = json.load(file)

    # Read test_paragraphs
    with open(data_path+'test_paragraphs.json') as file:
        test_paragraphs = json.load(file)

    # Read test_labels
    with open(data_path+'test_labels.json') as file:
        test_labels = json.load(file)

    #  Apply undersample to train data (only ready for regression)
    dict_density_values = density_ranges(train_labels, 0, 1, 10)
    indexes_to_remove = select_indexes_to_remove(dict_density_values, args.undersample)
    train_paragraphs, train_labels = do_undersampling(train_paragraphs, train_labels, indexes_to_remove)

    # Encode paragraphs 
    train_inputs, _, train_masks = paragraphs_encoding(train_paragraphs, args)
    validation_inputs, _, validation_masks = paragraphs_encoding(validation_paragraphs, args)
    test_inputs, test_lengths, test_masks = paragraphs_encoding(test_paragraphs, args)

    # Convert all inputs and labels into torch tensors, the required datatype 
    # for our model.
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    test_inputs = torch.tensor(test_inputs)

    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    test_labels = torch.tensor(test_labels)

    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)
    test_masks = torch.tensor(test_masks)

    # The DataLoader needs to know our batch size for training, so we specify it 
    # here.
    # For fine-tuning BERT on a specific task, the authors recommend a batch size of
    # 16 or 32.

    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    # Create the DataLoader for our validation set.
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=args.batch_size)

    # Create the DataLoader for our test set.
    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.batch_size)

    all_data = {
    'train_inputs': train_inputs, 
    'validation_inputs': validation_inputs, 
    'test_inputs': test_inputs,
    'train_dataloader': train_dataloader,
    'validation_dataloader': validation_dataloader,
    'test_dataloader': test_dataloader
    }

    print("\nTrain, Validation and Test have been pre-processed and saved.\n")

    return all_data