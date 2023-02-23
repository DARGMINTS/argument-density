from collections import defaultdict
import os
import shutil
from typing import DefaultDict
# need this because of the following error:
# forrtl: error (200): program aborting due to control-C event
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

# https://stackoverflow.com/questions/59823283/could-not-load-dynamic-library-cudart64-101-dll-on-tensorflow-cpu-only-install
# https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#import tensorflow as tf

import torch
import torch.nn.functional as F
torch.cuda.empty_cache()

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from bert_models import BertForSequenceClassification

import sys
sys.path.append('../')

from utils import mse, format_time, dir_path
from pre_processing import pre_processing_set

import numpy as np
import pandas as pd
import time
import statistics
import random
import datetime
import argparse
import json

from sklearn.metrics import confusion_matrix, classification_report

file_report_model = ''

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

def show_confusion_matrix(confusion_matrix):
  hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
  hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
  hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()

def do_train_epoch(model_type, model, device, epochs, epoch_i, train_dataloader, optimizer, scheduler):

    # ========================================
    #               Training
    # ========================================
    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128
        
    # Perform one full pass over the training set.

    print("")
    file_report_model.write('')
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    file_report_model.write('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs) + "\n")
    print('Training...')
    file_report_model.write('Training...\n')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_loss = 0
    eval_accuracy = 0
    nb_eval_steps = 0
    loss_cross_entropy = torch.nn.CrossEntropyLoss().to(device)

    # Put the model into training mode. Don't be mislead--the call to 
    # `train` just changes the *mode*, it doesn't *perform* the training.
    # `dropout` and `batchnorm` layers behave differently during training
    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            file_report_model.write('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed) + "\n")

        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the 
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because 
        # accumulating the gradients is "convenient while training RNNs". 
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()        

        # Perform a forward pass (evaluate the model on this training batch).
        # This will return the loss (rather than the model output) because we
        # have provided the `labels`.
        # The documentation for this `model` function is here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

        # Track the number of batches
        nb_eval_steps += 1
        
        # The call to `model` always returns a tuple, so we need to pull the 
        # loss value out of the tuple.
        loss = outputs[0]
        #loss_other = loss_cross_entropy(outputs.logits, b_labels) #same output


        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)

    print("")
    file_report_model.write("")
    print("  Average training loss: {0:.8f}".format(avg_train_loss))
    file_report_model.write("  Average training loss: {0:.8f}".format(avg_train_loss) + "\n")

    print("  Training epoch took: {:}".format(format_time(time.time() - t0)))
    file_report_model.write("  Training epoch took: {:}".format(format_time(time.time() - t0)))
    file_report_model.write("\n")

    return avg_train_loss

def do_validation(model_type, model, device, validation_dataloader):
  # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    file_report_model.write("\n")
    print("Running Validation...")
    file_report_model.write("Running Validation...\n")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables 
    eval_loss = 0
    nb_eval_steps = 0
    loss_cross_entropy = torch.nn.CrossEntropyLoss().to(device)
    loss_mse = torch.nn.MSELoss().to(device)
    eval_accuracy = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        
        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():        

            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have
            # not provided labels.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        
        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs.logits
        
        # Calculate the loss for this batch of test paragraphs.
        tmp_eval_loss = loss_mse(logits.squeeze(), b_labels.squeeze())
        
        # Accumulate the total Loss.
        eval_loss += tmp_eval_loss.item()

        # Track the number of batches
        nb_eval_steps += 1

    # Report the final loss for this validation run.
    print("  Loss: ", eval_loss/nb_eval_steps)
    file_report_model.write("  Loss: " + str(eval_loss/nb_eval_steps) + "\n")

    print("  Validation took: {:}".format(format_time(time.time() - t0)))
    file_report_model.write("  Validation took: {:}".format(format_time(time.time() - t0)))
    file_report_model.write("\n\n")

    return eval_loss/nb_eval_steps

def do_prediction(model_type, model, device, test_dataloader, test_inputs, set_type):
    
    # Prediction on test set
    print('Predicting labels for {:,} paragraphs...'.format(len(test_inputs)))
    print(set_type)
    file_report_model.write('\n' + 'Predicting labels for {:,} paragraphs...'.format(len(test_inputs)))
    file_report_model.write(set_type)
    file_report_model.write("\n")

    # Put model in evaluation mode
    model.eval()

    loss_mse = torch.nn.MSELoss().to(device)
    
    # Tracking variables 
    predictions = []
    prediction_probs = []
    real_values = []

    # Predict 
    for batch in test_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        logits = outputs[0]
        predictions.extend(logits)
        real_values.extend(b_labels)

        # Report the final MSE for test.
        result_mse = loss_mse(torch.tensor(predictions), torch.tensor(real_values))
        print(" MSE: ", result_mse)
        print("\n")
        file_report_model.write(" MSE: " + str(result_mse) + "\n\n")

        predictions = torch.stack(predictions).cpu()
        predictions_list = predictions.tolist()
        # https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists
        predictions_flat = [item for sublist in predictions_list for item in sublist]

        real_values = torch.stack(real_values).cpu()
        real_values_list = real_values.tolist()

        file_report_model.write(" Mean (of all predictions): " + str(statistics.mean(predictions_flat)) + "\n")
        file_report_model.write(" Median (of all predictions): " + str(statistics.median(predictions_flat)) + "\n")
        file_report_model.write(" Max (of all predictions): " + str(max(predictions_flat)) + "\n")
        file_report_model.write(" Min (of all predictions): " + str(min(predictions_flat)) + "\n")

        return predictions_flat, -1, real_values_list

def run(all_data, args, model_name_folder):

    # If there's a GPU available...
    if torch.cuda.is_available():    

        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")

        print('PyTorch: There are %d GPU(s) available.' % torch.cuda.device_count())
        file_report_model.write('PyTorch: There are %d GPU(s) available.' % torch.cuda.device_count() + "\n")

        print('PyTorch: We will use the GPU:', torch.cuda.get_device_name(0))
        file_report_model.write('PyTorch: We will use the GPU: ' + torch.cuda.get_device_name(0))
        file_report_model.write("\n\n")
    # If gpu is not available...
    else:
        print('No GPU available, using the CPU instead.')
        file_report_model.write('No GPU available, using the CPU instead.\n')
        device = torch.device("cpu")


    # Load inputs
    train_inputs = all_data['train_inputs']
    validation_inputs = all_data['validation_inputs']
    test_inputs = all_data['test_inputs']

    # Load dataloaders
    train_dataloader = all_data['train_dataloader'] # torch.load('../../data/train_dataloader.pth')
    validation_dataloader = all_data['validation_dataloader'] # torch.load('../../data/validation_dataloader.pth')
    test_dataloader = all_data['test_dataloader'] #torch.load('../../data/test_dataloader.pth')

    # Load BertForSequenceClassification, the pretrained BERT model with a single 
    # linear classification layer on top. 
    """ model = BertForSequenceClassification.from_pretrained(
        "bert-base-multilingual-cased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 1, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    ) """

    if args.model_type == "BertForSequenceClassification":
        model = BertForSequenceClassification(
            model_name = args.model_name, # Use the 12-layer BERT model, with cased vocab.
            num_labels = args.num_labels, # The number of output labels--2 for binary classification.
                            # You can increase this for multi-class tasks.   
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )
    else:
        print("Error! No model specified...")
        sys.exit()

    # Freeze BERT weights parameters
    if args.freeze == True and args.model_type == 'BertForSequenceClassification':
        for p in model.bert.parameters():
            p.requires_grad = False
    elif args.freeze == False:
        pass
    else:
        print("Error. Confusion with freeze.")
        sys.exit()
        #for p in model.named_parameters():
            #print(p)

    # Tell pytorch to run this model on the GPU.
    if args.use_cuda == True:
        model.cuda()

    if args.optimizer == 'AdamW':
        # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
        # I believe the 'W' stands for 'Weight Decay fix"
        optimizer = AdamW(model.parameters(),
                        lr = args.learning_rate, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                        eps = args.epsilon # args.adam_epsilon  - default is 1e-6.
                        )
    else:
        print("Optimizer not found.")
        file_report_model.write('Optimizer not found.\n')
        sys.exit()

    # Number of training epochs (authors recommend between 2 and 4)
    epochs = args.epochs

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    # Set the seed value all over the place to make this reproducible.
    seed_val = args.seed_value

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # Store the average loss after each epoch so we can plot them.
    train_loss_values = []
    val_loss_values = []

    #History
    min_val_loss = 100000
    times_not_improved = 0

    # For each epoch...
    for epoch_i in range(0, epochs):
        
        train_loss = do_train_epoch(args.model_type, model, device, epochs, epoch_i, train_dataloader, optimizer, scheduler)
        val_loss = do_validation(args.model_type, model, device, validation_dataloader)

        train_loss_values.append(train_loss)
        val_loss_values.append(val_loss)

        if args.early_stop == True:
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                times_not_improved = 0
                torch.save(model.state_dict(), model_name_folder + '/model_best_epoch.bin')
                print("\nSaving best epoch: " + str(epoch_i+1) + "\n")
                file_report_model.write(" Saving best epoch: " + str(epoch_i+1) + "\n\n")
            else:
                times_not_improved += 1
                if times_not_improved == args.patience:
                    # https://discuss.pytorch.org/t/how-to-save-the-best-model/84608
                    model.load_state_dict(torch.load(model_name_folder + '/model_best_epoch.bin'))
                    print("\nInfo: Training will stop since the number of epochs tolerance has been achieved without improvement.")
                    file_report_model.write("\nInfo: Training will stop since the number of epochs tolerance has been achieved without improvement.\n")
                    break
        else:
            torch.save(model.state_dict(), model_name_folder + '/model_last_epoch.bin')
            print("Saving last epoch: " + str(epoch_i+1) + "\n")
            file_report_model.write(" Saving last epoch: " + str(epoch_i+1) + "\n\n")

    if args.early_stop == True and times_not_improved < args.patience:
        model.load_state_dict(torch.load(model_name_folder + '/model_best_epoch.bin'))

    print("")
    print("--> Training complete!")
    file_report_model.write('\n--> Training Complete!\n')

    train_predictions, train_prediction_probs, train_true_labels = do_prediction(args.model_type, model, device, train_dataloader, train_inputs, "train")
    validation_predictions, validation_prediction_probs, validation_true_labels = do_prediction(args.model_type, model, device, validation_dataloader, validation_inputs, "validation")
    test_predictions, test_prediction_probs, test_true_labels = do_prediction(args.model_type, model, device, test_dataloader, test_inputs, "test")

    #Save these information for graphic usage
    model_path = args.model_path

    with open(model_path+'train_predictions.json', 'w') as file:
        json.dump(train_predictions, file)
    with open(model_path+'train_true_labels.json', 'w') as file:
        json.dump(train_true_labels, file)
        
    with open(model_path+'validation_predictions.json', 'w') as file:
        json.dump(validation_predictions, file)
    with open(model_path+'validation_true_labels.json', 'w') as file:
        json.dump(validation_true_labels, file)

    with open(model_path+'test_predictions.json', 'w') as file:
        json.dump(test_predictions, file)
    with open(model_path+'test_true_labels.json', 'w') as file:
        json.dump(test_true_labels, file)
        
    file_report_model.close()

    return {'train_predictions': train_predictions, 'train_true_labels': train_true_labels,
            'validation_predictions': validation_predictions, 'validation_true_labels': validation_true_labels,
            'test_predictions': test_predictions, 'test_true_labels': test_true_labels}

def write_report_file(args, file_report_model, iteration):
    file_report_model.write("Parameters:\n\n")
    if iteration >= 1:
        file_report_model.write("Data Path: " + str(args.data_path) + "iteration_" + str(iteration) + "\n")
        file_report_model.write("Model Path: " + str(args.model_path) + "/iteration_" + str(iteration) + "\n")
    else:
        file_report_model.write("Data Path: " + str(args.data_path) + "\n")
        file_report_model.write("Model Path: " + str(args.model_path) + "\n")
    file_report_model.write("Column to use: " + str(args.column) + "\n")
    file_report_model.write("Model Type: " + str(args.model_type) + "\n")
    file_report_model.write("Model Name: " + str(args.model_name) + "\n")
    file_report_model.write("Tokenizer Type: " + str(args.tokenizer_type) + "\n")
    file_report_model.write("Tokenizer Name: " + str(args.tokenizer_name) + "\n")
    file_report_model.write("Undersample Value: " + str(args.undersample) + "\n")
    file_report_model.write("Number of epochs: " + str(args.epochs) + "\n")
    file_report_model.write("Early Stop: " + str(args.early_stop) + "\n")
    file_report_model.write("Patience: " + str(args.patience) + "\n")
    file_report_model.write("Batch Size: " + str(args.batch_size) + "\n")
    file_report_model.write("Max Length: " + str(args.max_length) + "\n")
    file_report_model.write("Optimizer: " + str(args.optimizer) + "\n")
    file_report_model.write("Learning Rate: " + str(args.learning_rate) + "\n")
    file_report_model.write("Epsilon: " + str(args.epsilon) + "\n")
    file_report_model.write("Freeze: " + str(args.freeze) + "\n")
    file_report_model.write("Seed Value: " + str(args.seed_value) + "\n\n")
    file_report_model.write("Use Cuda: " + str(args.use_cuda) + "\n\n")

if __name__ == '__main__':
    # Initialize the Parser
    parser = argparse.ArgumentParser(description ='Train BERT models.')
    
    # Add arguments
    parser.add_argument('--model_type', type=str, default='BertForSequenceClassification', metavar='', required=False, help='Model type')
    parser.add_argument('--model_name', type=str, default='bert-base-multilingual-cased', metavar='', required=False, help='Model name')
    parser.add_argument('--tokenizer_type', type=str, default='BertTokenizer', metavar='', required=False, help='Tokenizer type')
    parser.add_argument('--tokenizer_name', type=str, default='bert-base-multilingual-cased', metavar='', required=False, help='Tokenizer name')
    
    parser.add_argument('-dp','--data_path', type=str, default='../../data/set_holdout_union_regression_v3/', metavar='', required=False, help='Data Path for Train/Dev/Test')
    parser.add_argument('-mp','--model_path', type=str, default='../../models/holdout_bert_union_regression_v3_seed_42', metavar='', required=False, help='Model Path')

    parser.add_argument('-col','--column', type=str, default='argumentative_density_union', metavar='', required=False, help='Column to consider for argumentative values')
    parser.add_argument('-nl','--num_labels', type=int, default=1, metavar='', required=False, help='Number of labels')
    parser.add_argument('-uv','--undersample', type=int, default=1000000, metavar='', required=False, help='Undersampling absolute value. Examples above this value are randomly deleted.')

    parser.add_argument('-e','--epochs', type=int, default=2, metavar='', required=False, help='Number of Epochs')
    
    parser.add_argument('--early_stop', default=True, action='store_true', help='Early Stopping to Halt the Training of Neural Networks At the Right Time')
    parser.add_argument('--no_early_stop', dest='early_stop', action='store_false', help='Early Stopping to Halt the Training of Neural Networks At the Right Time')

    parser.add_argument('-ptc','--patience', type=int, default=3, metavar='', required=False, help='Patience')
    parser.add_argument('-b','--batch_size', type=int, default=16, metavar='', required=False, help='Batch Size')
    parser.add_argument('-m','--max_length', type=int, default=128, metavar='', required=False, help='Max Length')
    parser.add_argument('-o','--optimizer', type=str, default='AdamW', metavar='', required=False, help='Optimizer')
    parser.add_argument('-lr','--learning_rate', type=float, default=2e-5, metavar='', required=False, help='The learning rate to use.')
    parser.add_argument('-eps','--epsilon', type=float, default=1e-6, metavar='', required=False, help='Adams epsilon for numerical stability')

    parser.add_argument('--freeze', default=False, action='store_true', help='Freeze BERT layers except classifier')
    parser.add_argument('--no_freeze', dest='freeze', action='store_false', help='Freeze BERT layers except classifier')

    parser.add_argument('-sv','--seed_value', type=int, default=42, metavar='', required=False, help='Random Seed Value')

    #https://stackoverflow.com/questions/52403065/argparse-optional-boolean
    parser.add_argument('--use_cuda', default=True, action='store_true')
    parser.add_argument('--no_use_cuda', dest='use_cuda', action='store_false')

    # Parse arguments
    args = parser.parse_args()

    model_name_folder = args.model_path

    #https://stackoverflow.com/questions/11660605/how-to-overwrite-a-folder-if-it-already-exists-when-creating-it-with-makedirs
    if os.path.exists(model_name_folder):
        shutil.rmtree(model_name_folder)
    os.makedirs(model_name_folder)

    # Create .txt file for reporting model parameters and results
    file_report_model = open(model_name_folder + "/report.txt", "a")
    write_report_file(args, file_report_model, 0)

    all_data = pre_processing_set(args, None)
    run(all_data, args, model_name_folder)