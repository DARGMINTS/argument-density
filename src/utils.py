import numpy as np
import sys
import datetime
import os
from transformers import BertTokenizer, DistilBertTokenizer

# Function to calculate the accuracy of our predictions vs labels
from sklearn.metrics import mean_squared_error

def mse(preds, labels):
   return mean_squared_error(labels, preds)

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def paragraphs_encoding(paragraphs, args):

    # Load the BERT tokenizer.
    if args.tokenizer_type == "BertTokenizer":
        print('Loading BERT tokenizer...')
        tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name, do_lower_case=False)
    elif args.tokenizer_type == "DistilBertTokenizer":
        print('Loading DistilBERT tokenizer...')
        tokenizer = DistilBertTokenizer.from_pretrained(args.tokenizer_name, do_lower_case=False)
    else:
        print("Error! No tokenizer specified.")
        sys.exit()

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    paragraphs_ids = []

    # Create attention masks
    attention_masks = []

    # Record the length of each sequence (after truncating to 512).
    lengths = []

    print('Tokenizing paragraphs...')

    greater_128 = 0
    sum_overflow = 0
    sum_tokens = 0

    # For every sentence...
    for para in paragraphs:
        
        # Report progress.
        if ((len(paragraphs_ids) % 500) == 0):
            print('  Read {:,} paragraphs.'.format(len(paragraphs_ids)))
        
        # Encode the sentence
        encoded_sent = tokenizer.encode_plus(
            text=para,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length = args.max_length,  # maximum length of a sentence
            truncation = True,
            padding = 'max_length',  # Add [PAD]s
            return_attention_mask = True,  # Generate the attention mask
            return_overflowing_tokens=True,
            #return_tensors = 'pt',  # ask the function to return PyTorch tensors
        )

        paragraphs_ids.append(encoded_sent['input_ids'])
        attention_masks.append(encoded_sent['attention_mask'])

        nr_tokens_paragraph = args.max_length + encoded_sent['num_truncated_tokens'] - 2 - len(encoded_sent['overflowing_tokens'])

        lengths.append(nr_tokens_paragraph)

        sum_tokens = sum_tokens + nr_tokens_paragraph

        if (len(encoded_sent['overflowing_tokens']) > 0):
            greater_128 = greater_128 + 1
            sum_overflow = sum_overflow + len(encoded_sent['overflowing_tokens'])

    return paragraphs_ids, lengths, attention_masks

def get_paragraphs_from_articles(set_ids, articleid_paragraphs): 
    set_paragraphs = []
    set_densities = []
    set_paragraphs_articleid = []
    set_class_labels = []
    for article_id in set_ids:
        # https://stackoverflow.com/questions/8653516/python-list-of-dictionaries-search
        dict_elem = next((item for item in articleid_paragraphs if item["article_id"] == article_id), None)
        if dict_elem != None:
            set_paragraphs.extend(dict_elem['paragraphs'])
            set_densities.extend(dict_elem['density_values'])
            set_paragraphs_articleid.extend([article_id] * len(dict_elem['paragraphs']))
            set_class_labels.extend(dict_elem['class_labels'])
        else:
            print("Error!")
            sys.exit()
    return set_paragraphs, set_densities, set_paragraphs_articleid
