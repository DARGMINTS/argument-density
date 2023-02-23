from transformers import BertTokenizer
from bert_models import BertForSequenceClassification
import torch
import sys

torch.manual_seed(42) # use this for prediction consistency

PRE_TRAINED_MODEL_NAME = 'bert-base-multilingual-cased'
MODEL_FINE_TUNED = '../../models/holdout_bert_union_regression_v3_seed_42/model_best_epoch.bin'
SAMPLE_TEXT = 'Este é um teste simples.' # example text for prediction

# If there's a GPU available...
if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('PyTorch: There are %d GPU(s) available.' % torch.cuda.device_count())
    print('PyTorch: We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

# Load the BERT model
print('Loading BERT model...')
model = BertForSequenceClassification(
    model_name = PRE_TRAINED_MODEL_NAME, # Use the 12-layer BERT model, with cased vocab.
    num_labels = 1, # The number of output labels--1 for regression
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)
# Load the fine-tuned BERT model for regression
model.load_state_dict(torch.load(MODEL_FINE_TUNED), strict=False)
model = model.to(device)

encoding_speech = tokenizer.encode_plus(
  SAMPLE_TEXT,
  add_special_tokens=True, # Add '[CLS]' and '[SEP]'
  max_length=512,
  truncation = True,
  padding = 'max_length',  # Add [PAD]s
  return_attention_mask=True,
  return_token_type_ids=False,
  return_tensors='pt',  # Return PyTorch tensors
)

input_ids = encoding_speech['input_ids'].to(device)
attention_mask = encoding_speech['attention_mask'].to(device)

with torch.no_grad():
    output = model(input_ids, attention_mask)

prediction = output[0].item()
print("Density prediction: ", prediction) # density values range from 0 to 1