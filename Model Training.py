
import json
import os
import pandas as pd
from transformers import Trainer, TrainingArguments
import glob
import math
import random
import re
import argparse
import nltk
# Specify the path to the folder containing the JSON files
folder_path = 'Dataset'

# Initialize an empty list to store data from all files
all_data = []

# Iterate over all files in the folder
for file_name in os.listdir(folder_path):
    # Check if the file is a JSON file
    if file_name.endswith('.json'):
        # Construct the full file path
        file_path = os.path.join(folder_path, file_name)
        
        # Open the file and load its contents as JSON
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # Append the data to the list
        all_data.append(data)

# Convert the list of dictionaries into a DataFrame
df = pd.DataFrame(all_data)
df=df.iloc[:10]
# Now 'df' contains the data from all JSON files in a DataFrame
print(df)

df1=df.iloc[:, :9]


# Pre requisites for model training

import glob
from nltk import tokenize
import nltk
import transformers
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, Dataset
import pandas as pd
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

#Source - https://colab.research.google.com/drive/1Cy27V-7qqYatqMA7fEqG2kgMySZXw9I4?usp=sharing&pli=1
class LitModel(pl.LightningModule):
  # Instantiate the model
  def __init__(self, learning_rate, tokenizer, model):
    super().__init__()
    self.tokenizer = tokenizer
    self.model = model
    self.learning_rate = learning_rate
    # self.freeze_encoder = freeze_encoder
    # self.freeze_embeds_ = freeze_embeds
#     self.hparams = argparse.Namespace()

    self.hparams.freeze_encoder = True
    self.hparams.freeze_embeds = True
    self.hparams.eval_beams = 4
    # self.hparams = hparams

    if self.hparams.freeze_encoder:
      freeze_params(self.model.get_encoder())

    if self.hparams.freeze_embeds:
      self.freeze_embeds()
  
  def freeze_embeds(self):
    ''' freeze the positional embedding parameters of the model; adapted from finetune.py '''
    freeze_params(self.model.model.shared)
    for d in [self.model.model.encoder, self.model.model.decoder]:
      freeze_params(d.embed_positions)
      freeze_params(d.embed_tokens)

  # Do a forward pass through the model
  def forward(self, input_ids, **kwargs):
    return self.model(input_ids, **kwargs)
  
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
    return optimizer

  def training_step(self, batch, batch_idx):
    # Load the data into variables
    src_ids, src_mask = batch[0], batch[1]
    tgt_ids = batch[2]
    # Shift the decoder tokens right (but NOT the tgt_ids)
    decoder_input_ids = shift_tokens_right(tgt_ids, self.tokenizer.pad_token_id)

    # Run the model and get the logits
    outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False)
    lm_logits = outputs[0]
    # Create the loss function
    ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
    # Calculate the loss on the un-shifted tokens
    loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))

    return {'loss':loss}

  def validation_step(self, batch, batch_idx):

    src_ids, src_mask = batch[0], batch[1]
    tgt_ids = batch[2]

    decoder_input_ids = shift_tokens_right(tgt_ids, self.tokenizer.pad_token_id)
    
    # Run the model and get the logits
    outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False)
    lm_logits = outputs[0]

    ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
    val_loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))

    return {'loss': val_loss}
  
  # Method that generates text using the BartForConditionalGeneration's generate() method
  def generate_text(self, text, eval_beams, early_stopping = True, max_len = 1024):
    ''' Function to generate text '''
    generated_ids = self.model.generate(
        text["input_ids"],
        attention_mask=text["attention_mask"],
        use_cache=True,
        decoder_start_token_id = self.tokenizer.pad_token_id,
        num_beams= eval_beams,
        max_length = max_len,
        early_stopping = early_stopping
    )
    return [self.tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True) for w in generated_ids]

def freeze_params(model):
  ''' Function that takes a model as input (or part of a model) and freezes the layers for faster training
      adapted from finetune.py '''
  for layer in model.parameters():
    layer.requires_grade = False


# Create a dataloading module as per the PyTorch Lightning Docs
class SummaryDataModule(pl.LightningDataModule):
  def __init__(self, tokenizer, df, batch_size):
    super().__init__()
    self.tokenizer = tokenizer
    self.batch_size = batch_size
    self.data = df
     
  # Loads and splits the data into training, validation and test sets with a 60/20/20 split
  def prepare_data(self):
    self.train, self.validate, self.test = np.split(self.data.sample(frac=1), [int(.6*len(self.data)), int(.8*len(self.data))])

  # encode the sentences using the tokenizer  
  def setup(self, stage):
    self.train = encode_sentences(self.tokenizer, self.train['source'], self.train['target'])
    self.validate = encode_sentences(self.tokenizer, self.validate['source'], self.validate['target'])
    self.test = encode_sentences(self.tokenizer, self.test['source'], self.test['target'])

  # Load the training, validation and test sets in Pytorch Dataset objects
  def train_dataloader(self):
    dataset = TensorDataset(self.train['input_ids'], self.train['attention_mask'], self.train['labels'])                          
    train_data = DataLoader(dataset, sampler = RandomSampler(dataset), batch_size = self.batch_size)
    return train_data

  def val_dataloader(self):
    dataset = TensorDataset(self.validate['input_ids'], self.validate['attention_mask'], self.validate['labels']) 
    val_data = DataLoader(dataset, batch_size = self.batch_size)                       
    return val_data

  def test_dataloader(self):
    dataset = TensorDataset(self.test['input_ids'], self.test['attention_mask'], self.test['labels']) 
    test_data = DataLoader(dataset, batch_size = self.batch_size)                   
    return test_data



def shift_tokens_right(input_ids, pad_token_id):
  """ Shift input ids one token to the right, and wrap the last non pad token (usually <eos>).
      This is taken directly from modeling_bart.py
  """
  prev_output_tokens = input_ids.clone()
  index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
  prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
  prev_output_tokens[:, 1:] = input_ids[:, :-1]
  return prev_output_tokens

def encode_sentences(tokenizer, source_sentences, target_sentences, max_length=1024, min_length = 512, pad_to_max_length=True, return_tensors="pt"):
  ''' Function that tokenizes a sentence 
      Args: tokenizer - the BART tokenizer; source and target sentences are the source and target sentences
      Returns: Dictionary with keys: input_ids, attention_mask, target_ids
  '''

  input_ids = []
  attention_masks = []
  target_ids = []
  tokenized_sentences = {}

  for sentence in source_sentences:
    encoded_dict = tokenizer(
          sentence,
          max_length=max_length,
          padding="max_length" if pad_to_max_length else None,
          truncation=True,
          return_tensors=return_tensors,
          add_prefix_space = True
      )

    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

  input_ids = torch.cat(input_ids, dim = 0)
  attention_masks = torch.cat(attention_masks, dim = 0)

  for sentence in target_sentences:
    encoded_dict = tokenizer(
          sentence,
          max_length=min_length,
          padding="max_length" if pad_to_max_length else None,
          truncation=True,
          return_tensors=return_tensors,
          add_prefix_space = True
      )
    # Shift the target ids to the right
    # shifted_target_ids = shift_tokens_right(encoded_dict['input_ids'], tokenizer.pad_token_id)
    target_ids.append(encoded_dict['input_ids'])

  target_ids = torch.cat(target_ids, dim = 0)
  

  batch = {
      "input_ids": input_ids,
      "attention_mask": attention_masks,
      "labels": target_ids,
  }

  return batch


#-------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# import seaborn as sns

from mlxtend.plotting import plot_confusion_matrix

from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, BartConfig
#==========================DATA SELECTION AND LOADING==========================
df=df1
print(df.head())
print(df.info())
print(df.describe())

# from sklearn import preprocessing
# label_encoder = preprocessing.LabelEncoder()
# df = df.astype(str).apply(label_encoder.fit_transform)
print("-------------------------------------------")
print(" After label Encoding ")
print("------------------------------------------")
print()
X = df.drop(columns=['recitals'],axis=1)
Y = df['recitals']

#**************************DATA SPLITTING   *****************************************
print("6.DATA SPLITTING 80% TRAINING AND 20% TESTING ")
print("==================================================")
print("==================================================")
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print("X_train Shapes ",x_train.shape)
print("y_train Shapes ",y_train.shape)
print("x_test Shapes ",x_test.shape)
print("y_test Shapes ",y_test.shape)


import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer

# Load pre-trained BART model and tokenizer
model_name = 'facebook/bart-large-cnn'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
# Sample DataFrame with legal case judgments
data =df1
df = pd.DataFrame(data)

# Function to generate summaries
def generate_summary(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Apply summarization function to each row in the DataFrame
df['Summary'] = df['recitals'].apply(generate_summary)

# Display the DataFrame with summaries
print(df)
df['Summary']

#model

new_tokens = ['<F>', '<RLC>', '<A>', '<S>', '<P>', '<R>', '<RPC>']

special_tokens_dict = {'additional_special_tokens': new_tokens}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
bart_model.resize_token_embeddings(len(tokenizer))

summary_data = SummaryDataModule(tokenizer, df, batch_size = 2)
model = LitModel(learning_rate = 2e-5, tokenizer = tokenizer, model = bart_model)


trainer = pl.Trainer(gpus = 1,
                     max_epochs = 3,
                     min_epochs = 2,
                     auto_lr_find = False,
                     progress_bar_refresh_rate = 5,
                     precision = 16)

trainer.fit(model, summary_data)

trainer.save('summarized_legal_cases.pt', key='df', mode='w')


from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge

# Function to compute accuracy
def compute_accuracy(true_summaries, predicted_summaries):
    correct = 0
    total = len(true_summaries)
    for true_summary, predicted_summary in zip(true_summaries, predicted_summaries):
        if true_summary == predicted_summary:
            correct += 1
    accuracy = correct / total
    return accuracy

# Function to compute BLEU score
def compute_bleu_score(true_summaries, predicted_summaries):
    true_summaries = [[summary.split()] for summary in true_summaries]
    predicted_summaries = [summary.split() for summary in predicted_summaries]
    return corpus_bleu(true_summaries, predicted_summaries)

# Function to compute ROUGE score
def compute_rouge_score(true_summaries, predicted_summaries):
    rouge = Rouge()
    rouge_scores = rouge.get_scores(predicted_summaries, true_summaries, avg=True)
    return rouge_scores

# Example usage:
true_summaries = df['recitals']
predicted_summaries = df['Summary']

accuracy = compute_accuracy(true_summaries, predicted_summaries)
bleu_score = compute_bleu_score(true_summaries, predicted_summaries)
rouge_score = compute_rouge_score(true_summaries, predicted_summaries)

print("Accuracy:", accuracy)
print("BLEU Score:", bleu_score)
print("ROUGE Score:", rouge_score)


from transformers import BartForConditionalGeneration, BartTokenizer
model_name = 'facebook/bart-large-cnn'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
with open("input.txt", "r") as file:
    input_text = file.read()

print(input_text)
summary = generate_summary(input_text)
print("Input Text:")
print(input_text)
print("\nSummary:")
print(summary)


import fitz  # PyMuPDF
from transformers import BartForConditionalGeneration, BartTokenizer
import textwrap
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text()
    doc.close()
    return text
def text_summarizer_from_pdf(pdf_path):
    pdf_text = extract_text_from_pdf(pdf_path)

    model_name = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)

    inputs = tokenizer.encode("summarize: " + pdf_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=300, min_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    formatted_summary = "\n".join(textwrap.wrap(summary, width=80))
    return formatted_summary

def save_summary_as_pdf(pdf_path, summary):
    doc = fitz.open()

    page = doc.new_page()
    page.insert_text((10, 200), summary, fontname="helv", fontsize=12)  # Adjust the vertical position as needed

    output_pdf_path = pdf_path.replace(".pdf", "_summary.pdf")
    doc.save(output_pdf_path)
    doc.close()

    return output_pdf_path

pdf_file_path = "Sale Agreement.pdf"
summary = text_summarizer_from_pdf(pdf_file_path)
output_pdf_path = save_summary_as_pdf(pdf_file_path, summary)
print("Summary saved as PDF:", output_pdf_path)


