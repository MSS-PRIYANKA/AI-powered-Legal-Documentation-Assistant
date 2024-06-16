from transformers import BartTokenizer, BartForConditionalGeneration
import re

def format_text(text):
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Add space after periods, commas, and exclamation marks
    text = re.sub(r'([.,!])', r'\1 ', text)
    return text.strip()

def Summarization(text):
    # Load BART model and tokenizer
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

    # Tokenize input text
    inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)

    # Generate summary
    summary_ids = model.generate(inputs.input_ids,
                                  num_beams=4,
                                  length_penalty=2.0,
                                  max_length=500,
                                  min_length=300,
                                  early_stopping=True)

    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    summary = format_text(summary)
    return summary
