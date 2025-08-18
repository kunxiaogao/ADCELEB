from transformers import AutoTokenizer, AutoModelForMaskedLM
import os
import re
from tqdm import tqdm
import sys
import torch
import sentencepiece
from transformers import AutoTokenizer, AutoModel
import torch
from numpy import save
import google.protobuf

def remove_enclosed_parts(input_string):
    # Use a regular expression to remove all parts of the string enclosed by < and >
    result = re.sub(r'<[^>]*>', '', input_string)
    return result.strip()

if __name__ == "__main__":
    input_dir = sys.argv[1] # path to transcripts
    output_dir = sys.argv[2]
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("T-Systems-onsite/cross-en-fr-roberta-sentence-transformer")
    model = AutoModel.from_pretrained("T-Systems-onsite/cross-en-fr-roberta-sentence-transformer")

    all_sents = sorted([os.path.join(input_dir, elem) for elem in os.listdir(input_dir)])
    for sentences in tqdm(all_sents, desc='extract text embeddings'):
        base_name = os.path.basename(sentences).split(".txt")[0]
        # sentences = open(sentences, 'r', encoding="utf-8",errors='ignore').read().strip().lower()
        with open(sentences, 'r', encoding="utf-8", errors='ignore') as file:
            sentences = file.read().strip()
            sentences = remove_enclosed_parts(sentences)
            if sentences != '':
                # Tokenize input text
                tokens = tokenizer(sentences, return_tensors="pt", max_length=512, truncation=True)

                # Forward pass through the model to get embeddings
                with torch.no_grad():
                    outputs = model(**tokens)

                # Extract the embeddings
                embeddings = outputs.last_hidden_state

                # Take the mean across embeddings
                mean_embedding = torch.mean(embeddings, dim=1)

                # Convert the mean embedding to a NumPy array
                mean_embedding_np = mean_embedding.numpy()

                # Save the mean embedding as a NumPy array
                save(output_dir + base_name + '.npy', mean_embedding_np)
