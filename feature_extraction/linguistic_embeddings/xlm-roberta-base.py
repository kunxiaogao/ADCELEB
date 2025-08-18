from tqdm import tqdm
import sys
import os
from numpy import save
from transformers import AutoTokenizer, XLMRobertaModel
import torch
from transformers import AutoTokenizer, AutoModel
import re

def remove_enclosed_parts(input_string):
    # Use a regular expression to remove all parts of the string enclosed by < and >
    result = re.sub(r'<[^>]*>', '', input_string)
    return result.strip()

if __name__ == "__main__":
    input_dir = sys.argv[1] # path to transcripts
    output_dir = sys.argv[2]

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = XLMRobertaModel.from_pretrained("xlm-roberta-base")

    all_sents = sorted([os.path.join(input_dir, elem) for elem in os.listdir(input_dir)])
    for sentences in tqdm(all_sents, desc='extract acoustic embeddings'):
        base_name = os.path.basename(sentences).split(".txt")[0]
        print(base_name)
        #sentences = open(sentences, 'r', encoding="utf-8",errors='ignore').read().strip().lower()
        with open(sentences, 'r', encoding="utf-8", errors='ignore') as file:
            sentences = file.read().strip()
            sentences = remove_enclosed_parts(sentences)
            if sentences != '':
                inputs = tokenizer(sentences, return_tensors="pt", truncation=True)
                outputs = model(**inputs)
                last_hidden_states = outputs.last_hidden_state #take last hidden state
                sentence_embedding = torch.mean(last_hidden_states, dim=1) # mean pooling
                sentence_embedding = sentence_embedding.detach().numpy()  # dim: (1, 768)
                print(sentence_embedding.shape)
                save(output_dir + base_name + '.npy', sentence_embedding)