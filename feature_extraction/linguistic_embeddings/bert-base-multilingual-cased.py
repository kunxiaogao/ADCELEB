from sentence_transformers import SentenceTransformer
import sys
import os
import re
from transformers import BertTokenizer, BertModel
from numpy import save
from tqdm import tqdm

def remove_enclosed_parts(input_string):
    # Use a regular expression to remove all parts of the string enclosed by < and >
    result = re.sub(r'<[^>]*>', '', input_string)
    return result.strip()

if __name__ == "__main__":

    input_dir = sys.argv[1] # path to transcripts
    output_dir = sys.argv[2]

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertModel.from_pretrained("bert-base-multilingual-cased")
    all_sents = sorted([os.path.join(input_dir, elem) for elem in os.listdir(input_dir)])
    for sentences in tqdm(all_sents, desc='extract acoustic embeddings'):
        base_name = os.path.basename(sentences).split(".txt")[0]
        with open(sentences, 'r', encoding="utf-8", errors='ignore') as file:
            sentences = file.read().strip()
            sentences = remove_enclosed_parts(sentences)
            if sentences != '':
                print(sentences)
                encoded_input = tokenizer(sentences, return_tensors='pt', truncation=True)
                output = model(**encoded_input)
                embeddings = output['pooler_output'].detach().numpy()
                print(embeddings.shape)
                save(output_dir + base_name + '.npy', embeddings)
##