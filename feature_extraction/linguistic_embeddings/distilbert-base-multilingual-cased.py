# conda activate mulitlingual_clip
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import sys
import os
import re
from numpy import save

def remove_enclosed_parts(input_string):
    # Use a regular expression to remove all parts of the string enclosed by < and >
    result = re.sub(r'<[^>]*>', '', input_string)
    return result.strip()

if __name__ == "__main__":

    input_dir = sys.argv[1] # path to transcripts
    output_dir = sys.argv[2]
    model = SentenceTransformer('distilbert/distilbert-base-multilingual-cased').to('cpu')

    all_sents = sorted([os.path.join(input_dir, elem) for elem in os.listdir(input_dir)])
    for sentences in tqdm(all_sents, desc='extract acoustic embeddings'):
        base_name = os.path.basename(sentences).split(".txt")[0]
        # sentences = open(sentences, 'r', encoding="utf-8",errors='ignore').read().strip().lower()
        with open(sentences, 'r', encoding="utf-8", errors='ignore') as file:
            sentences = file.read().strip()
            sentences = remove_enclosed_parts(sentences)
            if sentences != '':
                print(sentences)
                embeddings = model.encode(sentences)
                embeddings = embeddings.reshape(1, -1)
                print(type(embeddings))
                print(embeddings.shape)
                save(output_dir + base_name + '.npy', embeddings)
