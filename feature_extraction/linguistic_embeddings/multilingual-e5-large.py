# conda activate mulitlingual_clip

from sentence_transformers import SentenceTransformer
import sys
import os
import re
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from numpy import save
from tqdm import tqdm

def remove_enclosed_parts(input_string):
    # Use a regular expression to remove all parts of the string enclosed by < and >
    result = re.sub(r'<[^>]*>', '', input_string)
    return result.strip()


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    input_dir = sys.argv[1] # path to transcripts
    output_dir = sys.argv[2]
    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
    model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')

    all_sents = sorted([os.path.join(input_dir, elem) for elem in os.listdir(input_dir)])
    for sentences in tqdm(all_sents, desc='extract acoustic embeddings'):
        base_name = os.path.basename(sentences).split(".txt")[0]
        print(base_name)
        with open(sentences, 'r', encoding="utf-8", errors='ignore') as file:
            sentences = file.read().strip()
            sentences = remove_enclosed_parts(sentences)
            if sentences != '':
                batch_dict = tokenizer(sentences, max_length=512, padding=True, truncation=True, return_tensors='pt')
                outputs = model(**batch_dict)
                embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                # normalize embeddings
                embeddings = F.normalize(embeddings, p=2, dim=1)

                embeddings = embeddings.detach().numpy()
                print(type(embeddings))
                print(embeddings.shape)
                save(output_dir + base_name + '.npy', embeddings)