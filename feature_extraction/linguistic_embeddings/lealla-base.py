# conda activate lealla
import tensorflow_hub as hub
import tensorflow as tf
from tqdm import tqdm
import re
import os
import sys
from numpy import save

def remove_enclosed_parts(input_string):
    # Use a regular expression to remove all parts of the string enclosed by < and >
    result = re.sub(r'<[^>]*>', '', input_string)
    return result.strip()


if __name__ == "__main__":
    #https://www.kaggle.com/models/google/lealla/frameworks/TensorFlow2/variations/lealla-base/versions/1
    encoder = hub.KerasLayer("https://www.kaggle.com/models/google/lealla/TensorFlow2/lealla-base/1")
    input_dir = sys.argv[1] # path to transcripts
    output_dir = sys.argv[2]

    all_sents = sorted([os.path.join(input_dir, elem) for elem in os.listdir(input_dir)])
    for sentences in tqdm(all_sents, desc='extract acoustic embeddings'):
        base_name = os.path.basename(sentences).split(".txt")[0]
        print(base_name)
        # sentences = open(sentences, 'r', encoding="utf-8",errors='ignore').read().strip().lower()
        with open(sentences, 'r', encoding="utf-8", errors='ignore') as file:
            sentences = file.read().strip()
            sentences = remove_enclosed_parts(sentences)
            if sentences != '':
                sentences = tf.constant([sentences])
                embeds = encoder(sentences)
                embeds = embeds.numpy()
                save(output_dir + base_name + '.npy', embeds)







