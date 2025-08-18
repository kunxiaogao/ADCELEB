from transformers import AutoTokenizer, AutoModel
import sys
import os
import torch
from numpy import save


# Function to get sentence embeddings from LLaMA
def get_sentence_embedding(sentence, model, tokenizer):
    # Tokenize the input sentence
    encoded_input = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=512)

    # Move the input to the same device as the model (CUDA if available)
    encoded_input = {key: value.to(model.device) for key, value in encoded_input.items()}

    # Get the model's output
    with torch.no_grad():
        outputs = model(**encoded_input)

    # Extract the last hidden state (we'll use the mean over the sequence for sentence embedding)
    last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)
    sentence_embedding = last_hidden_state.mean(dim=1)  # Average over the tokens

    # Return the embedding as a NumPy array
    return sentence_embedding.detach().cpu().numpy()


if __name__ == "__main__":

    input_dir = sys.argv[1]  # Path to transcripts
    output_dir = sys.argv[2]
    hf_token = ''
    # Load LLaMA tokenizer and model
    model_name = "meta-llama/Meta-Llama-3-8B"  # Replace with the desired LLaMA model
    tokenizer = AutoTokenizer.from_pretrained(model_name,use_auth_token=hf_token)
    hf_token = ''
    # Add a padding token if it's missing
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = AutoModel.from_pretrained(model_name,use_auth_token=hf_token)

    # Ensure the model is in evaluation mode and on the correct device
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_sents = sorted([os.path.join(input_dir, elem) for elem in os.listdir(input_dir)])
    for sentence_file in all_sents:
        base_name = os.path.basename(sentence_file).split(".txt")[0]
        print(base_name)

        # Read the sentence from the file
        with open(sentence_file, 'r', encoding="utf-8", errors='ignore') as file:
            sentences = file.read().strip()

        # Get sentence embedding from LLaMA
        embedding = get_sentence_embedding(sentences, model, tokenizer)
        print(type(embedding))
        print(embedding.shape)

        # Save the embeddings as a .npy file
        save(os.path.join(output_dir, base_name + '.npy'), embedding)