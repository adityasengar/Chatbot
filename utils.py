import json
from nltk.tokenize import RegexpTokenizer
import torch

# ------------------------------------------------------------------
# Data and Preprocessing
# ------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizer and Vocabulary
tokenizer = RegexpTokenizer(r'\\w+')
word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
idx2word = {v: k for k, v in word2idx.items()}

def load_conversations(filepath):
    """Loads conversations from a JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return [(item['question'], item['answer']) for item in data]

def build_vocab(conversations):
    """Builds vocabulary from a list of conversations."""
    global word2idx, idx2word
    all_sentences = [s for conv in conversations for s in conv]
    for sentence in all_sentences:
        tokens = tokenizer.tokenize(sentence.lower())
        for word in tokens:
            if word not in word2idx:
                new_idx = len(word2idx)
                word2idx[word] = new_idx
                idx2word[new_idx] = word
    return word2idx, idx2word

def sentence_to_tensor(sentence):
    """Converts a sentence string to a tensor of indices."""
    tokens = tokenizer.tokenize(sentence.lower())
    indices = [word2idx.get(w, word2idx["<UNK>"]) for w in tokens]
    return torch.tensor(indices).unsqueeze(1).to(device)
