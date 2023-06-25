!pip install nltk

import torch
import torch.nn as nn
import torch.optim as optim
from nltk.tokenize import word_tokenize, RegexpTokenizer
from collections import defaultdict
import nltk
nltk.download('punkt')

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenize and build vocabulary
tokenizer = RegexpTokenizer(r'\w+')  # this will effectively remove punctuation by only keeping word characters
word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}

conversations = [
    ("Hello", "Hi there!"),
    ("How are you doing?", "I'm doing great."),
    ("What's your name?", "I'm a bot."),
    ("What is your favorite color?", "Blue."),
]

def tokenize_and_build_vocab(data, word2idx, idx2word):
    for sentence in data:
        tokens = tokenizer.tokenize(sentence.lower())
        for word in tokens:
            if word not in word2idx:
                idx2word[len(word2idx)] = word
                word2idx[word] = len(word2idx)
    return word2idx, idx2word

# Tokenize and build vocabulary from our conversations dataset
for conv in conversations:
    question, answer = conv
    word2idx, idx2word = tokenize_and_build_vocab([question, answer], word2idx, idx2word)

vocab_size = len(word2idx)
embedding_dim = 256
hidden_size = 512
num_layers = 2
dropout = 0.5

# Seq2Seq Model
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers, dropout=dropout)
        
    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, output_size, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, src, hidden, cell):
        src = src.unsqueeze(0)
        embedded = self.embedding(src)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc(output.squeeze(0))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, src, trg):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.fc.out_features
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(device)
        hidden, cell = self.encoder(src)
        input = trg[0, :]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            top1 = output.argmax(1)
            input = top1
        return outputs

# Instantiate the encoder and decoder, and define the final Seq2Seq model
encoder = Encoder(vocab_size, embedding_dim, hidden_size, num_layers, dropout).to(device)
decoder = Decoder(vocab_size, embedding_dim, hidden_size, vocab_size, num_layers, dropout).to(device)
model = Seq2Seq(encoder, decoder).to(device)
