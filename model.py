import torch
import torch.nn as nn
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
# Model Definition
# ------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers, dropout=dropout)
        
    def forward(self, src):
        embedded = self.embedding(src)
        _, (hidden, cell) = self.rnn(embedded)
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
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.fc.out_features
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(device)
        hidden, cell = self.encoder(src)
        
        # First input to the decoder is the <SOS> token
        input = trg[0, :]
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            
            # Decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            
            # If teacher forcing, use actual next token as next input
            # If not, use predicted token
            input = trg[t] if teacher_force else top1
            
        return outputs
