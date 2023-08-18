import torch
import torch.nn as nn
import torch.nn.functional as F
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
# Model Definition with Attention
# ------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, num_layers, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self, src):
        # src = [src len, batch size]
        embedded = self.embedding(src)
        # embedded = [src len, batch size, emb dim]
        
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [src len, batch size, hid dim * num directions]
        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        # Pass the concatenated final forward and backward hidden states through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        
        return outputs, hidden.unsqueeze(0)


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear((hidden_size * 2) + hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden = [1, batch size, hid dim]
        # encoder_outputs = [src len, batch size, hid dim * 2]
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        # Repeat decoder hidden state src_len times
        hidden = hidden.repeat(src_len, 1, 1)
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy = [src len, batch size, hid dim]

        attention = self.v(energy).squeeze(2)
        # attention = [src len, batch size]
        
        return F.softmax(attention, dim=0)


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, output_size, num_layers, dropout, attention):
        super().__init__()
        self.attention = attention
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.rnn = nn.LSTM((hidden_size * 2) + embedding_dim, hidden_size, num_layers, dropout=dropout)
        self.fc = nn.Linear((hidden_size * 2) + hidden_size + embedding_dim, output_size)
        
    def forward(self, src, hidden, encoder_outputs, cell_state): # Added cell_state
        src = src.unsqueeze(0)
        # src = [1, batch size]
        
        embedded = self.embedding(src)
        # embedded = [1, batch size, emb dim]
        
        a = self.attention(hidden, encoder_outputs).unsqueeze(1)
        # a = [src len, batch size, 1]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs = [batch size, src len, hid dim * 2]
        
        weighted = torch.bmm(a.permute(1, 2, 0), encoder_outputs).permute(1, 0, 2)
        # weighted = [1, batch size, hid dim * 2]
        
        rnn_input = torch.cat((embedded, weighted), dim=2)
        # rnn_input = [1, batch size, (hid dim * 2) + emb dim]
        
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell_state)) # Use cell_state
        
        prediction = self.fc(torch.cat((output.squeeze(0), weighted.squeeze(0), embedded.squeeze(0)), dim=1))
        
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
        
        encoder_outputs, hidden = self.encoder(src)
        
        # First input to the decoder is the <SOS> token
        input = trg[0, :]
        
        # The first cell state for the decoder is the final cell state of the encoder
        # Since the encoder is bidirectional, we need to handle this. For simplicity, we'll initialize it to zeros.
        cell = torch.zeros(hidden.shape).to(device)
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, encoder_outputs, cell)
            outputs[t] = output
            
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            
            input = trg[t] if teacher_force else top1
            
        return outputs