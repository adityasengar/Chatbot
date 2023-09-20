import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

from model import TransformerModel
from utils import device, load_conversations, build_vocab, word2idx, idx2word, tokenizer

# ------------------------------------------------------------------
# Transformer-specific helpers
# ------------------------------------------------------------------
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt, pad_idx):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = (src == pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# ------------------------------------------------------------------
# Training Logic
# ------------------------------------------------------------------
def train_epoch(model, data, optimizer, criterion, pad_idx):
    model.train()
    epoch_loss = 0
    
    # Simple batching (batch size = 1)
    for question, answer in data:
        optimizer.zero_grad()
        
        # Prepare Tensors
        src = torch.tensor([word2idx.get(w, word2idx["<UNK>"]) for w in tokenizer.tokenize(question.lower())]).to(device)
        tgt = torch.tensor([word2idx.get(w, word2idx["<UNK>"]) for w in tokenizer.tokenize(answer.lower())]).to(device)
        
        # Add <SOS> and <EOS> tokens
        src = torch.cat((torch.tensor([word2idx["<SOS>"]]), src, torch.tensor([word2idx["<EOS>"]]))).unsqueeze(1)
        tgt = torch.cat((torch.tensor([word2idx["<SOS>"]]), tgt, torch.tensor([word2idx["<EOS>"]]))).unsqueeze(1)
        
        tgt_input = tgt[:-1, :]
        
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, pad_idx)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        
        tgt_out = tgt[1:, :]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(data)

# ------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------
if __name__ == '__main__':
    # Load data and build vocabulary
    conversations = load_conversations('data/conversations.json')
    word2idx, idx2word = build_vocab(conversations)

    # Hyperparameters
    SRC_VOCAB_SIZE = len(word2idx)
    TGT_VOCAB_SIZE = len(word2idx)
    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 512
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    EPOCHS = 18

    # Model Instantiation
    model = TransformerModel(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD,
                             SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM).to(device)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # Training
    criterion = nn.CrossEntropyLoss(ignore_index=word2idx["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    print("Starting Transformer model training...")
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, conversations, optimizer, criterion, word2idx["<PAD>"])
        print(f'Epoch: {epoch:02} | Loss: {loss:.3f}')
    print("Training finished.")
