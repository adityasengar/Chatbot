import torch
import torch.nn as nn
import torch.optim as optim

from model import Encoder, Decoder, Seq2Seq
from utils import device, load_conversations, build_vocab, sentence_to_tensor, word2idx

# ------------------------------------------------------------------
# Training Logic
# ------------------------------------------------------------------

def train_epoch(model, data, optimizer, criterion):
    """A single training epoch."""
    model.train()
    epoch_loss = 0
    
    for question, answer in data:
        optimizer.zero_grad()
        
        question_tensor = sentence_to_tensor(question)
        answer_tensor = sentence_to_tensor(answer)
        
        output = model(question_tensor, answer_tensor)
        
        output_dim = output.shape[-1]
        
        # Trim off the <SOS> token from output and answer
        output = output[1:].view(-1, output_dim)
        answer_tensor = answer_tensor[1:].view(-1)
        
        loss = criterion(output, answer_tensor)
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
    VOCAB_SIZE = len(word2idx)
    EMBEDDING_DIM = 256
    HIDDEN_SIZE = 512
    NUM_LAYERS = 2
    DROPOUT = 0.5
    EPOCHS = 100

    # Model Instantiation
    encoder = Encoder(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
    decoder = Decoder(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, VOCAB_SIZE, NUM_LAYERS, DROPOUT)
    model = Seq2Seq(encoder, decoder).to(device)

    # Training
    criterion = nn.CrossEntropyLoss(ignore_index=word2idx["<PAD>"])
    optimizer = optim.Adam(model.parameters())

    print("Starting model training...")
    for epoch in range(EPOCHS):
        loss = train_epoch(model, conversations, optimizer, criterion)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch+1:02} | Loss: {loss:.3f}')
    print("Training finished.")

