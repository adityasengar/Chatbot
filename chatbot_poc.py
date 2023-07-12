import torch
import torch.nn as nn
import torch.optim as optim
from nltk.tokenize import RegexpTokenizer
import nltk

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# ------------------------------------------------------------------
# 1. Data and Preprocessing
# ------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simple conversation dataset
conversations = [
    ("Hello", "Hi there!"),
    ("How are you doing?", "I'm doing great."),
    ("What's your name?", "I'm a bot."),
    ("What is your favorite color?", "Blue."),
]

# Tokenizer and Vocabulary
tokenizer = RegexpTokenizer(r'\\w+')
word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
idx2word = {v: k for k, v in word2idx.items()}

def tokenize_and_build_vocab(data):
    """Builds vocabulary from a list of sentences."""
    for sentence in data:
        tokens = tokenizer.tokenize(sentence.lower())
        for word in tokens:
            if word not in word2idx:
                new_idx = len(word2idx)
                word2idx[word] = new_idx
                idx2word[new_idx] = word

# Build vocabulary from our conversations
all_sentences = [s for conv in conversations for s in conv]
tokenize_and_build_vocab(all_sentences)

# ------------------------------------------------------------------
# 2. Model Definition
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
        
        input = trg[0, :]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs

# ------------------------------------------------------------------
# 3. Training and Generation Logic
# ------------------------------------------------------------------

def train(model, data, optimizer, criterion):
    """A single training epoch."""
    model.train()
    epoch_loss = 0
    for question, answer in data:
        optimizer.zero_grad()
        
        question_tokens = tokenizer.tokenize(question.lower())
        answer_tokens = tokenizer.tokenize(answer.lower())

        question_tensor = torch.tensor([word2idx.get(w, word2idx["<UNK>"]) for w in question_tokens]).unsqueeze(1).to(device)
        answer_tensor = torch.tensor([word2idx.get(w, word2idx["<UNK>"]) for w in answer_tokens]).unsqueeze(1).to(device)
        
        output = model(question_tensor, answer_tensor)
        
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        answer_tensor = answer_tensor[1:].view(-1)
        
        loss = criterion(output, answer_tensor)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(data)

def generate_response(model, question):
    """Generates a response to a given question."""
    model.eval()
    with torch.no_grad():
        question_tokens = tokenizer.tokenize(question.lower())
        question_tensor = torch.tensor([word2idx.get(w, word2idx["<UNK>"]) for w in question_tokens]).unsqueeze(1).to(device)
        
        # Create a dummy target tensor
        dummy_trg = torch.zeros_like(question_tensor)

        # Pass a zero tensor for target and set teacher_forcing_ratio to 0
        output = model(question_tensor, dummy_trg, teacher_forcing_ratio=0)
        
        predicted_indices = output.squeeze(1).argmax(dim=-1)
        predicted_words = [idx2word.get(idx.item(), "<UNK>") for idx in predicted_indices]
        
        return ' '.join(predicted_words).strip()

# ------------------------------------------------------------------
# 4. Main Execution
# ------------------------------------------------------------------
if __name__ == '__main__':
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
        loss = train(model, conversations, optimizer, criterion)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch+1:02} | Loss: {loss:.3f}')
    print("Training finished.")

    # Inference Example
    print("\n--- Chatbot is ready ---")
    sample_question = "How are you doing?"
    response = generate_response(model, sample_question)
    print(f"Q: {sample_question}")
    print(f"A: {response}")

    sample_question = "What is your name?"
    response = generate_response(model, sample_question)
    print(f"Q: {sample_question}")
    print(f"A: {response}")
