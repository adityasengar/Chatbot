# Define the loss function, optimizer, and the number of epochs
criterion = nn.CrossEntropyLoss(ignore_index = word2idx["<PAD>"])
optimizer = optim.Adam(model.parameters())
epochs = 10

def train(model, data, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for question, answer in data:
        optimizer.zero_grad()
        
        # Tokenize and convert to tensor
        question_tensor = torch.tensor([word2idx[word] for word in tokenizer.tokenize(question.lower())]).unsqueeze(1).to(device)
        answer_tensor = torch.tensor([word2idx[word] for word in tokenizer.tokenize(answer.lower())]).unsqueeze(1).to(device)
        
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
    model.eval()
    with torch.no_grad():
        question_tensor = torch.tensor([word2idx[word] for word in tokenizer.tokenize(question.lower())]).unsqueeze(1).to(device)
        outputs = model(question_tensor, torch.zeros_like(question_tensor))
        predicted_indices = outputs.squeeze(1).argmax(1)
        predicted_words = [idx2word[idx.item()] for idx in predicted_indices]
        return ' '.join(predicted_words)



