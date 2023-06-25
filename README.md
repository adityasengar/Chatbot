# Simple Seq2Seq Chatbot using NLTK
This repository contains a simple chatbot implemented using Seq2Seq model in PyTorch. The chatbot uses a very basic sequence-to-sequence model architecture without attention mechanism. It's designed for educational purposes and not intended for production use.

Description
The chatbot implementation consists of two main files:

chatbot_model.py: This file defines the architecture of the Seq2Seq model, which consists of an encoder and a decoder.

chatbot_train.py: This file trains the Seq2Seq model on a set of question-answer pairs. It includes a tokenizer to preprocess the text, and handles punctuation and case normalization.

The chatbot is trained using a simple method where each question-answer pair is processed independently (i.e., the batch size is 1).

Requirements
Python 3.7 or later
PyTorch 1.7 or later
NLTK 3.5 or later
Usage
Clone this repository:
git clone <repository_url>

Navigate to the directory of the cloned repository:
cd <directory_path>

Run the chatbot_train.py script to train the model:
python chatbot_train.py

Once the model is trained, you can use it to generate responses to input questions. A simple runtime example can be performed as follows:

from chatbot_model import Seq2Seq
from chatbot_train import generate_response, word2idx, idx2word, device
model = Seq2Seq(input_dim=len(word2idx), output_dim=len(word2idx), emb_dim=256, hid_dim=512, n_layers=1, dropout=0.5).to(device)
model.load_state_dict(torch.load("model.pt"))
sample_question = "Hello, how are you?"
response = generate_response(model, sample_question)
print(response)


Please replace <repository_url> and <directory_path> with your actual repository URL and directory path.

Note
This is a very simplistic demonstration and the model won't work well in a real-world application without significant enhancements. For instance, this model doesn't handle out-of-vocabulary words, doesn't generate different responses to the same input, and doesn't maintain the context of the conversation. Advanced techniques such as attention mechanism, subword tokenization, or pre-trained embeddings are needed for a more effective chatbot.



