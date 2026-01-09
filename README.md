# Advanced PyTorch Chatbot

This project demonstrates the evolution of a chatbot, starting from a simple LSTM-based Seq2Seq model and progressing to a modern approach that involves fine-tuning a pre-trained GPT-2 model.

The commit history of this repository is designed to showcase a clear and logical development path, highlighting key advancements in NLP modeling techniques.

---

## Project Structure

-   `data/conversations.json`: The dataset used for training the models.
-   `utils.py`: Helper functions for loading data and preprocessing text.
-   `model.py`: Contains the from-scratch model implementations (LSTM Seq2Seq with Attention and Transformer).
-   `train.py`: The script used to train the from-scratch models in `model.py`.
-   `train_gpt.py`: The script used to fine-tune the pre-trained `distilgpt2` model.
-   `requirements.txt`: Project dependencies.

---

## Model Evolution

This project contains three distinct model architectures, each representing a step up in complexity and performance:

1.  **Seq2Seq with LSTM (Initial PoC)**: A simple sequence-to-sequence model that forms the baseline. *(See commits from July 2023)*
2.  **Seq2Seq with Attention**: An enhanced version of the first model that includes a Bahdanau-style attention mechanism to improve context handling. *(See commits from August 2023)*
3.  **Transformer**: A from-scratch implementation of a Transformer model, reflecting the modern standard for sequence-to-sequence tasks. *(See commits from September 2023)*
4.  **Fine-Tuned DistilGPT-2**: The most advanced and effective approach in this repository, which fine-tunes a large pre-trained language model from Hugging Face on the specific conversation dataset. *(See commits from October 2023)*

---

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/adityasengar/Chatbot.git
    cd Chatbot
    ```

2.  It is recommended to use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

You can choose which model to train and run.

### Option 1: Train the From-Scratch Models (Seq2Seq or Transformer)

The `train.py` script is used to train the models defined in `model.py`. The model code would need to be uncommented or selected within the script to choose between the Attention-based model and the Transformer.

```bash
python train.py
```

### Option 2: Fine-Tune and Run the GPT-2 Model (Recommended)

This is the most effective model in the repository.

1.  **Fine-Tune the Model:**
    Run the `train_gpt.py` script. This will download the `distilgpt2` model, fine-tune it on the `conversations.json` data, and save the resulting model to a directory named `./fine_tuned_gpt2`.

    ```bash
    python train_gpt.py
    ```

2.  **Interact with the Chatbot:**
    After training, the script will automatically drop into an example inference session. You can modify the script to create a standalone interactive chat application.

```
# Updated on 2026-01-09
