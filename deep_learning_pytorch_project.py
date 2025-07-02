import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# Tokenizer
tokenizer = get_tokenizer('basic_english')

# Yield tokens
def yield_tokens(data_iter):
    for label, text in data_iter:
        yield tokenizer(text)

# Build vocab
train_iter = IMDB(split='train')
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<pad>"])
vocab.set_default_index(vocab["<pad>"])

# Encode text
def encode(text):
    return vocab(tokenizer(text))

# Collate function
def collate_batch(batch):
    label_map = {'neg': 0, 'pos': 1}
    texts, labels = [], []
    for label, text in batch:
        texts.append(torch.tensor(encode(text), dtype=torch.long))
        labels.append(torch.tensor(label_map[label], dtype=torch.long))
    texts = pad_sequence(texts, padding_value=vocab["<pad>"])
    return texts.T, torch.tensor(labels)

# Model definition
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab["<pad>"])
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden[-1])

# Load data
train_iter, test_iter = IMDB()
train_loader = DataLoader(list(train_iter), batch_size=32, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(list(test_iter), batch_size=32, collate_fn=collate_batch)

# Init model
model = LSTMClassifier(len(vocab), 64, 128, 2)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(3):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.2f}")

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predicted = torch.argmax(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {correct / total:.2f}")
