import torch
import torch.nn as nn
import torch.optim as optim

EMBED_SIZE = 16    
HIDDEN_SIZE = 32   
MAX_WORDS = 10     
MAX_LEN = 3       

def word_to_ascii_vector(word):
    word = word[:MAX_LEN]
    vec = [ord(c) for c in word] + [0]*(MAX_LEN - len(word))
    return vec

def words_to_tensor(words):
    words = words[:MAX_WORDS] + [""]*(MAX_WORDS - len(words))
    return torch.tensor([word_to_ascii_vector(w.lower()) for w in words], dtype=torch.float)

def decode_indices(indices, words):
    sorted_words = []
    for i in indices:
        if i < len(words):
            sorted_words.append(words[i])
    return sorted_words

TRAIN_DATA = [
    ["sp", "ak", "wi"],
    ["gs","sp","bi","ak","du"],
    ["sk","en","ne"],
    ["gs","du","ne","na"],
    ["ec","sk","wi"],
    ["ec","bi","na"],
    ["ec","bi", "sp","du", "sk","na","ak","ne","en","wi"],
    ["en","ec", "wi"]
]

X_train = [words_to_tensor(lst) for lst in TRAIN_DATA]
y_train = [torch.tensor(sorted(range(len(lst)), key=lambda i: lst[i].lower()), dtype=torch.long) for lst in TRAIN_DATA]

class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size):
        super().__init__()
        self.embed = nn.Linear(input_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)
    def forward(self, x):
        x = self.embed(x)
        out, (h,c) = self.rnn(x)
        return out, (h,c)

class PointerDecoder(nn.Module):
    def __init__(self, hidden_size, seq_len):
        super().__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.vt = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, encoder_outputs):
        batch_size, seq_len, hidden_size = encoder_outputs.size()
        h = torch.zeros(1, batch_size, hidden_size)
        c = torch.zeros(1, batch_size, hidden_size)
        pointers = []
        mask = torch.zeros(batch_size, seq_len)
        
        for _ in range(seq_len):
            out, (h,c) = self.rnn(h, (h,c))
            e = self.vt(torch.tanh(self.W1(encoder_outputs) + self.W2(out).repeat(1,seq_len,1))).squeeze(-1)
            e = e - mask*1e6
            alpha = torch.softmax(e, dim=1)
            pointers.append(alpha)
            mask = mask + (alpha>0.5).float()
        pointers = torch.stack(pointers, dim=1)
        return pointers

input_size = MAX_LEN
seq_len = MAX_WORDS
encoder = Encoder(input_size, EMBED_SIZE, HIDDEN_SIZE)
decoder = PointerDecoder(HIDDEN_SIZE, seq_len)

params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(params, lr=0.01)
criterion = nn.CrossEntropyLoss()

EPOCHS = 200  
for epoch in range(EPOCHS):
    total_loss = 0
    for x, y in zip(X_train, y_train):
        x = x.unsqueeze(0)
        optimizer.zero_grad()
        enc_out, _ = encoder(x)
        pointers = decoder(enc_out)
        loss = sum(criterion(pointers[0,i].unsqueeze(0), y[i].unsqueeze(0)) for i in range(len(y)))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

while True:
    user_input = input("\nVoer woorden in, gescheiden door spaties (of 'exit' om te stoppen): ")
    if user_input.lower() == "exit":
        break
    words = user_input.split()
    x_test = words_to_tensor(words).unsqueeze(0)
    with torch.no_grad():
        enc_out,_ = encoder(x_test)
        pointers = decoder(enc_out)
        pred_indices = pointers.argmax(dim=2).squeeze().tolist()
        sorted_words = decode_indices(pred_indices, words)
    print("Input:", words)
    print("Network gesorteerd:", sorted_words)
