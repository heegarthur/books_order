import math, random

# --- 1. Parameters ---
EMBED_SIZE = 16
HIDDEN_SIZE = 32
MAX_WORDS = 10
MAX_LEN = 3
LR = 0.01
EPOCHS = 200

# --- 2. Helpers ---
def word_to_ascii_vector(word):
    word = word[:MAX_LEN]
    vec = [ord(c)/255 for c in word] + [0]*(MAX_LEN - len(word))
    return vec

def words_to_matrix(words):
    words = words[:MAX_WORDS] + [""]*(MAX_WORDS - len(words))
    return [word_to_ascii_vector(w.lower()) for w in words]

def sorted_indices(words):
    return sorted(range(len(words)), key=lambda i: words[i].lower())

# --- 3. Data ---
TRAIN_DATA = [
    ["sp", "ak", "wi"],
    ["gs","sp","bi","ak","du"],
    ["sk","en","ne"],
    ["gs","du","ne","na"],
    ["ec","sk","wi"],
    ["ec","bi","na"],
    ["ec","bi","sp","du","sk","na","ak","ne","en","wi"],
    ["en","ec","wi"]
]

X_train = [words_to_matrix(lst) for lst in TRAIN_DATA]
y_train = [sorted_indices(lst) for lst in TRAIN_DATA]

# --- 4. Eenvoudig netwerk ---
def rand_matrix(rows, cols):
    return [[random.uniform(-0.5, 0.5) for _ in range(cols)] for _ in range(rows)]

def matmul(a, b):
    return [[sum(a[i][k]*b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]

def relu(v): return [[max(0, x) for x in row] for row in v]

def softmax(v):
    m = max(v)
    exps = [math.exp(x-m) for x in v]
    s = sum(exps)
    return [x/s for x in exps]

# --- 5. Initialisatie ---
W1 = rand_matrix(MAX_LEN, HIDDEN_SIZE)
W2 = rand_matrix(HIDDEN_SIZE, MAX_WORDS)

# --- 6. Training ---
for epoch in range(EPOCHS):
    total_loss = 0
    for x, y in zip(X_train, y_train):
        # Voor elk woord
        preds = []
        for word_vec in x:
            # Forward
            h = [sum(word_vec[i]*W1[i][j] for i in range(MAX_LEN)) for j in range(HIDDEN_SIZE)]
            h = [max(0, v) for v in h]
            out = [sum(h[i]*W2[i][j] for i in range(HIDDEN_SIZE)) for j in range(MAX_WORDS)]
            probs = softmax(out)
            preds.append(probs)
        # Simple loss = afstand tot one-hot targets
        for i, target_index in enumerate(y):
            if target_index < len(preds):
                target = [0]*MAX_WORDS
                target[target_index] = 1
                loss = sum((preds[i][j]-target[j])**2 for j in range(MAX_WORDS))
                total_loss += loss
                # Backprop (heel simpel)
                grad_out = [(preds[i][j]-target[j])*2 for j in range(MAX_WORDS)]
                for j in range(HIDDEN_SIZE):
                    for k in range(MAX_WORDS):
                        W2[j][k] -= LR * grad_out[k]
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# --- 7. Test ---
while True:
    user_input = input("\nVoer woorden in, gescheiden door spaties (of 'exit'): ")
    if user_input.lower() == "exit":
        break
    words = user_input.split()
    vecs = words_to_matrix(words)
    scores = []
    for word_vec in vecs:
        h = [sum(word_vec[i]*W1[i][j] for i in range(MAX_LEN)) for j in range(HIDDEN_SIZE)]
        h = [max(0, v) for v in h]
        out = [sum(h[i]*W2[i][j] for i in range(HIDDEN_SIZE)) for j in range(MAX_WORDS)]
        scores.append(sum(out))
    
    # sorteer enkel binnen het bereik van de echte woorden
    sorted_indices_pred = sorted(range(len(words)), key=lambda i: scores[i])
    
    print("Input:", words)
    print("Gesorteerd:", [words[i] for i in sorted_indices_pred])
