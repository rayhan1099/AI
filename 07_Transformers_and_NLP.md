# Transformers and Natural Language Processing

## 📖 Table of Contents
1. [NLP Fundamentals](#nlp-fundamentals)
2. [Text Preprocessing](#text-preprocessing)
3. [Word Embeddings](#word-embeddings)
4. [Transformer Architecture](#transformer-architecture)
5. [Hugging Face Transformers](#hugging-face-transformers)
6. [BERT and GPT](#bert-and-gpt)
7. [Fine-tuning Pre-trained Models](#fine-tuning-pre-trained-models)
8. [Complete Examples](#complete-examples)

---

## NLP Fundamentals

### What is NLP?
- **Natural Language Processing**: Teaching computers to understand human language
- **Applications**: Sentiment analysis, machine translation, chatbots, text generation

### Key Challenges
- **Ambiguity**: Words can have multiple meanings
- **Context**: Meaning depends on context
- **Variability**: Same meaning, different words
- **Structure**: Grammar, syntax, semantics

---

## Text Preprocessing

### Basic Preprocessing

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    # Or Lemmatization (better than stemming)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)
```

### Using spaCy (More Efficient)

```python
import spacy

# Load model
nlp = spacy.load('en_core_web_sm')

def preprocess_spacy(text):
    doc = nlp(text)
    
    # Extract tokens, remove stopwords and punctuation
    tokens = [
        token.lemma_.lower() 
        for token in doc 
        if not token.is_stop and not token.is_punct
    ]
    
    return ' '.join(tokens)
```

---

## Word Embeddings

### Word2Vec

```python
from gensim.models import Word2Vec

# Prepare sentences (list of tokenized sentences)
sentences = [['hello', 'world'], ['machine', 'learning']]

# Train Word2Vec
model = Word2Vec(
    sentences,
    vector_size=100,      # Embedding dimension
    window=5,            # Context window
    min_count=1,         # Minimum word count
    workers=4
)

# Get word vector
vector = model.wv['hello']

# Find similar words
similar = model.wv.most_similar('hello', topn=5)
```

### GloVe (Global Vectors)

```python
# Download pre-trained GloVe embeddings
# Load GloVe vectors
def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

glove_embeddings = load_glove_embeddings('glove.6B.100d.txt')
```

### FastText

```python
from gensim.models import FastText

# Train FastText
model = FastText(
    sentences,
    vector_size=100,
    window=5,
    min_count=1
)

# FastText can handle out-of-vocabulary words
vector = model.wv['unknownword']
```

---

## Transformer Architecture

### Key Concepts

1. **Self-Attention**: Model relationships between all words
2. **Multi-Head Attention**: Multiple attention mechanisms
3. **Position Encoding**: Adds position information
4. **Encoder-Decoder**: For sequence-to-sequence tasks

### Self-Attention Mechanism

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.projection_dim = embed_dim // num_heads
        
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)
    
    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights
    
    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output
```

### Transformer Block

```python
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
    
    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
```

---

## Hugging Face Transformers

### Installation
```bash
pip install transformers torch
```

### Using Pre-trained Models

```python
from transformers import (
    AutoTokenizer, AutoModel,
    BertTokenizer, BertModel,
    GPT2Tokenizer, GPT2LMHeadModel,
    T5Tokenizer, T5ForConditionalGeneration
)

# Auto classes (automatically load correct model)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Or specific models
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
```

### Text Classification with BERT

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2  # Binary classification
)

# Tokenize text
text = "I love this product!"
encoded = tokenizer(
    text,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors='pt'
)

# Predict
with torch.no_grad():
    outputs = model(**encoded)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(predictions, dim=-1)
```

### Text Generation with GPT

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set pad token
tokenizer.pad_token = tokenizer.eos_token

# Generate text
prompt = "The future of AI is"
inputs = tokenizer.encode(prompt, return_tensors='pt')

with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_length=100,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

---

## BERT and GPT

### BERT (Bidirectional Encoder Representations from Transformers)

**Key Features:**
- **Bidirectional**: Reads text left-to-right and right-to-left
- **Masked Language Model**: Predicts masked words
- **Next Sentence Prediction**: Understands sentence relationships

**Use Cases:**
- Text classification
- Named Entity Recognition (NER)
- Question Answering
- Sentiment Analysis

```python
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Get embeddings
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
outputs = model(**inputs)

# Use [CLS] token embedding for classification
cls_embedding = outputs.last_hidden_state[:, 0, :]

# Or use all token embeddings
all_embeddings = outputs.last_hidden_state
```

### GPT (Generative Pre-trained Transformer)

**Key Features:**
- **Autoregressive**: Generates text one token at a time
- **Unidirectional**: Reads text left-to-right only
- **Large scale**: Trained on massive datasets

**Use Cases:**
- Text generation
- Language modeling
- Chatbots
- Creative writing

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Generate text
prompt = "Once upon a time"
inputs = tokenizer.encode(prompt, return_tensors='pt')

outputs = model.generate(
    inputs,
    max_length=150,
    num_return_sequences=3,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    do_sample=True
)

for output in outputs:
    print(tokenizer.decode(output, skip_special_tokens=True))
```

---

## Fine-tuning Pre-trained Models

### Fine-tuning BERT for Classification

```python
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    Trainer, TrainingArguments
)
from datasets import Dataset
import torch

# Load model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)

# Prepare dataset
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=128
    )

dataset = Dataset.from_dict({
    'text': texts,
    'label': labels
})
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
)

# Train
trainer.train()
```

### Fine-tuning with PyTorch

```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Create dataloader
train_dataset = TextDataset(train_texts, train_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Model and optimizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
model.train()
for epoch in range(3):
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
```

---

## Complete Examples

### Example 1: Sentiment Analysis

```python
from transformers import pipeline

# Use pre-built pipeline
classifier = pipeline('sentiment-analysis')

result = classifier("I love this movie!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]

# Custom model
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

inputs = tokenizer("This is amazing!", return_tensors='pt')
outputs = model(**inputs)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
```

### Example 2: Named Entity Recognition

```python
from transformers import pipeline

# NER pipeline
ner = pipeline('ner', aggregation_strategy='simple')

text = "Apple is looking at buying U.K. startup for $1 billion"
entities = ner(text)

for entity in entities:
    print(f"{entity['word']}: {entity['entity_group']}")
```

### Example 3: Question Answering

```python
from transformers import pipeline

qa = pipeline('question-answering')

context = """
Machine learning is a method of data analysis that automates analytical model building.
It is a branch of artificial intelligence based on the idea that systems can learn from data.
"""

question = "What is machine learning?"
answer = qa(question=question, context=context)
print(answer)
```

### Example 4: Text Summarization

```python
from transformers import pipeline

summarizer = pipeline('summarization')

text = """
Machine learning is a method of data analysis that automates analytical model building.
It is a branch of artificial intelligence based on the idea that systems can learn from data,
identify patterns and make decisions with minimal human intervention.
"""

summary = summarizer(text, max_length=50, min_length=30, do_sample=False)
print(summary[0]['summary_text'])
```

---

## Key Takeaways

1. **Preprocessing**: Clean and normalize text data
2. **Embeddings**: Use pre-trained word embeddings
3. **Transformers**: State-of-the-art for NLP tasks
4. **Hugging Face**: Easy access to pre-trained models
5. **Fine-tuning**: Adapt pre-trained models to your task

---

## Next Steps

- **[08_Computer_Vision.md](08_Computer_Vision.md)** - Computer Vision techniques
- **[09_MLOps_and_Deployment.md](09_MLOps_and_Deployment.md)** - Deploy NLP models

---

**Master transformers to build cutting-edge NLP applications!**

