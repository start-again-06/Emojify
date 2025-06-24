 😄 Emojify: Text-to-Emoji Classifier

This repository demonstrates how to build two emoji prediction models:
1. A simple softmax classifier based on word vector averages
2. An LSTM-based deep network using pre-trained GloVe embeddings

---

## 🧠 Models Overview

### 🔸 Softmax Classifier (Model 1)
- Converts sentences to averaged GloVe word vectors
- Uses a NumPy-implemented softmax layer for classification
- Lightweight and interpretable

### 🔸 LSTM Model (Model 2)
- Builds a Keras model with a pre-trained embedding layer
- Uses LSTM → Dropout → Dense → Softmax pipeline
- Captures contextual dependencies between words

---

## 🧾 Dataset and Embeddings
- Training: `train_emoji.csv`
- Testing: `tesss.csv`
- Word Vectors: `glove.6B.50d.txt`
- Five emoji classes:
  - ❤️ Love
  - ⚽ Sports
  - 😄 Happy
  - 😞 Sad
  - 🍴 Food

---

## 📦 Key Components

### 🧮 Data Utilities
- `read_csv()` – Load training and test CSV files
- `convert_to_one_hot()` – One-hot encodes label indices
- `read_glove_vecs()` – Load and process GloVe vectors

### ✍️ Feature Extraction
- `sentence_to_avg()` – Convert a sentence to averaged word vector
- `sentences_to_indices()` – Convert sentences to index arrays (for LSTM input)

### 🧠 Models
- `model()` – NumPy-based softmax classifier using averaged vectors
- `Emojify_V2()` – Keras LSTM model with pre-trained GloVe embeddings

### 🧪 Evaluation
- `predict()` – Predict labels using a trained model
- `print_predictions()` – Display predictions with emojis
- `plot_confusion_matrix()` – Visualize test accuracy

---

## ⚙️ File Structure
```bash
📁 data/
├── train_emoji.csv
├── tesss.csv
├── glove.6B.50d.txt

📁 utils/
├── emo_utils.py         # Label/emoji mapping, predictions
├── test_utils.py        # Custom tests for validation
```

---

## 🧪 Sample Results

```text
Sentence: I love you → ❤️
Sentence: Lets play football → ⚽
Sentence: Food is ready → 🍴
Sentence: Not feeling happy → 😞
```

---

## 📈 Training Example
```python
model = Emojify_V2((maxLen,), word_to_vec_map, word_to_index)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_indices, Y_train_oh, epochs=50, batch_size=32)
```

---

## 🧠 Embedding Layer
- Constructed from GloVe vectors
- Non-trainable to preserve semantic structure
- Input: sequence of word indices
- Output: sentence embeddings used by LSTM

---

## 📚 References
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)


