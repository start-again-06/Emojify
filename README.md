Emojify: Text-to-Emoji Classifier  

A comprehensive NLP project demonstrating two emoji prediction models: a simple softmax classifier based on averaged word vectors and an LSTM-based deep network leveraging pre-trained GloVe embeddings. The project focuses on converting text into emojis by capturing semantic and contextual information.

## Models Overview

### Softmax Classifier (Model 1)
- Converts sentences to averaged GloVe word vectors  
- Implements a NumPy-based softmax layer for classification  
- Lightweight, interpretable model  

### LSTM Model (Model 2)
- Keras model with a pre-trained embedding layer  
- Pipeline: LSTM → Dropout → Dense → Softmax  
- Captures contextual dependencies between words  

## Dataset and Embeddings
- Training: `train_emoji.csv`  
- Testing: `tesss.csv`  
- Word Vectors: `glove.6B.50d.txt`  
- Emoji Classes:  
  ❤️ Love  
  ⚽ Sports  
  😄 Happy  
  😞 Sad  
  🍴 Food  

## Key Components

### Data Utilities
- `read_csv()` – Load training and test CSV files  
- `convert_to_one_hot()` – One-hot encodes label indices  
- `read_glove_vecs()` – Load and process GloVe vectors  

### Feature Extraction
- `sentence_to_avg()` – Convert a sentence to averaged word vector  
- `sentences_to_indices()` – Convert sentences to index arrays (for LSTM input)  

### Models
- `model()` – NumPy-based softmax classifier using averaged vectors  
- `Emojify_V2()` – Keras LSTM model with pre-trained GloVe embeddings  

### Evaluation
- `predict()` – Predict labels using a trained model  
- `print_predictions()` – Display predictions with emojis  
- `plot_confusion_matrix()` – Visualize test accuracy 


## Sample Results
- Sentence: "I love you" → ❤️  
- Sentence: "Lets play football" → ⚽  
- Sentence: "Food is ready" → 🍴  
- Sentence: "Not feeling happy" → 😞  

## Training Example

# Initialize the LSTM-based Emojify model with pre-trained embeddings

Shape of input sequences (max sentence length)
input_shape = (maxLen,)

Pre-trained GloVe embeddings
word_to_vec_map_param = word_to_vec_map

Mapping from words to indices
word_to_index_param = word_to_index

model = Emojify_V2(
    input_shape=input_shape,
    word_to_vec_map=word_to_vec_map_param,
    word_to_index=word_to_index_param
)

# Compile the LSTM-based Emojify model

#Loss function for multi-class classification
loss_function = 'categorical_crossentropy'

#Optimizer for training
optimizer_method = 'adam'

#Evaluation metric
evaluation_metrics = ['accuracy']

model.compile(
    loss=loss_function,
    optimizer=optimizer_method,
    metrics=evaluation_metrics
)

# Train the model

Input training data (sentence indices)
X_train_data = X_train_indices

One-hot encoded labels
Y_train_data = Y_train_oh

Number of training epochs
num_epochs = 50

Mini-batch size
batch_size = 32

model.fit(
    x=X_train_data,
    y=Y_train_data,
    epochs=num_epochs,
    batch_size=batch_size
)

## Embedding Layer
- Constructed from GloVe vectors  
- Non-trainable to preserve semantic structure  
- Input: sequence of word indices  
- Output: sentence embeddings used by LSTM  

## References
- GloVe: Global Vectors for Word Representation  

## License
This project is intended for educational and research purposes.  
Free to use and modify with proper attribution.
