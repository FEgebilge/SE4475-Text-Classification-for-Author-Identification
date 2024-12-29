# Multi-Class-Text-Classification-for-Author-Identification
 This repository contains the implementation of a natural language processing (NLP) final project that classifies 1500 Turkish newspaper articles into 30 author classes using advanced NLP techniques. The project includes preprocessing, feature extraction, model training, and evaluation with stratified 10-fold cross-validation.

---

## Features

- Multi-class classification task (30 classes).
- Dataset: 1500 Turkish newspaper articles (50 per author).
- Techniques: TF-IDF, n-grams, embeddings, and transformers.
- Evaluation: Precision, Recall, and F1-Score per class and overall average.

---

## Project Structure

- `data/`: Contains the dataset and preprocessing scripts.
- `reports/`: Performance and methodology reports.
- `outputs/`: Stores performance metrics and visualizations.

---

## Methodology

1. **Data Preprocessing:**
   - Tokenization, normalization, and stopword removal.
   - Lemmatization with Zemberek for Turkish language support.

2. **Feature Extraction:**
   - TF-IDF vectorization.
   - N-grams (unigrams, bigrams, trigrams).
   - Word embeddings (e.g., Word2Vec, FastText).
   - Pre-trained models (e.g., BERT).

3. **Models:**
   - Classical ML: SVM, Random Forest, k-NN.
   - Neural Networks and fine-tuned transformers.

4. **Evaluation:**
   - Stratified 10-fold cross-validation.
   - Metrics: Precision, Recall, and F1-Score.

---

## Results

| Class      | Precision | Recall | F1-Score |
|------------|-----------|--------|----------|
| Class 1    | 0.89      | 0.87   | 0.88     |
| Class 2    | 0.91      | 0.89   | 0.90     |
| ...        | ...       | ...    | ...      |
| Class 30   | 0.88      | 0.86   | 0.87     |
| **Average**| **0.90**  | **0.88** | **0.89** |

Detailed results can be found in the [Performance Report](reports/performance_report.pdf).

---

## Usage

### Prerequisites

- Python 3.9+
- Libraries: NumPy, pandas, scikit-learn, TensorFlow, PyTorch, NLTK, spaCy



### Acknowledgements

- Lecturer: Assoc. Prof. Dr. Mete Eminağaoğlu
- Course: SE4475 Introduction to NLP
- Tools: scikit-learn, TensorFlow, PyTorch, Hugging Face Transformers