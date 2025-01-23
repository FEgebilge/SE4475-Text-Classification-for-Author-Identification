# Author Identification Using Fine-Tuned Turkish BERT  
**Multi-Class Text Classification for 30 Turkish Authors**  

This repository contains the implementation of a natural language processing (NLP) project that classifies **1,500 Turkish newspaper articles** into **30 distinct author classes** using a fine-tuned BERT model. The solution leverages **5-fold cross-validation** and achieves state-of-the-art performance metrics.  

---

## Key Features  
- **Multi-class classification** (30 authors, 50 articles per author).  
- **Transformer-based approach**: Fine-tuned `dbmdz/bert-base-turkish-cased` model.  
- **Advanced training strategies**:  
  - Dynamic learning rate with cosine decay and warmup.  
  - Label smoothing and gradient clipping for stability.  
- **Robust evaluation**:  
  - Stratified 5-fold cross-validation.  
  - Per-class and macro-averaged Precision, Recall, and F1-Score.  

---

## Project Structure  
```
├── data/ # Raw text files (30 authors, 50 articles each)
├── fold_metrics/ # Fold-wise performance metrics (CSV files)
│ └── dbmdz_bert-base-turkish-cased/
├── plots/ # Training/validation curves and confusion matrices
├── identification.ipynb # Jupyter notebook with full implementation
├── requirements.txt # Dependency list
└── README.md
```
---


---

## Methodology  

### 1. **Data Preprocessing**  
- **Tokenization**: Uses `AutoTokenizer` from Hugging Face with a max sequence length of **512**.  
- **Stratified Splitting**: Preserves class balance across 5 folds.  

### 2. **Model Architecture**  
- **Base Model**: `dbmdz/bert-base-turkish-cased` (pre-trained on Turkish text).  
- **Fine-tuning**: Added a classification layer (30 output neurons).  

### 3. **Training Configuration**  
- **Optimizer**: AdamW with weight decay (`initial=0.01`, `final=0.001`).  
- **Learning Rate**: Cosine schedule with warmup (`3e-5` → `1e-5`).  
- **Regularization**: Label smoothing (`smoothing=0.1`) and gradient clipping (`max_norm=1.0`).  
- **Early Stopping**: Patience of 3 epochs based on validation F1-score.  

### 4. **Evaluation**  
- **Metrics**: Macro-averaged Precision, Recall, and F1-score.  
- **Visualization**: Confusion matrices and loss/accuracy curves per fold.  

---

## Results  

### Average Performance (5-Fold Cross-Validation)  
| Metric    | Score  |  
|-----------|--------|  
| F1-Score  | 91.76% |  
| Precision | 92.74% |  
| Recall    | 91.93% |  

### Example Per-Class Metrics  
| Class   | Precision | Recall | F1-Score |  
|---------|-----------|--------|----------|  
| Class 1 | 91.11%    | 96.00% | 93.20%   |  
| Class 3 | 90.66%    | 92.00% | 91.11%   |  
| Class 30| 85.64%    | 64.00% | 72.02%   |  

📊 Full results: [CSV files](fold_metrics/dbmdz_bert-base-turkish-cased/).  

---

## Usage  

### Prerequisites  
- Python 3.9+  
- Libraries:  
  ```bash
  pip install -r requirements.txt


### Run the Code
1. Data Setup: Place articles in data/ with subdirectories for each author.
2. Training: Execute cells in identification.ipynb.
3. Results: Metrics saved to fold_metrics/ and plots to plots/.


## Acknowledgements

- Lecturer: Assoc. Prof. Dr. Mete Eminağaoğlu
- Course: SE4475 Introduction to NLP
- Tools: Hugging Face Transformers, PyTorch, scikit-learn