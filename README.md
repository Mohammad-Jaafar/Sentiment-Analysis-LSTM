# Sentiment Analysis with LSTM on IMDB Dataset

## Overview
This project demonstrates how to build, train, and evaluate a deep learning model for **sentiment analysis** on the [IMDB Movie Reviews Dataset](https://ai.stanford.edu/~amaas/data/sentiment/).  
The model uses **Long Short-Term Memory (LSTM)** networks to classify reviews as **positive** or **negative**.

---

## Features
- Preprocessing of raw text sequences (filtering & padding).
- Word embedding layer for text representation.
- Stacked **LSTM layers** for sequential learning.
- Dropout layers to reduce overfitting.
- Model saving (both `.keras` and `.h5` formats).
- Configurations & word index saved for inference.
- Training history visualization.

---

## Technologies Used
- **Python 3.9+**
- **TensorFlow / Keras**
- **NumPy**
- **Matplotlib**
- **Pickle**

---

## Project Structure
```
Sentiment-Analysis-LSTM
│── sentiment_analysis_lstm_imdb.ipynb
│── model_architecture.json
│── word_index.pkl
│── config.pkl
│── README.md
│── requirements.txt
```
---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Mohammad-Jaafar/Sentiment-Analysis-LSTM.git
   ```
2. Open the notebook in Jupyter or Google Colab.
3. Run all cells step-by-step to reproduce the results.

---

## Results
- Training Accuracy: ~**90%**
- Validation Accuracy: ~**85%**
- Test Accuracy: ~**85–88%** (depending on random initialization and hardware)

---

## Demo on HuggingFace Spaces
- **Sentiment Analysis LSTM**  
[HuggingFace](https://huggingface.co/spaces/Mhdjaafar/Sentiment-LSTM-Analyzer)

---

## Author
**Mohammad Jaafar**  
mhdjaafar24@gmail.com  
[LinkedIn](https://www.linkedin.com/in/mohammad-jaafar-)  
[HuggingFace](https://huggingface.co/Mhdjaafar)  
[GitHub](https://github.com/Mohammad-Jaafar)

---

## License
This project is licensed under the MIT License.  
Feel free to use, modify, and distribute.
