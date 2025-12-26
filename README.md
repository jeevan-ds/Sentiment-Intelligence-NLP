<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/77a1460e-3e8c-4099-be3c-032130f64658" />

# Sentiment-Intelligence-NLP
End-to-end NLP project combining TF-IDF, machine learning, neural networks, and Bi-LSTM with GloVe to analyze review-level sentiment and visualize sentiment shifts within reviews.
# Sentiment Analysis & Sentiment Shift Detection

##  Overview
This project performs sentiment analysis at two levels:  
1) **Review-Level Sentiment Classification** and  
2) **Sentence-Level Sentiment Shift Analysis** to understand how sentiment evolves within a review.

The project demonstrates a complete NLP pipeline, progressing from traditional machine learning models using TF-IDF to deep learning models using word embeddings and Bi-LSTM.

---

##  Objectives
- Classify movie reviews as **positive or negative**
- Compare **TF-IDF–based ML models** with **neural network models**
- Analyze **sentence-wise sentiment progression** within a review
- Visualize **sentiment shifts across sentences**

---

##  Review-Level Sentiment Analysis
- Text preprocessing and feature extraction using **TF-IDF**
- Machine learning models: Logistic Regression, SVM, Naive Bayes, Random Forest, XGBoost
- Neural Network trained on TF-IDF features for non-linear learning
- Achieved strong performance for overall sentiment prediction

---

##  Sentence-Level Sentiment Shift Analysis (Core Contribution)
- Reviews split into individual sentences
- Sentences converted to semantic vectors using **GloVe word embeddings**
- **Bidirectional LSTM (Bi-LSTM)** used to capture contextual sentiment
- Sentence-level sentiment probabilities predicted using weak supervision
- Sentiment progression visualized across sentence order to detect shifts

---
<img width="768" height="393" alt="download" src="https://github.com/user-attachments/assets/5a55cdc2-821c-4011-bfa9-61e31634ca70" />

##  Sentiment Shift Visualization
- X-axis: Sentence index within a review  
- Y-axis: Positive sentiment probability  
- Enables detection of **negative → positive**, **positive → negative**, and **mixed sentiment** patterns

---

##  Tech Stack
- Python, Pandas, NumPy
- Scikit-learn
- TensorFlow / Keras
- Gensim (GloVe)
- NLTK
- Matplotlib, Seaborn

---

##  Conclusion
This project goes beyond traditional sentiment classification by combining word embeddings and Bi-LSTM models to analyze how sentiment changes within a review. It highlights the importance of contextual understanding in NLP and demonstrates a clear progression from classical machine learning to deep learning–based text analysis.
