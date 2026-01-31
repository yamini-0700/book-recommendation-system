# 📚 Book Recommendation System
## Final Year Project

A machine learning-based book recommendation system with an interactive reader analytics dashboard.

---

## ✨ Features

- **Hybrid Recommendation System**  
  Combines content-based filtering and collaborative filtering techniques.

- **Machine Learning Models**  
  Random Forest (85.3% accuracy), Logistic Regression, and Decision Tree models.

- **Reader Analytics Dashboard**  
  Provides insights into user behavior, preferences, and wishlist trends.

- **Web Interface**  
  Streamlit-based application for easy and interactive usage.

- **End-to-End Pipeline**  
  Data collection → preprocessing → model training → recommendation → analytics.

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/yamini-0700/book-recommendation-system.git
cd book-recommendation-system

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py


📊 ML Models Performance
Model	                Accuracy	Best For
Random Forest	        85.3%	    Overall prediction
Logistic Regression   	78.2%	    Baseline comparison
Decision Tree	        82.1%	    Interpretability


## 📁 Project Structure

Book_recommendation_system/
├── app.py # Main Streamlit application
├── train_recommendation_models.py # ML model training script
├── add_covers.py # Book cover generation
├── requirements.txt # Python dependencies
├── README.md # Project documentation
│
├── report/ # Project documentation files
│ ├── Project_Work_document FINAL.pdf # Project report
│ └── book recommendation.pptx # Project presentation
│
├── src/
│ ├── models/ # Recommendation algorithms
│ │ ├── collaborative.py
│ │ ├── content_based.py
│ │ ├── hybrid.py
│ │ └── recommender.py
│ ├── analytics/
│ │ └── wishlist_analytics.py
│ ├── data/
│ │ └── load_data.py
│ └── utils/
│
├── notebook/
│ ├── 01_data_overview.ipynb
│ ├── 02_data_cleaning.ipynb
│ ├── 03_eda_feature_engineering.ipynb
│ ├── 04_baseline_models.ipynb
│ ├── 05_model_optimization.ipynb
│ └── 06_reader_wishlist_analytics.ipynb
│
├── models/
│ ├── final_random_forest.pkl
│ ├── content_vectorizer.pkl
│ ├── scaler.pkl
│ └── sample_info.pkl
│
├── data/
│ ├── raw/
│ └── processed/
│
├── assets/
├── generated_covers/
└── venv/

🛠️ Technologies Used

Python 3.9+

Scikit-learn

Streamlit

Pandas & NumPy

Matplotlib & Seaborn

Jupyter Notebook

📄 Project Documentation

📘 Project Report: report/Project_Work_document FINAL.pdf

📊 Project Presentation: report/book recommendation.pptx

📌 Conclusion

This project demonstrates the effective use of machine learning techniques to build a personalized book recommendation system with reader analytics. The system is scalable, user-friendly, and suitable for real-world applications.