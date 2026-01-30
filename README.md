# 📚 Book Recommendation System
## Final Year Project

A machine learning-based book recommendation system with reader analytics dashboard.

## ✨ Features
- **Hybrid Recommendation**: Content-based + collaborative filtering
- **ML Models**: Random Forest (85.3% accuracy), Logistic Regression, Decision Trees
- **Analytics Dashboard**: User behavior analysis and visualization
- **Web Interface**: Streamlit application for easy interaction
- **Complete Pipeline**: Data processing → ML training → Deployment

## 🚀 Quick Start
```bash
# 1. Clone repository
git clone https://github.com/yamini-0700/book-recommendation-system.git
cd book-recommendation-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run application
streamlit run app.py

📊 ML Models Performance
Model	                    Accuracy	       Best For
Random Forest            	85.3%	        Overall prediction
Logistic Regression     	78.2%       	Baseline comparison
Decision Tree	            82.1%	        Interpretability

📁 Project Structure


Book_recommendation_system/
├── app.py                              # Main Streamlit application
├── train_recommendation_models.py      # ML model training script
├── add_covers.py                       # Book cover generation
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
│
├── src/                                # Source code modules
│   ├── models/                         # Recommendation algorithms
│   │   ├── collaborative.py            # Collaborative filtering
│   │   ├── content_based.py            # Content-based filtering
│   │   ├── hybrid.py                   # Hybrid model
│   │   └── recommender.py              # Main recommender class
│   ├── analytics/                      # Analytics functions
│   │   └── wishlist_analytics.py       # Wishlist analysis
│   ├── data/                           # Data utilities
│   │   └── load_data.py                # Data loading
│   └── utils/                          # Helper functions
│
├── notebook/                           # Jupyter notebooks
│   ├── 01_data_overview.ipynb          # Data exploration
│   ├── 02_data_cleaning.ipynb          # Data preprocessing
│   ├── 03_eda_feature_engineering.ipynb # Feature engineering
│   ├── 04_baseline_models.ipynb        # Initial ML models
│   ├── 05_model_optimization.ipynb     # Model tuning
│   └── 06_reader_wishlist_analytics.ipynb # Analytics
│
├── models/                             # Trained ML models
│   ├── final_random_forest.pkl         # Random Forest model (85.3% accuracy)
│   ├── content_vectorizer.pkl          # TF-IDF vectorizer
│   ├── scaler.pkl                      # Feature scaler
│   └── sample_info.pkl                 # Sample data
│
├── data/                               # Datasets
│   ├── raw/                            # Original data
│   └── processed/                      # Cleaned data
│
├── assets/                             # Static files
├── generated_covers/                   # Generated book covers
└── venv/                               # Virtual environment


🛠️ Technologies Used

Python 3.9+ - Core language
Scikit-learn - Machine learning
Streamlit - Web interface
Pandas/NumPy - Data processing
Matplotlib/Seaborn - Visualization
Jupyter - Data analysis
