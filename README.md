# 😴 Sleep Disorder Predictor


https://github.com/user-attachments/assets/a565c304-8f33-4ba1-b1c1-59ac6d7c6adb


![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A machine learning web application that predicts sleep disorders based on lifestyle and health factors using Random Forest classification.

## 🚀 Features

- **Health Assessment**: Comprehensive input form for sleep patterns, lifestyle factors, and health metrics
- **ML Predictions**: Real-time sleep disorder classification (None, Insomnia, Sleep Apnea)
- **Interactive Dashboard**: Visualizations and health metrics analysis
- **Personalized Recommendations**: Actionable insights based on prediction results
- **Model Interpretation**: Feature importance and SHAP analysis
 
## 📊 Model Performance

- **Algorithm**: Random Forest Classifier
- **Accuracy**: 89%+
- **F1-Score**: 0.8936
- **Classes**: None, Insomnia, Sleep Apnea

## 🛠️ Installation
 sleep-disorder-predictor/
├── sleep_health_app.py          # Main Streamlit application
├── create_demo_model.py         # Demo model creation script
├── requirements.txt             # Python dependencies
├── best_sleep_disorder_model.pkl # Trained ML model (generated)
├── model_interpretation_results.pkl # Model analysis results
├── sleep_health_processed.csv   # Processed dataset
├── README.md                    # Project documentation
├── .gitignore                   # Git ignore rules
└── LICENSE                      # MIT License
### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/your-username/sleep-disorder-predictor.git
cd sleep-disorder-predictor
