# ⚽ La Liga Match Predictor

A comprehensive project that combines **web scraping** and **machine learning** to predict La Liga match outcomes.

---

## 📖 Overview

This project has two main components:

1. **Data Scraper** – Collects historical La Liga match data from FBref.
2. **Machine Learning Model** – Trains predictive models to forecast match outcomes.

---

## 📂 Project Structure

```
LaLiga_Predictor/
├── Scrapper/
│   ├── la_liga_scraper.py       # Script to scrape La Liga data
│   └── data/
│       └── la_liga_results_10_years.csv
├── ML_Model/
│   ├── train_model.py           # Train ML models
│   ├── predict.py               # Make predictions
│   ├── models/                  # Saved trained models
│   └── data/                    # Processed datasets
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Abhigyan-GG/La_Liga_Predictor.git
cd LaLiga_Predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Data Collection

Run the scraper to collect historical match data:

```bash
cd Scrapper
python la_liga_scraper.py
```

This generates a CSV file (`la_liga_results_10_years.csv`) in the `Scrapper/data/` directory.

---

### 2. Model Training

Train the machine learning model:

```bash
cd ML_Model
python train_model.py
```

This will:

* Preprocess historical data
* Train multiple ML algorithms
* Save trained models to the `models/` directory
* Evaluate model performance

---

### 3. Making Predictions

Predict upcoming match outcomes:

```bash
cd ML_Model
python predict.py
```

---

## ✨ Features

### 🔹 Data Collection

* Scrapes match data from FBref (last 10 seasons)
* Extracts team names, scores, results, and statistics
* Handles multiple formats and edge cases

### 🔹 Machine Learning Model

* Preprocesses historical match data
* Implements algorithms (Random Forest, Gradient Boosting, etc.)
* Evaluates models with cross-validation
* Provides feature importance analysis

### 🔹 Predictions

* Forecasts outcomes of upcoming matches
* Outputs probability estimates for each result
* Presents results in a readable format

---

## 🧠 Model Details

The ML model considers:

* Historical team performance
* Home vs. away advantage
* Recent form indicators
* Head-to-head records
* Team strength metrics

---

## ⚙️ Configuration

* **Scraper:** Change seasons to scrape in `Scrapper/la_liga_scraper.py` (`SEASONS` variable).
* **Training:** Modify algorithm parameters in `ML_Model/train_model.py`.
* **Predictions:** Configure matches to predict in `ML_Model/predict.py`.

---

## 📦 Dependencies

* Python 3.7+
* pandas
* numpy
* scikit-learn
* requests
* beautifulsoup4
* lxml

(See `requirements.txt` for the full list.)

---

## 🚧 Future Enhancements

* Real-time data updates
* Player statistics & injury impact
* Web interface for predictions
* REST API for integration
* Advanced model explainability

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a Pull Request

---

## 📜 License

This project is open-source. Feel free to use and contribute!

---
