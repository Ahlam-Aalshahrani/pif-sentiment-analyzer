# 📊 PIF Sentiment Analyzer

A bilingual (Arabic/English) NLP tool that classifies the sentiment of news headlines and social media posts about Saudi Arabia's Public Investment Fund (PIF) and its giga-projects — NEOM, Qiddiya, Red Sea Project, Diriyah Gate, and more.

Built on a **TF-IDF + classical ML** pipeline, extending prior work on Arabic/English intent classification (90.82% accuracy on CLINC150/BANKING77 benchmarks) into the domain of sovereign investment sentiment monitoring.

## ✨ Features

- **Bilingual support** — handles Arabic and English text in the same pipeline, with Arabic-specific normalization (diacritics removal, letter unification).
- **Three-class sentiment** — positive / negative / neutral.
- **Interactive dashboard** — Streamlit app for single-headline and batch (CSV) analysis, with confidence scores and downloadable results.
- **Lightweight & interpretable** — TF-IDF + Logistic Regression/SVM, no GPU required, fast to retrain on new data.
- **Extensible dataset** — sample dataset included; easily swap in real scraped news/tweets.

## 🗂️ Project Structure

```
pif-sentiment-analyzer/
├── data/
│   └── pif_sentiment_sample.csv    # Sample bilingual labeled dataset
├── src/
│   ├── preprocess.py               # Arabic/English text cleaning & normalization
│   ├── train.py                    # Train TF-IDF + classifier
│   ├── predict.py                  # CLI inference on new text
│   └── app.py                      # Streamlit dashboard
├── models/                         # Saved vectorizer & classifier (generated)
├── requirements.txt
└── README.md
```

## 🚀 Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/pif-sentiment-analyzer.git
cd pif-sentiment-analyzer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model on the sample dataset
python src/train.py --data data/pif_sentiment_sample.csv --out models/

# 4. Predict on a single headline
python src/predict.py --text "PIF announces new investment in green hydrogen"

# 5. Launch the interactive dashboard
streamlit run src/app.py
```

## 🧠 How It Works

1. **Preprocessing** (`src/preprocess.py`) — normalizes Arabic script (removes diacritics, unifies alef/ya/ta-marbuta forms), strips URLs/mentions/punctuation, and lowercases English text.
2. **Feature extraction** — TF-IDF vectorization with unigrams + bigrams.
3. **Classification** — Logistic Regression (default) or Linear SVM, with class balancing for imbalanced sentiment distributions.
4. **Inference** — `SentimentPredictor` class loads the saved vectorizer/model and returns predicted label + class probabilities.

## 📈 Example Output

```
Text:       PIF announces record $20 billion investment in green hydrogen
Sentiment:  POSITIVE
Confidence: 78.4%
Probabilities:
  negative  : 8.1%
  neutral   : 13.5%
  positive  : 78.4%
```

## 🔧 Extending This Project

- **Real data**: Swap `data/pif_sentiment_sample.csv` with scraped headlines from Saudi news outlets (Arab News, Al Arabiya) or the X/Twitter API.
- **Transformer models**: Replace TF-IDF with AraBERT or multilingual BERT for higher accuracy on nuanced text.
- **Live monitoring**: Wrap `SentimentPredictor` in a scheduled pipeline (e.g., n8n) to track sentiment trends over time.
- **Topic modeling**: Add project-specific tagging (NEOM vs. Qiddiya vs. Red Sea) alongside sentiment.

## 📄 License

MIT License — free to use, modify, and build on.

## 👤 Author

Built as a portfolio project demonstrating applied NLP for sovereign investment and Vision 2030 project monitoring.
