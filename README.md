Below is a **clean, professional, GitHub-ready README.md** tailored exactly to your project structure, code, and UI.
You can copy-paste this directly into `README.md`.

---

# 🕵️ Bangla Crime News Summarizer

An **automatic Bangla crime news summarization system** built using an **ensemble of fine-tuned transformer models (BanglaT5 + mT5)** with **metric-based selection** to produce concise, faithful, and informative summaries.

This project combines **deep learning**, **NLP evaluation metrics**, and a **Flask web interface** to deliver a complete research-grade summarization pipeline.

---

## 🔍 Project Overview

Bangla crime news articles are often long, repetitive, and difficult to skim. Manual summarization is time-consuming and subjective.

This system:

* Generates **multiple candidate summaries** using different ensemble strategies
* Evaluates them using **semantic, lexical, coverage, and faithfulness metrics**
* Automatically selects the **best, least-hallucinated summary**
* Presents results through an **interactive web interface**

---

## 🚀 Key Features

### 🧠 Ensemble Summarization

* **BanglaT5** (fine-tuned)
* **mT5** (fine-tuned)
* Five summarization strategies:

  * Extractive Fusion
  * Rank Fusion (SBERT similarity)
  * Voting Fusion (n-gram voting)
  * Transformer Fusion
  * Hybrid Fusion (best of rank + voting)

### 📊 Metric-Based Selection

Each candidate summary is evaluated using:

* SBERT semantic similarity
* BERTScore (Bangla)
* ROUGE-L
* BLEU
* Character Error Rate (CER)
* Coverage
* Length adequacy
* Faithfulness / Hallucination score
* Keyword preservation
* Lexical diversity & complexity

The final summary is selected using a **weighted scoring framework** with a **confidence score**.

---

## 🖥️ Web Application

Built using **Flask**, the web interface provides:

### Pages

* **Home** – Project overview and team
* **Dataset Analysis** – EDA dashboard with plots & wordclouds
* **Model & Results** – Interactive summarization + metrics
* **Defense Report** – Embedded PDF report

### Visual Outputs

* Per-method final score bar chart
* Metric breakdown of the best method
* Grouped metric comparison across methods

---

## 🗂️ Project Structure

```
bangla-crime-news-summarizer/
│
├── app.py                     # Flask application entry point
├── summarizer_utils.py        # Ensemble logic, metrics, plotting
├── models_config.py           # Model path configuration
│
├── templates/
│   ├── home.html
│   ├── data_analysis.html
│   ├── model.html
│   └── report.html
│
├── static/
│   ├── eda/                   # Dataset plots & wordclouds
│   ├── img/                   # Team images
│   └── report.pdf             # Defense report
│
├── models/
│   ├── saved_model/           # Fine-tuned BanglaT5
│   └── saved_mt5/             # Fine-tuned mT5
│
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/bangla-crime-news-summarizer.git
cd bangla-crime-news-summarizer
```

### 2️⃣ Create Environment & Install Dependencies

```bash
pip install -r requirements.txt
```

**Key libraries used:**

* `transformers`
* `sentence-transformers`
* `torch`
* `evaluate`
* `bert-score`
* `rouge-score`
* `flask`
* `pandas`
* `matplotlib`

---

## 🧠 Model Setup

Place your fine-tuned models in:

```
models/
├── saved_model/   (BanglaT5)
└── saved_mt5/     (mT5)
```

Or set custom paths:

```bash
export BANG_LAT5_PATH=/path/to/banglat5
export MT5_PATH=/path/to/mt5
```

The system automatically detects **GPU vs CPU** and adjusts beam size and sequence lengths.

---

## ▶️ Run the Application

```bash
python app.py
```

Then open:

```
http://127.0.0.1:5000
```

⚠️ **First request may take longer** due to model loading.

---

## 📈 Dataset Analysis

The dataset analysis dashboard includes:

* Dataset preview (Excel-based)
* Train / Validation / Test split
* Word count distributions
* Summary ratio analysis
* Wordclouds for text & summaries

Dataset is loaded from:

```
static/eda/dataset.xlsx
```

---

## 📄 Defense Report

A full research report is available inside the app and as a PDF:

* Dataset construction
* Preprocessing pipeline
* Model fine-tuning
* Ensemble strategies
* Metric design
* Result analysis

---

## 👥 Team

* **Tanzil Islam**
  Undergraduate Researcher
  📧 [kazimdtanzilislam@gmail.com](mailto:kazimdtanzilislam@gmail.com)

---

## 🧪 Research & Academic Use

This project is suitable for:

* NLP research
* Low-resource language summarization
* Faithfulness & hallucination analysis
* Final year / thesis / defense demonstrations

---

## 📜 License

This project is intended for **academic and research use**.
Please cite appropriately if used in publications.

---

