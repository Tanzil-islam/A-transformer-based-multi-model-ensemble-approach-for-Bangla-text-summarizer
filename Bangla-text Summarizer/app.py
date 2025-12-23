# app.py
from flask import Flask, render_template, request, jsonify
from summarizer_utils import summarize_single_input
import pandas as pd

app = Flask(__name__)
app.secret_key = "change-this-to-something-random"


# ----------------------------
#  HOME PAGE
# ----------------------------
@app.route("/")
def home():
    return render_template("home.html")


# ----------------------------
#  DATA ANALYSIS (STATIC)
# ----------------------------
@app.route("/data-analysis")
def data_analysis():
    """
    No file upload. Dataset is already available.
    Images and stats are static and loaded from /static/eda/
    """
    # Load dataset stored in /static/eda/dataset.xlsx
    df = pd.read_excel("static/eda/dataset.xlsx")

    # Prepare small preview (first 15 rows)
    preview_html = df.head(15).to_html(classes="table table-striped", index=False)

    # Static stats you provided
    stats = {
        "total_rows": 3300,
        "train": {"rows": 2310, "pct": "70.00%"},
        "val":   {"rows": 346,  "pct": "10.48%"},
        "test":  {"rows": 644,  "pct": "19.52%"},
    }

    # Word-count table you provided (first rows)
    wc_table = pd.DataFrame({
        "Text_word_count": [140, 227, 351, 192, 450],
        "Summary_word_count": [37, 43, 18, 25, 29],
        "summary_ratio": [0.264286, 0.189427, 0.051282, 0.130208, 0.064444],
    }).to_html(classes="table table-bordered", index=False)

    # Image paths (STATIC)
    images = {
        "wordcloud_text":      "/static/eda/wordcloud_Text.png",
        "wordcloud_summary":   "/static/eda/wordcloud_summary.png",
        "split_bar":           "/static/eda/train_testsplitbar.png",
        "text_wc_dist":        "/static/eda/text word count distribution.png",
        "summary_wc_dist":     "/static/eda/summary word count distribuition.png",
        "ratio_dist":          "/static/eda/distribution of summary ratio.png",
    }

    return render_template(
        "data_analysis.html",
        preview_table=preview_html,
        stats=stats,
        wc_table=wc_table,
        images=images
    )


# ----------------------------
#  MODEL IMPLEMENTATION (SUMMARIZER)
# ----------------------------
@app.route("/model", methods=["GET", "POST"])
def model():
    text = ""
    reference = ""
    result = None

    if request.method == "POST":
        text = request.form.get("text", "")
        reference = request.form.get("reference", "")

        if len(text.strip()) == 0:
            return render_template("model.html", error="Please paste news text.")

        try:
            result = summarize_single_input(text, reference)
        except Exception as e:
            return render_template("model.html", error=f"Summarization error: {e}")

    return render_template("model.html", text=text, reference=reference, result=result)


# ----------------------------
#  REPORT PAGE
# ----------------------------
@app.route("/report")
def report():
    return render_template("report.html")


# ----------------------------
#  MAIN ENTRY
# ----------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
