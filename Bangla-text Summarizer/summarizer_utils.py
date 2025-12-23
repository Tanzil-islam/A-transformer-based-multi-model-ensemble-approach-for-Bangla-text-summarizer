# ============================================================
#  summarizer_utils.py — Ensemble + Metrics + Extras (FINAL)
# ============================================================

import io
import base64
import re
from collections import Counter
import warnings

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
from bert_score import score as bert_score_fn
import evaluate
from rouge_score import rouge_scorer
from unidecode import unidecode

from models_config import BANG_LAT5_PATH, MT5_PATH

warnings.filterwarnings("ignore")

# ============================================================
#  DEVICE & RUNTIME CONFIG (GPU/CPU AWARE)
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Slightly lighter settings if only CPU is available
if DEVICE == "cuda":
    MAX_INPUT = 1024
    MAX_OUTPUT = 360
    NUM_BEAMS = 6
else:
    MAX_INPUT = 896
    MAX_OUTPUT = 300
    NUM_BEAMS = 4

print(f"[summarizer_utils] DEVICE = {DEVICE}")
print(f"[summarizer_utils] MAX_INPUT = {MAX_INPUT}, MAX_OUTPUT = {MAX_OUTPUT}, NUM_BEAMS = {NUM_BEAMS}")

# ============================================================
#  MODEL LOADING (ONE-TIME)
# ============================================================

print("[summarizer_utils] Loading BanglaT5 tokenizer/model...")
tok_b5 = AutoTokenizer.from_pretrained(BANG_LAT5_PATH, local_files_only=True)
model_b5 = AutoModelForSeq2SeqLM.from_pretrained(BANG_LAT5_PATH).to(DEVICE)

print("[summarizer_utils] Loading mT5 tokenizer/model...")
tok_m5 = AutoTokenizer.from_pretrained(MT5_PATH, local_files_only=True)
model_m5 = AutoModelForSeq2SeqLM.from_pretrained(MT5_PATH).to(DEVICE)

print("[summarizer_utils] Loading SentenceTransformer (SBERT)...")
SBERT = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=DEVICE)

print("[summarizer_utils] Loading metrics (bleu, cer, rouge)...")
bleu_metric = evaluate.load("bleu")
cer_metric = evaluate.load("cer")
ROUGE_SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

print("[summarizer_utils] Models & metrics loaded successfully.")


# ============================================================
#  MATPLOTLIB → BASE64
# ============================================================

def _fig_to_base64():
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    return img_b64


# ============================================================
#  SIMPLE SENTENCE SPLITTER (BENGALI-FRIENDLY)
# ============================================================

def split_sentences(text: str):
    if not text or not isinstance(text, str):
        return []
    text = re.sub(r"\.(?=\S)", ". ", text)
    parts = re.split(r"[।!?]+", text)
    return [p.strip() for p in parts if len(p.strip()) > 3]


# ============================================================
#  GENERATION HELPERS (BanglaT5 + mT5)
# ============================================================

def generate_banglat5(text: str) -> str:
    inputs = tok_b5(
        text,
        return_tensors="pt",
        truncation=True,
        padding="longest",
        max_length=MAX_INPUT,
    ).to(DEVICE)

    with torch.no_grad():
        out = model_b5.generate(
            **inputs,
            max_length=MAX_OUTPUT,
            num_beams=NUM_BEAMS,
            length_penalty=1.0,
            early_stopping=False,
            no_repeat_ngram_size=3,
        )
    return tok_b5.decode(out[0], skip_special_tokens=True).strip()


def generate_mt5(text: str) -> str:
    inputs = tok_m5(
        text,
        return_tensors="pt",
        truncation=True,
        padding="longest",
        max_length=MAX_INPUT,
    ).to(DEVICE)

    with torch.no_grad():
        out = model_m5.generate(
            **inputs,
            max_length=MAX_OUTPUT,
            num_beams=NUM_BEAMS,
            length_penalty=1.0,
            early_stopping=False,
            no_repeat_ngram_size=3,
        )
    return tok_m5.decode(out[0], skip_special_tokens=True).strip()


# ============================================================
#  VOTING HELPERS
# ============================================================

def extract_ngrams(tokens, n: int):
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def phrase_reconstruction(phrases, max_phrases=30):
    if not phrases:
        return ""
    chosen = []
    used = set()
    for p in phrases:
        if p not in used:
            chosen.append(p)
            used.add(p)
        if len(chosen) >= max_phrases:
            break

    output = chosen[0]
    for ph in chosen[1:]:
        out_tokens = output.split()
        ph_tokens = ph.split()
        max_ov = 0
        for k in range(min(len(out_tokens), len(ph_tokens)), 0, -1):
            if out_tokens[-k:] == ph_tokens[:k]:
                max_ov = k
                break
        if max_ov > 0:
            output = " ".join(out_tokens + ph_tokens[max_ov:])
        else:
            output += " " + ph
    return output


# ============================================================
#  ENSEMBLE METHODS
# ============================================================

def extractive_fusion(text: str) -> str:
    sents = split_sentences(text)
    if len(sents) <= 3:
        return " ".join(sents).strip()
    mid = sents[len(sents) // 2]
    return " ".join([sents[0], mid, sents[-1]]).strip()


def rank_fusion(text: str) -> str:
    s1 = generate_banglat5(text)
    s2 = generate_mt5(text)

    emb_doc = SBERT.encode(text, convert_to_tensor=True)
    emb_s1 = SBERT.encode(s1, convert_to_tensor=True)
    emb_s2 = SBERT.encode(s2, convert_to_tensor=True)

    sim1 = util.cos_sim(emb_doc, emb_s1).item()
    sim2 = util.cos_sim(emb_doc, emb_s2).item()

    return s1 if sim1 >= sim2 else s2


def voting_fusion(text: str, top_k_phrases=50, ngram_range=(2, 4)) -> str:
    s1 = generate_banglat5(text)
    s2 = generate_mt5(text)

    phrase_counts = Counter()
    for s in [s1, s2]:
        tokens = s.split()
        for n in range(ngram_range[0], ngram_range[1] + 1):
            phrase_counts.update(extract_ngrams(tokens, n))

    if not phrase_counts:
        return rank_fusion(text)

    top_phrases = [p for p, _ in phrase_counts.most_common(top_k_phrases)]
    return phrase_reconstruction(top_phrases, max_phrases=30).strip()


def transformer_fusion(text: str) -> str:
    s1 = generate_banglat5(text)
    s2 = generate_mt5(text)

    fusion_input = f"S1: {s1}\nS2: {s2}\nFuse:"
    inputs = tok_b5(
        fusion_input,
        return_tensors="pt",
        truncation=True,
        padding="longest",
        max_length=1500,
    ).to(DEVICE)

    with torch.no_grad():
        out = model_b5.generate(
            **inputs,
            max_length=480,
            num_beams=NUM_BEAMS,
            length_penalty=1.1,
            early_stopping=False,
            no_repeat_ngram_size=3,
        )
    return tok_b5.decode(out[0], skip_special_tokens=True).strip()


def hybrid_fusion(text: str) -> str:
    r = rank_fusion(text)
    v = voting_fusion(text)
    combined = (r + " " + v).split()

    final_tokens = []
    prev = None
    repeat = 0
    for t in combined:
        if t == prev:
            repeat += 1
        else:
            repeat = 0
        if repeat < 2:
            final_tokens.append(t)
        prev = t
    return " ".join(final_tokens).strip()


def ensemble_summarize(text: str, method: str = "hybrid") -> str:
    m = method.lower()
    if m == "extractive":
        return extractive_fusion(text)
    if m == "rank":
        return rank_fusion(text)
    if m == "voting":
        return voting_fusion(text)
    if m == "transformer":
        return transformer_fusion(text)
    if m == "hybrid":
        return hybrid_fusion(text)
    raise ValueError(f"Unknown ensemble method: {method}")


# ============================================================
#  METRICS
# ============================================================

# -- helper: tokenize Bengali/Latin words
def _content_tokens(text: str):
    tokens = re.findall(r"[\w\u0980-\u09FF]+", text.lower())
    return [t for t in tokens if len(t) >= 4 and not t.isdigit()]


def compute_sbert_sim(original: str, summary: str) -> float:
    if not original or not summary:
        return 0.0
    emb_doc = SBERT.encode(original, convert_to_tensor=True)
    emb_sum = SBERT.encode(summary, convert_to_tensor=True)
    sim = util.cos_sim(emb_doc, emb_sum).item()
    return float(max(0.0, min(1.0, sim)))


def compute_bertscore(pred: str, ref: str) -> float:
    if not ref:
        return 0.0
    try:
        _, _, f1 = bert_score_fn([pred], [ref], lang="bn", verbose=False)
        return float(f1.mean().item())
    except Exception:
        return 0.0


def compute_rouge_scores(pred: str, ref: str):
    if not ref:
        return 0.0, 0.0, 0.0
    try:
        cand = unidecode(pred)
        refp = unidecode(ref)
        scores = ROUGE_SCORER.score(refp, cand)
        return (
            float(scores["rouge1"].fmeasure),
            float(scores["rouge2"].fmeasure),
            float(scores["rougeL"].fmeasure),
        )
    except Exception:
        return 0.0, 0.0, 0.0


def compute_bleu_single(pred: str, ref: str) -> float:
    if not ref:
        return 0.0
    try:
        out = bleu_metric.compute(predictions=[pred], references=[[ref]])
        return float(out.get("bleu", 0.0))
    except Exception:
        return 0.0


def compute_cer(pred: str, ref: str) -> float:
    if not ref:
        return 1.0
    try:
        r = cer_metric.compute(predictions=[pred], references=[ref])
        return float(r)
    except Exception:
        return 1.0


# ---------- Hallucination Detection (heuristic) ----------

def compute_hallucination_score(summary: str, article: str) -> float:
    """
    Approx: tokens that appear in summary but never appear (even as substring)
    in the article. Higher = more hallucination risk.
    """
    if not summary:
        return 0.0

    art_tokens = _content_tokens(article)
    sum_tokens = _content_tokens(summary)
    if not sum_tokens:
        return 0.0

    art_set = set(art_tokens)
    hallucinated = 0
    for t in set(sum_tokens):
        if t not in art_set:
            hallucinated += 1

    score = hallucinated / max(1, len(set(sum_tokens)))
    return float(max(0.0, min(1.0, score)))


def compute_faithfulness(summary: str, article: str) -> float:
    """
    Faithfulness = 1 - hallucination_score.
    """
    return 1.0 - compute_hallucination_score(summary, article)


# ---------- Keyword Preservation Metric ----------

def extract_keywords(text: str, top_k: int = 15):
    tokens = _content_tokens(text)
    if not tokens:
        return []
    freq = Counter(tokens)
    return [w for w, _ in freq.most_common(top_k)]


def compute_keyword_preservation(summary: str, article: str) -> float:
    """
    Fraction of top-k article keywords that appear in the summary.
    """
    if not article or not summary:
        return 0.0

    keywords = extract_keywords(article, top_k=15)
    if not keywords:
        return 0.0

    summary_tokens = set(_content_tokens(summary))
    preserved = sum(1 for k in keywords if k in summary_tokens)
    return float(preserved / len(keywords))


# ---------- Coverage / Length / Complexity / Diversity ----------

def compute_coverage(summary: str, reference: str) -> float:
    if not reference or not summary:
        return 0.0
    ref_tokens = _content_tokens(reference)
    sum_tokens = set(_content_tokens(summary))
    if not ref_tokens:
        return 0.0
    covered = sum(1 for t in ref_tokens if t in sum_tokens)
    return float(covered / len(ref_tokens))


def compute_length_score(summary: str, reference: str) -> float:
    if not reference or not summary:
        return 0.0
    Ls = len(summary.split())
    Lr = max(1, len(reference.split()))
    ratio = Ls / Lr

    # strongly penalize very short
    if ratio < 0.40:
        return 0.05 * ratio

    # sweet spot
    if 0.70 <= ratio <= 1.10:
        return 1.0 - abs(1.0 - ratio) * 0.25

    # too long
    if ratio > 1.10:
        return max(0.0, 1.0 - (ratio - 1.10) * 0.5)

    base = 1.0 - abs(1.0 - ratio)
    return (base ** 1.5) * 0.65


def compute_complexity(summary: str) -> float:
    tokens = [t for t in summary.split() if t.strip()]
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def compute_diversity(summary: str) -> float:
    s = summary.replace(" ", "")
    if not s:
        return 0.0
    return len(set(s)) / len(s)


# ============================================================
#  WEIGHTS (INCLUDING NEW METRICS)
# ============================================================

BASE_WEIGHTS = {
    "sbert": 0.17,
    "bertscore": 0.14,
    "rouge": 0.11,
    "coverage": 0.21,
    "length": 0.18,
    "faithfulness": 0.09,           # new
    "keywords": 0.06,               # new
    "bleu": 0.02,
    "cer": 0.01,
    "complexity": 0.01,
    "diversity": 0.01,
}

_total = sum(BASE_WEIGHTS.values())
WEIGHTS = {k: v / _total for k, v in BASE_WEIGHTS.items()}


# ============================================================
#  SCORE A CANDIDATE
# ============================================================

def score_candidate(summary: str, article: str, reference: str):
    """
    Compute all metrics for one candidate summary.
    """
    ref_for_metrics = reference.strip() or article

    sbert = compute_sbert_sim(article, summary)
    bert = compute_bertscore(summary, ref_for_metrics)
    rouge1, rouge2, rougeL = compute_rouge_scores(summary, ref_for_metrics)
    rouge = rougeL
    bleu = compute_bleu_single(summary, ref_for_metrics)
    cer = compute_cer(summary, ref_for_metrics)
    cer_inv = max(0.0, 1.0 - cer)
    coverage = compute_coverage(summary, ref_for_metrics)
    length_sc = compute_length_score(summary, ref_for_metrics)
    complexity = compute_complexity(summary)
    diversity = compute_diversity(summary)
    faith = compute_faithfulness(summary, article)
    hall = 1.0 - faith
    kw_pres = compute_keyword_preservation(summary, article)

    def clamp(x):
        return float(max(0.0, min(1.0, x)))

    sbert = clamp(sbert)
    bert = clamp(bert)
    rouge = clamp(rouge)
    bleu = clamp(bleu)
    cer_inv = clamp(cer_inv)
    coverage = clamp(coverage)
    length_sc = clamp(length_sc)
    complexity = clamp(complexity)
    diversity = clamp(diversity)
    faith = clamp(faith)
    hall = clamp(hall)
    kw_pres = clamp(kw_pres)

    final = (
        WEIGHTS["sbert"] * sbert +
        WEIGHTS["bertscore"] * bert +
        WEIGHTS["rouge"] * rouge +
        WEIGHTS["coverage"] * coverage +
        WEIGHTS["length"] * length_sc +
        WEIGHTS["faithfulness"] * faith +
        WEIGHTS["keywords"] * kw_pres +
        WEIGHTS["bleu"] * bleu +
        WEIGHTS["cer"] * cer_inv +
        WEIGHTS["complexity"] * complexity +
        WEIGHTS["diversity"] * diversity
    )

    # ---------- Confidence score ----------
    # Combines final score, faithfulness, and coverage.
    confidence = final * (0.5 + 0.5 * faith) * (0.4 + 0.6 * coverage)+0.2
    #confidence = clamp(confidence)

    return {
        "sbert": sbert,
        "bertscore": bert,
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeL": rougeL,
        "rouge": rouge,
        "bleu": bleu,
        "cer": cer,
        "cer_inv": cer_inv,
        "coverage": coverage,
        "length": length_sc,
        "complexity": complexity,
        "diversity": diversity,
        "faithfulness": faith,
        "hallucination_score": hall,
        "keyword_preservation": kw_pres,
        "final": final,
        "confidence": confidence,
    }


# ============================================================
#  SELECT BEST SUMMARY
# ============================================================

def select_best_summary(article: str, reference: str, candidates_dict: dict):
    score_table = {}

    for m, summ in candidates_dict.items():
        if not summ or not isinstance(summ, str) or not summ.strip():
            score_table[m] = {
                "sbert": 0.0,
                "bertscore": 0.0,
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0,
                "rouge": 0.0,
                "bleu": 0.0,
                "cer": 1.0,
                "cer_inv": 0.0,
                "coverage": 0.0,
                "length": 0.0,
                "complexity": 0.0,
                "diversity": 0.0,
                "faithfulness": 0.0,
                "hallucination_score": 1.0,
                "keyword_preservation": 0.0,
                "final": 0.0,
                "confidence": 0.0,
            }
        else:
            score_table[m] = score_candidate(summ, article, reference)

    # primary: final, secondary: confidence
    best_method = max(
        score_table.keys(),
        key=lambda k: (score_table[k]["final"], score_table[k]["confidence"])
    )
    return best_method, candidates_dict[best_method], score_table


# ============================================================
#  PLOT HELPERS
# ============================================================

def make_final_scores_bar(score_tab: dict) -> str:
    methods = list(score_tab.keys())
    finals = [float(score_tab[m]["final"]) for m in methods]

    plt.figure(figsize=(7, 3))
    plt.bar(methods, finals, color=plt.get_cmap("tab10").colors[: len(methods)])
    plt.ylim(0, 1)
    plt.ylabel("Final score (0..1)")
    plt.title("Final score per method (single input)")
    return _fig_to_base64()


def make_best_method_metric_plot(best_scores: dict) -> str:
    labels = [
        "sbert",
        "bertscore",
        "rouge",
        "coverage",
        "length",
        "faithfulness",
        "keyword_preservation",
        "bleu",
        "cer_inv",
        "complexity",
        "diversity",
    ]
    vals = [
        best_scores["sbert"],
        best_scores["bertscore"],
        best_scores["rouge"],
        best_scores["coverage"],
        best_scores["length"],
        best_scores["faithfulness"],
        best_scores["keyword_preservation"],
        best_scores["bleu"],
        best_scores["cer_inv"],
        best_scores["complexity"],
        best_scores["diversity"],
    ]

    plt.figure(figsize=(10, 3))
    plt.bar(labels, vals, color=plt.get_cmap("tab20").colors[: len(labels)])
    plt.ylim(0, 1)
    plt.xticks(rotation=25)
    plt.ylabel("Score (0..1)")
    plt.title("Metric breakdown for best method")
    return _fig_to_base64()


def make_grouped_metrics(score_tab: dict) -> str:
    metrics = [
        "sbert",
        "bertscore",
        "rouge",
        "coverage",
        "length",
        "faithfulness",
        "keyword_preservation",
        "bleu",
        "cer_inv",
        "complexity",
        "diversity",
    ]
    methods = list(score_tab.keys())
    M = len(methods)
    K = len(metrics)

    vals = np.zeros((M, K))
    for i, m in enumerate(methods):
        for j, metric in enumerate(metrics):
            vals[i, j] = float(score_tab[m].get(metric, 0.0))
            vals[i, j] = max(0.0, min(1.0, vals[i, j]))

    x = np.arange(M)
    total_width = 0.85
    bar_width = total_width / K
    cmap = plt.get_cmap("tab10")

    plt.figure(figsize=(max(10, M * 1.5), 5))
    for j in range(K):
        offsets = x - total_width / 2 + (j + 0.5) * bar_width
        plt.bar(
            offsets,
            vals[:, j],
            width=bar_width * 0.95,
            label=metrics[j].upper(),
            color=cmap(j % 10),
            alpha=0.9,
        )

    plt.xticks(x, methods)
    plt.ylim(0, 1)
    plt.ylabel("Metric (0..1)")
    plt.title("Metrics per method (single input)")
    plt.legend(ncol=3, bbox_to_anchor=(0.5, -0.25), loc="upper center")
    return _fig_to_base64()


# ============================================================
#  FLASK ENTRY POINT
# ============================================================

def summarize_single_input(text: str, reference: str = ""):
    methods = ["extractive", "rank", "voting", "transformer", "hybrid"]

    outputs = {}
    for m in methods:
        try:
            outputs[m] = ensemble_summarize(text, method=m)
        except Exception as e:
            print(f"[summarizer_utils] Error generating with {m}: {e}")
            outputs[m] = ""

    best_method, best_summary, score_tab = select_best_summary(text, reference, outputs)

    return {
        "input_preview": text[:400],
        "summaries": outputs,
        "best_method": best_method,
        "best_summary": best_summary,
        "scores": score_tab,
        "plots": {
            "per_method_final_scores": make_final_scores_bar(score_tab),
            "best_method_metric_breakdown": make_best_method_metric_plot(score_tab[best_method]),
            "per_method_metrics_grouped": make_grouped_metrics(score_tab),
        },
        "runtime": {
            "device": DEVICE,
            "max_input": MAX_INPUT,
            "max_output": MAX_OUTPUT,
            "num_beams": NUM_BEAMS,
        },
    }
