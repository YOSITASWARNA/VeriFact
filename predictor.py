import pickle
import re
import torch
import os
import warnings
import requests
import subprocess
import tempfile
from urllib.parse import urlparse, parse_qs
from transformers import RobertaTokenizer, RobertaForSequenceClassification

warnings.filterwarnings("ignore")

# 🔑 KEYS (UNCHANGED)
GNEWS_API_KEY = "a97e7ca27352ffdd1c312f201fbde06c"
YOUTUBE_API_KEY = "AIzaSyAa4SzQY02-L8oWC_ZKdaSzEHY0kLDOebo"
NEWS_API_KEY = "748c90a46f4a4e3c8f9505f631b4c277"

TRUSTED_SOURCES = [
    "bbc.com", "reuters.com", "ndtv.com",
    "thehindu.com", "indiatimes.com"
]

# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ---------------- LOAD MODELS ----------------
baseline_loaded = False
try:
    baseline_model = pickle.load(open("baseline_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    baseline_loaded = True
    print("Baseline model loaded")
except:
    print("Baseline not found")

roberta_loaded = False
if os.path.exists("best_roberta_model"):
    tokenizer = RobertaTokenizer.from_pretrained("best_roberta_model")
    roberta_model = RobertaForSequenceClassification.from_pretrained("best_roberta_model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roberta_model.to(device)
    roberta_model.eval()

    roberta_loaded = True
    print("RoBERTa loaded on", device)

# ---------------- PREDICTIONS ----------------
def predict_baseline(text):
    vec = vectorizer.transform([clean_text(text)])
    pred = baseline_model.predict(vec)[0]
    prob = baseline_model.predict_proba(vec)[0]
    return ("Real News" if pred == 1 else "Fake News", max(prob))

def predict_roberta(text):
    encoding = tokenizer(text, truncation=True, padding="max_length",
                         max_length=64, return_tensors="pt")

    with torch.no_grad():
        outputs = roberta_model(
            input_ids=encoding["input_ids"].to(device),
            attention_mask=encoding["attention_mask"].to(device)
        )
        probs = torch.softmax(outputs.logits, dim=1)[0]

    pred = torch.argmax(probs).item()
    return ("Real News" if pred == 1 else "Fake News", float(probs.max()))

# ---------------- 🔥 FIXED QUERY ----------------
def build_search_query(text):
    text = clean_text(text)

    words = text.split()

    stopwords = {
        "the","is","a","an","and","or","to","of","in","on","for",
        "with","that","this","it","as","at","by","from","was","are"
    }

    keywords = [w for w in words if w not in stopwords]

    return " ".join(keywords[:15])  # keep context

# ---------------- GNEWS ----------------
def gnews_search(query):
    url = "https://gnews.io/api/v4/search"

    params = {
        "q": query,
        "lang": "en",
        "max": 5,
        "token": GNEWS_API_KEY
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        return resp.json().get("articles", [])
    except:
        return []

# ---------------- NEWSAPI ----------------
def newsapi_search(query):
    url = "https://newsapi.org/v2/everything"

    params = {
        "q": query,
        "language": "en",
        "pageSize": 5,
        "apiKey": NEWS_API_KEY
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        return resp.json().get("articles", [])
    except:
        return []

# ---------------- 🔥 FIXED SEARCH ----------------
def search_news(text):
    query = build_search_query(text)
    print("🔍 Query:", query)

    results = gnews_search(query)

    if not results:
        print("⚠ GNews empty → retry full text")
        results = gnews_search(text)

    if not results:
        print("⚠ Still empty → NewsAPI fallback")
        results = newsapi_search(text)

    return [{
        "title": r.get("title"),
        "link": r.get("url")
    } for r in results]

def is_trusted(link):
    return any(src in link for src in TRUSTED_SOURCES)

# ---------------- YOUTUBE ----------------
def extract_youtube_id(url):
    try:
        parsed = urlparse(url)

        if parsed.hostname in ("youtu.be",):
            return parsed.path.lstrip("/")

        if parsed.hostname in ("youtube.com", "www.youtube.com"):
            qs = parse_qs(parsed.query)
            if "v" in qs:
                return qs["v"][0]
    except:
        return None
    return None

def fetch_youtube_metadata(video_id):
    url = "https://www.googleapis.com/youtube/v3/videos"

    params = {
        "part": "snippet",
        "id": video_id,
        "key": YOUTUBE_API_KEY
    }

    try:
        resp = requests.get(url, params=params)
        items = resp.json().get("items", [])

        if not items:
            return None

        snippet = items[0]["snippet"]
        return snippet.get("title", "") + " " + snippet.get("description", "")
    except:
        return None

def transcribe_video(url):
    try:
        import whisper

        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "audio.%(ext)s")

            subprocess.run([
                "yt-dlp","-x","--audio-format","mp3","-o",output,url
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            audio_file = None
            for f in os.listdir(tmpdir):
                if f.endswith(".mp3"):
                    audio_file = os.path.join(tmpdir, f)
                    break

            if not audio_file:
                return None

            model = whisper.load_model("base")
            return model.transcribe(audio_file)["text"]

    except:
        return None

# ---------------- 🔥 FIXED DECISION ----------------
def final_decision(model_label, model_conf, trusted_count):

    model_conf = model_conf * 100

    if trusted_count >= 2:
        return "Real News ✅"

    if trusted_count == 1:
        if model_label == "Real News":
            return "Likely Real ⚠"
        else:
            return "Unverified ⚠"

    if trusted_count == 0:
        if model_conf >= 80:
            return model_label + " (Model confident)"
        else:
            return "Unverified ❌"

# ---------------- OUTPUT ----------------
def print_result(label, conf, text):
    print("\n" + "="*60)
    print(f"MODEL: {label} ({round(conf*100,1)}%)")

    results = search_news(text)
    trusted = 0

    if results:
        print("\nTop News:")
        for r in results:
            t = is_trusted(r["link"])
            if t:
                trusted += 1
            print(("✅" if t else "⚠"), r["title"], "-", r["link"])

    print(f"\nTrusted sources: {trusted}/5")

    final = final_decision(label, conf, trusted)

    print(f"\nFINAL DECISION: {final}")
    print("="*60)

# ---------------- MENU ----------------
print("\nSelect model:")
print("1 - Baseline")
print("2 - RoBERTa")
print("3 - Both")

choice = input("Enter choice: ").strip()

print("\nMode:")
print("T - Text")
print("V - YouTube URL")

mode = input("Enter mode: ").strip().upper()

# ---------------- MAIN LOOP ----------------
while True:
    user_input = input("\nEnter input (or exit): ")

    if user_input.lower() == "exit":
        break

    text = user_input

    if mode == "V":
        vid = extract_youtube_id(user_input)
        combined = ""

        if vid:
            meta = fetch_youtube_metadata(vid)
            if meta:
                combined += meta + " "

        transcript = transcribe_video(user_input)
        if transcript:
            combined += transcript

        if combined:
            text = combined[:1000]

    if choice == "1":
        label, conf = predict_baseline(text)

    elif choice == "2":
        label, conf = predict_roberta(text)

    elif choice == "3":
        bl_label, bl_conf = predict_baseline(text)
        rb_label, rb_conf = predict_roberta(text)

        print(f"\nBaseline → {bl_label} ({round(bl_conf*100,1)}%)")
        print(f"RoBERTa  → {rb_label} ({round(rb_conf*100,1)}%)")

        label, conf = rb_label, rb_conf

    else:
        print("Invalid choice")
        continue

    print_result(label, conf, text)