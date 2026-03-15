import pickle
import re
import torch
import os
import warnings
from transformers import RobertaTokenizer, RobertaForSequenceClassification
warnings.filterwarnings("ignore")
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


print("Loading baseline model...")

baseline_loaded = False

try:
    baseline_model = pickle.load(open("baseline_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    baseline_loaded = True
    print("Baseline model loaded.")
except Exception as e:
    print("Baseline model not found:", e)


print("Loading RoBERTa model...")

model_path = "best_roberta_model"
roberta_loaded = False

if not os.path.exists(model_path):
    print("RoBERTa model folder not found!")
    print("Run Train_Models.py first.")
else:
    try:
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        roberta_model = RobertaForSequenceClassification.from_pretrained(model_path)

        if torch.cuda.is_available():
            try:
                device = torch.device("cuda")
                roberta_model.to(device)
                print("Device: GPU")
            except:
                device = torch.device("cpu")
                roberta_model.to(device)
                print("GPU memory full. Using CPU.")
        else:
            device = torch.device("cpu")
            roberta_model.to(device)
            print("Device: CPU")

        roberta_model.eval()
        roberta_loaded = True
        print("RoBERTa model loaded.")

    except Exception as e:
        print("Error loading RoBERTa model:", e)
        roberta_loaded = False
def predict_baseline(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    pred = baseline_model.predict(vec)[0]
    return "Real News" if pred == 1 else "Fake News"

def predict_roberta(text):

    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=64,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = roberta_model(input_ids=input_ids, attention_mask=attention_mask)
        pred = torch.argmax(outputs.logits, dim=1).item()

    return "Real News" if pred == 1 else "Fake News"


print("\n####### VeriFact System #######")
print("1 - Baseline Model (Fast)")
print("2 - RoBERTa Model (Accurate)")

choice = input("Select model (1/2): ").strip()

if choice not in ["1", "2"]:
    print("Invalid choice.")
    exit()

if choice == "1" and not baseline_loaded:
    print("Baseline model not available.")
    exit()

if choice == "2" and not roberta_loaded:
    print("RoBERTa model not available.")
    exit()
print("\nType news text to predict.")
print("Type 'exit' to quit.\n")

while True:

    text = input("Enter news: ")

    if text.lower() == "exit":
        print("Exiting VeriFact...")
        break

    if len(text.strip()) == 0:
        print("Please enter valid text.")
        continue

    if choice == "1":
        result = predict_baseline(text)
    else:
        result = predict_roberta(text)
    print("Result:", result)
    print("*" * 80)
