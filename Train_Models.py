import pandas
import re
import pickle
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def main():
    print("\nLoading dataset...")

    fake_df = pandas.read_csv("Fake.csv")
    true_df = pandas.read_csv("True.csv")

    fake_df["label"] = 0
    true_df["label"] = 1

    df = pandas.concat([fake_df, true_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Dataset size:", len(df))

    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        text = re.sub(r"\breuters\b|\bap\b|\bafp\b", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    df["content"] = (
        df["title"].fillna("") + " " +
        df["title"].fillna("") + " " +
        df["text"].fillna("")
    )

    df["content"] = df["content"].apply(lambda x: clean_text(x)[:400])

    X = df["content"]
    y = df["label"]

    print("\nClass Distribution:")
    print(y.value_counts())

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print("\nTrain size:", len(X_train))
    print("Validation size:", len(X_val))
    print("Test size:", len(X_test))
    print("\nVectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=10000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print("\nTraining Baseline Models...")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
    }

    best_model = None
    best_f1 = 0

    for name, model in models.items():
        print(f"\nTraining {name}")
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("Accuracy :", round(acc, 4))
        print("Precision:", round(prec, 4))
        print("Recall   :", round(rec, 4))
        print("F1 Score :", round(f1, 4))

        if f1 > best_f1:
            best_f1 = f1
            best_model = model

    pickle.dump(best_model, open("baseline_model.pkl", "wb"))
    pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))
    print("\nBest baseline model saved!")
    print("\nTraining RoBERTa Model...")
    epochs = 3
    batch_size = 16
    max_length = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    model.to(device)

    class NewsDataset(Dataset):
        def __init__(self, texts, labels):
            self.texts = texts.tolist()
            self.labels = labels.tolist()

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            encoding = tokenizer(
                self.texts[idx],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            item = {k: v.squeeze(0) for k, v in encoding.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

    train_dataset = NewsDataset(X_train, y_train)
    val_dataset = NewsDataset(X_val, y_val)
    test_dataset = NewsDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_training_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        print("Average loss:", total_loss / len(train_loader))

    print("\nTesting model...")
    model.eval()
    preds, true = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            outputs = model(batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
            p = torch.argmax(outputs.logits, dim=1)
            preds.extend(p.cpu().numpy())
            true.extend(batch["labels"].numpy())
    acc = accuracy_score(true, preds)
    prec = precision_score(true, preds)
    rec = recall_score(true, preds)
    f1 = f1_score(true, preds)
    print("\nTest Results")
    print("Accuracy :", round(acc, 4))
    print("Precision:", round(prec, 4))
    print("Recall   :", round(rec, 4))
    print("F1 Score :", round(f1, 4))

    model.save_pretrained("best_roberta_model")
    tokenizer.save_pretrained("best_roberta_model")
    print("\nRoBERTa model saved to folder: best_roberta_model")
    print("\nTraining Completed!")

if __name__ == "__main__":
    main()
