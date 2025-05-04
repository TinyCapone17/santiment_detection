import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1️⃣  путь к чек‑пойнту, который Trainer записал после обучения
#    └─ папка rubert_finetuned лежит рядом со скриптом
MODEL_PATH = "rubert_finetuned"          # <‑‑ заменили MODEL_NAME

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# 2️⃣  устройство: m‑процессор → 'mps', иначе CPU / CUDA:0
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)

def predict_sentiment(text: str, return_label: bool = True):
    """
    Определяет тональность текста:
      'NEGATIVE', 'NEUTRAL' или 'POSITIVE'
      либо массив вероятностей [p_neg, p_neu, p_pos]
    """
    if not isinstance(text, str) or not text.strip():
        return "NEUTRAL" if return_label else [0, 1, 0]

    with torch.no_grad():
        inputs = tokenizer(text,
                           return_tensors="pt",
                           truncation=True,
                           padding=True,
                           max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        logits = model(**inputs).logits
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    if return_label:
        return model.config.id2label[int(probs.argmax())]
    return probs.tolist()

def main():
    df = pd.read_csv("test.csv")   # ваш исходный файл
    df["Sentiment"] = df["Review"].apply(predict_sentiment)

    output_file = "finetuned_test.csv"     # 3️⃣ новое имя
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Сохранён файл: {output_file}")

if __name__ == "__main__":
    main()
