import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "cointegrated/rubert-tiny-sentiment-balanced"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# конфиги запуска цпу для м-процессоров мак
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
model.to(device)

def predict_sentiment(text: str, return_label=True):
    """
    Определяет тональность текста:
    - Если return_label=True, возвращает 'NEGATIVE', 'NEUTRAL' или 'POSITIVE'
    - Если return_label=False, возвращает массив вероятностей [p_neg, p_neu, p_pos]
    """
    if not isinstance(text, str) or not text.strip():
        return "NEUTRAL" if return_label else [0, 1, 0]

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]  # [p_neg, p_neu, p_pos]

    if return_label:
        label_id = probs.argmax()
        label = model.config.id2label[label_id]  # 'NEGATIVE', 'NEUTRAL', 'POSITIVE'
        return label
    else:
        return probs.tolist()

def main():
    df = pd.read_csv("collected_data.csv")

    df["Sentiment"] = df["Review"].apply(lambda x: predict_sentiment(x, return_label=True))

    output_file = "rubert_collected_data_with_sentiment.csv"
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Сохранён файл: {output_file}")

if __name__ == "__main__":
    main()
