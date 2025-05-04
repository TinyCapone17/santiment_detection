import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate, numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("data_with_sentiment.csv")

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

ts = int(len(df) * 0.7)   # 70 %

tr_df = df.iloc[:ts].reset_index(drop=True)
ts_df  = df.iloc[ts:].reset_index(drop=True)

ts_df.to_csv("test.csv")

df = tr_df

# убеждаемся, что метки в формате int32 и соответствуют id2label модели:
label_map = {-1: 0, 0: 1, 1: 2}  # NEG, NEU, POS
df["label"] = df["Sentiment"].map(label_map)
assert not df["label"].isna().any(), "Есть неизвестные метки!"


# Перевод в Dataset и разбиение 70 / 30
train_df, val_df = train_test_split(
    df[["Review", "label"]], 
    test_size=0.3, 
    random_state=42, 
    stratify=df["label"],
)

ds_train = Dataset.from_pandas(train_df.reset_index(drop=True))
ds_val   = Dataset.from_pandas(val_df.reset_index(drop=True))


# Токенизация
MODEL_NAME = "cointegrated/rubert-tiny-sentiment-balanced"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["Review"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

ds_train = ds_train.map(tokenize, batched=True, remove_columns=["Review"])
ds_val   = ds_val.map(tokenize, batched=True, remove_columns=["Review"])

ds_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
ds_val.set_format(**ds_train.format)  # тот же формат

#  Конфигурация обучения

metric_acc = evaluate.load("accuracy")
metric_f1  = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": metric_acc.compute(
            predictions=preds, references=labels
        )["accuracy"],
        "f1": metric_f1.compute(
            predictions=preds, references=labels,
            average="macro"
        )["f1"],
    }

device = "mps" if torch.backends.mps.is_available() else "cpu"
model  = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)

args = TrainingArguments(
    output_dir          = "rubert_finetuned",
    eval_strategy = "epoch",
    save_strategy       = "epoch",
    load_best_model_at_end = True,
    metric_for_best_model  = "f1",
    num_train_epochs    = 20,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size  = 32,
    learning_rate       = 2e-5,
    weight_decay        = 0.01,
    warmup_ratio        = 0.1,
    logging_steps       = 50,
    gradient_accumulation_steps = 2,  # если памяти мало
    fp16                = False,      # на GPU можно включить
)

# Запуск обучения

trainer = Trainer(
    model          = model,
    args           = args,
    train_dataset  = ds_train,
    eval_dataset   = ds_val,
    compute_metrics= compute_metrics,
)

trainer.train()
trainer.save_model("rubert_finetuned")
tokenizer.save_pretrained("rubert_finetuned")



# hist = pd.DataFrame(trainer.state.log_history)


# hist = hist.dropna(subset=["epoch"])

# ── график ────────────────────────────────────────────────────────────────────
# plt.figure(figsize=(8, 5))
# plt.plot(hist["epoch"], hist["loss"],         label="train loss")
# plt.plot(hist["epoch"], hist["eval_loss"],    label="val  loss")
# plt.plot(hist["epoch"], hist["eval_accuracy"],label="val accuracy")
# plt.plot(hist["epoch"], hist["eval_f1"],      label="val F1‑macro")

# plt.xlabel("Epoch")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()

# out_path = Path("rubert_finetuned") / "training_curve.png"
# plt.savefig(out_path, dpi=150)
# plt.close()

# print(f"Кривые обучения сохранены: {out_path}")
