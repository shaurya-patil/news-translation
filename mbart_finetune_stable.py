"""
mBART-50 Multi-Language Fine-Tuning (Colab-Safe, Offline-Friendly)
------------------------------------------------------------------
Stable script with memory safety, error handling, and auto-cleanup.
"""

import os
import gc
import torch
import json
import traceback
import subprocess
import time
from typing import List, Dict
import inspect

import psutil
from datasets import load_dataset, Dataset, get_dataset_config_names
import sacrebleu

from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerCallback
)

# ============================================================
# SETUP
# ============================================================
try:
    subprocess.run([
        "pip", "install", "-q",
        "transformers", "datasets", "accelerate", "sentencepiece", "protobuf", "sacrebleu", "psutil"
    ], check=True)
except Exception as e:
    print(f"Warning: dependency installation failed — {e}")

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
OUTPUT_DIR = "./mbart_multilang_news"
SRC_LANG = "en_XX"
# Default language used for evaluation/generation; can be overridden when looping per-lang below
EVAL_TGT_LANG = "fr_XX"
MAX_LEN = 128
SAMPLES_PER_LANG = 50
VAL_SAMPLES = 20

LANGUAGES = {
    "es_XX": {"dataset_lang": "es"},
    "fr_XX": {"dataset_lang": "fr"},
    "de_DE": {"dataset_lang": "de"},
    "ru_RU": {"dataset_lang": "ru"},
    "ja_XX": {"dataset_lang": "ja"},
    "zh_CN": {"dataset_lang": "zh"},
    "it_IT": {"dataset_lang": "it"},
    "cs_CZ": {"dataset_lang": "cs"}
}

# ============================================================
# DEVICE CHECK
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

# ============================================================
# LOAD MODEL AND TOKENIZER
# ============================================================
print("Loading model...")
try:
    tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
    model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.to(device)
    tokenizer.src_lang = SRC_LANG
    # Do not force a single BOS at train time; we'll set it during eval per target language
    model.config.forced_bos_token_id = None
    model.config.use_cache = False  # needed for some optimizations during training
except Exception as e:
    raise RuntimeError(f"Model loading failed: {str(e)}")
print("Model and tokenizer ready.\n")

# ============================================================
# DATA PREPARATION
# ============================================================
print("Loading and preparing datasets...")

train_data, val_data = [], []

for tgt_lang, cfg in LANGUAGES.items():
    try:
        lang = cfg['dataset_lang']
        # Pick a valid config name (en-xx or xx-en)
        try:
            ds = load_dataset("news_commentary", name=f"en-{lang}", split="train")
        except Exception:
            try:
                available = get_dataset_config_names("news_commentary")
                alt = f"{lang}-en"
                if alt in available:
                    ds = load_dataset("news_commentary", name=alt, split="train")
                else:
                    raise ValueError(f"No config for en-{lang} or {lang}-en")
            except Exception:
                raise

        total = len(ds)
        train_size = min(SAMPLES_PER_LANG, max(0, total - VAL_SAMPLES))

        for i in range(train_size):
            item = ds[i]
            train_data.append({
                "source": item["translation"].get("en", ""),
                "target": item["translation"].get(lang, ""),
                "target_lang": tgt_lang
            })
        for i in range(train_size, train_size + VAL_SAMPLES):
            if i < total:
                item = ds[i]
                val_data.append({
                    "source": item["translation"].get("en", ""),
                    "target": item["translation"].get(lang, ""),
                    "target_lang": tgt_lang
                })
        del ds
        gc.collect()
    except Exception:
        print(f"Skipping {cfg['dataset_lang']} due to error:\n{traceback.format_exc(limit=1)}")

if not train_data:
    raise ValueError("No valid data loaded. Check dataset availability.")

train_dataset = Dataset.from_list(train_data).shuffle(seed=42)
val_dataset = Dataset.from_list(val_data).shuffle(seed=42)

print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}\n")

# ============================================================
# TOKENIZATION
# ============================================================
def preprocess(batch):
    try:
        sources = batch["source"]
        targets = batch["target"]
        target_langs = batch["target_lang"]

        # Encode sources with fixed source language
        model_inputs = tokenizer(
            sources,
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length"
        )

        # Encode targets language-aware per example so the correct lang token is added
        labels_batch = []
        for target, lang in zip(targets, target_langs):
            tokenizer.tgt_lang = lang
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    target,
                    max_length=MAX_LEN,
                    truncation=True,
                    padding="max_length"
                )
            labels_batch.append(labels["input_ids"])

        model_inputs["labels"] = labels_batch
        # Also keep target_lang for later per-language evaluation
        model_inputs["target_lang"] = target_langs
        return model_inputs
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return {"input_ids": [], "attention_mask": [], "labels": [], "target_lang": []}

print("Tokenizing datasets (this may take a few minutes)...")

tokenized_train = train_dataset.map(
    preprocess,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing train",
)

tokenized_val = val_dataset.map(
    preprocess,
    batched=True,
    remove_columns=val_dataset.column_names,
    desc="Tokenizing val",
)

del train_dataset, val_dataset
gc.collect()
torch.cuda.empty_cache()

print("Tokenization complete.\n")

# ============================================================
# TRAINING SETUP
# ============================================================
base_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Strip non-model keys before collation

def data_collator(features):
    for f in features:
        f.pop("target_lang", None)
    return base_collator(features)

# BLEU metric using sacrebleu
def postprocess_text(preds: List[str], labels: List[str]):
    preds = [p.strip() for p in preds]
    labels = [[l.strip()] for l in labels]
    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them
    labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in label] for label in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    bleu = sacrebleu.corpus_bleu(decoded_preds, decoded_labels)
    chrf = sacrebleu.corpus_chrf(decoded_preds, decoded_labels)
    # Exact match accuracy (string-level)
    exact_matches = sum(p == refs[0] for p, refs in zip(decoded_preds, decoded_labels))
    accuracy = 100.0 * exact_matches / max(1, len(decoded_preds))
    result = {
        "bleu": round(bleu.score, 2),
        "chrf": round(chrf.score, 2),
        "accuracy": round(accuracy, 2),
    }
    return result

# Build training args with backward-compatible kwargs filtering
sig_params = set(inspect.signature(Seq2SeqTrainingArguments.__init__).parameters.keys())
kwargs = {
    "output_dir": OUTPUT_DIR,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 16,
    "learning_rate": 2e-5,
    "num_train_epochs": 3,
    "weight_decay": 0.01,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "save_total_limit": 2,
    "fp16": torch.cuda.is_available(),
    "logging_steps": 50,
    "report_to": "none",
    "gradient_accumulation_steps": 2,
    "dataloader_pin_memory": False,
    "predict_with_generate": True,
    "generation_max_length": MAX_LEN,
    "generation_num_beams": 4,
    "load_best_model_at_end": True,
    "metric_for_best_model": "bleu",
    "greater_is_better": True,
}
# Fallback for older Transformers: drop unsupported keys
filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig_params}
# If evaluation_strategy not present, approximate with steps-based
if "evaluation_strategy" not in sig_params:
    filtered_kwargs.update({k: v for k, v in {"eval_steps": 500, "save_steps": 500}.items() if k in sig_params})
args = Seq2SeqTrainingArguments(**filtered_kwargs)

# Resource monitoring callback
class ResourceMonitor(TrainerCallback):
    def __init__(self):
        self.start_time = None
        self.train_runtime = None
        self.max_rss_bytes = 0
        self.max_cuda_reserved_bytes = 0
        self.max_cuda_allocated_bytes = 0
        self.process = psutil.Process(os.getpid())

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_log(self, args, state, control, logs=None, **kwargs):
        try:
            rss = self.process.memory_info().rss
            if rss > self.max_rss_bytes:
                self.max_rss_bytes = rss
            if torch.cuda.is_available():
                self.max_cuda_reserved_bytes = max(self.max_cuda_reserved_bytes, torch.cuda.max_memory_reserved())
                self.max_cuda_allocated_bytes = max(self.max_cuda_allocated_bytes, torch.cuda.max_memory_allocated())
        except Exception:
            pass

    def on_train_end(self, args, state, control, **kwargs):
        self.train_runtime = time.time() - (self.start_time or time.time())

monitor = ResourceMonitor()

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[monitor],
)

# ============================================================
# TRAINING
# ============================================================
try:
    print("Starting training...")
    train_result = trainer.train()
    # Persist train metrics
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    
    # Save consolidated training metrics to main directory
    training_metrics = {
        "train_metrics": train_result.metrics,
        "trainer_state": {
            "best_metric": trainer.state.best_metric,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "global_step": trainer.state.global_step,
            "epoch": trainer.state.epoch,
        },
        "log_history": trainer.state.log_history
    }
    with open("training_metrics.json", "w") as f:
        json.dump(training_metrics, f, indent=2)
    print("Training metrics saved to training_metrics.json\n")
    
    print("Training finished successfully.\n")
except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        print("⚠️ Out of memory detected. Try reducing batch size or samples.")
    else:
        print(f"Training error: {e}")
    torch.cuda.empty_cache()

# ============================================================
# EVALUATION PER TARGET LANGUAGE (generation with correct BOS)
# ============================================================
try:
    lang_code_to_id: Dict[str, int] = tokenizer.lang_code_to_id

    # Helper to filter dataset by target language
    def filter_by_lang(dataset, lang: str):
        return dataset.filter(lambda ex: ex["target_lang"] == lang)

    eval_langs = list(LANGUAGES.keys())
    per_lang_metrics = {}
    for lang in eval_langs:
        subset = filter_by_lang(tokenized_val, lang)
        if len(subset) == 0:
            continue
        print(f"Evaluating generation for {lang} on {len(subset)} samples...")
        # Temporarily set forced_bos_token_id for generation
        model.config.forced_bos_token_id = lang_code_to_id.get(lang, None)
        metrics = trainer.evaluate(eval_dataset=subset, metric_key_prefix=f"eval_{lang}")
        per_lang_metrics[lang] = metrics
        print({k: v for k, v in metrics.items() if any(k.endswith(suf) for suf in ["bleu", "chrf", "accuracy"])})

    # Save per-language metrics
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "per_language_eval.json"), "w") as f:
        json.dump(per_lang_metrics, f, indent=2)

    # Reset
    model.config.forced_bos_token_id = lang_code_to_id.get(EVAL_TGT_LANG, None)
except Exception as e:
    print(f"Evaluation error: {e}")

# ============================================================
# SAVE MODEL
# ============================================================
print("Saving model...")

os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save resource metrics
resource_metrics = {
    "train_runtime_sec": getattr(monitor, "train_runtime", None),
    "max_cpu_rss_mb": round(getattr(monitor, "max_rss_bytes", 0) / (1024**2), 2),
    "max_cuda_reserved_mb": round(getattr(monitor, "max_cuda_reserved_bytes", 0) / (1024**2), 2),
    "max_cuda_allocated_mb": round(getattr(monitor, "max_cuda_allocated_bytes", 0) / (1024**2), 2),
}
with open(os.path.join(OUTPUT_DIR, "resource_metrics.json"), "w") as f:
    json.dump(resource_metrics, f, indent=2)

config = {
    "languages_trained": list(LANGUAGES.keys()),
    "source_language": SRC_LANG,
    "samples_used": len(train_data),
    "dataset": "news_commentary",
    "method": "transfer_learning"
}

with open(f"{OUTPUT_DIR}/language_config.json", "w") as f:
    json.dump(config, f, indent=2)

del model, tokenizer, tokenized_train, tokenized_val
gc.collect()
torch.cuda.empty_cache()

print("Model saved successfully.\n")

# ============================================================
# PACKAGE FOR DOWNLOAD
# ============================================================
print("Creating ZIP package...")
try:
    import zipfile
    with zipfile.ZipFile("mbart_multilang_news.zip", "w") as zipf:
        for root, _, files in os.walk(OUTPUT_DIR):
            for file in files:
                zipf.write(os.path.join(root, file),
                           os.path.relpath(os.path.join(root, file), OUTPUT_DIR))
    print("Package ready: mbart_multilang_news.zip")
except Exception as e:
    print(f"ZIP creation failed: {e}")
