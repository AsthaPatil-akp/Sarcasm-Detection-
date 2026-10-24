"""Train the sentiment BiLSTM and compare it with the saved financial baseline.

Run from the project root:
    py -3.12 sentiment/train.py

Uses TensorFlow, which is already installed for this project. PyTorch is not
installed: adding it would change the shared Python environment and can break
the sarcasm detector. The network is still a trainable-embedding BiLSTM.
"""
import json
import os
import random
import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from data_lib import HEADLINES_PATH, build
from text import tokenize_many

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
MODEL_DIR = os.path.join(HERE, "models")
RESULTS_DIR = os.path.join(HERE, "results")
GLOVE_PATH = os.path.join(ROOT, "embeddings", "glove.6B.100d.txt")

SEED = 42
LABELS = ["negative", "neutral", "positive"]
MAX_VOCAB = 12000
MIN_FREQ = 2
EMBED_DIM = 100
LSTM_UNITS = 64
BATCH_SIZE = 32
EPOCHS = 20
PATIENCE = 3
LEARNING_RATE = 1e-3
# SEED fixes Python, NumPy, and TensorFlow randomness for this run.
# LABELS is the softmax order: index 0 negative, 1 neutral, 2 positive.
# MAX_VOCAB caps word types. MIN_FREQ drops a word seen in only one training headline.
# EMBED_DIM is 100 to match glove.6B.100d. LSTM_UNITS is the size of each LSTM direction.
# BATCH_SIZE is how many headlines Adam updates on at once.
# EPOCHS is the planned maximum. Training can stop earlier.
# PATIENCE is how many epochs in a row may fail to beat the best validation loss.
# LEARNING_RATE is Adam's starting step size (0.001).


def set_seed(seed):
    # Same seed for the three libraries that draw random numbers during this run.
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_vocab(token_lists):
    """Document frequency on training headlines only. 0 is padding, 1 is unknown."""
    # Count each word once per headline, then keep words that clear MIN_FREQ.
    counts = {}
    for tokens in token_lists:
        for token in set(tokens):
            counts[token] = counts.get(token, 0) + 1
    # Most common words first. Ids start at 2 so 0 and 1 stay reserved.
    ranked = [
        word for word, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if counts[word] >= MIN_FREQ
    ][: MAX_VOCAB - 2]
    return {word: index for index, word in enumerate(ranked, start=2)}


def encode(token_lists, word_index, max_len):
    # One row per headline, padded with 0 on the right. Words past max_len are cut.
    sequences = np.zeros((len(token_lists), max_len), dtype="int32")
    for row, tokens in enumerate(token_lists):
        # A word missing from the training vocabulary becomes id 1 (unknown).
        ids = [word_index.get(token, 1) for token in tokens][:max_len]
        if ids:
            sequences[row, : len(ids)] = ids
    return sequences


def choose_max_len(token_lists):
    # Pad length is the 95th percentile of training lengths, kept between 16 and 60.
    lengths = [len(tokens) for tokens in token_lists if tokens]
    if not lengths:
        return 30
    return int(min(60, max(16, np.percentile(lengths, 95))))


def load_glove(word_index):
    # Rows start as small random vectors. Row 0 stays zeros because id 0 is padding.
    matrix = np.random.normal(0, 0.05, (len(word_index) + 2, EMBED_DIM)).astype("float32")
    matrix[0] = 0.0
    if not os.path.exists(GLOVE_PATH):
        return matrix, 0, False
    # Copy a GloVe row only when that word is in this vocabulary. The file is read, not edited.
    wanted = set(word_index)
    found = 0
    with open(GLOVE_PATH, encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip().split(" ")
            if len(parts) != EMBED_DIM + 1 or parts[0] not in wanted:
                continue
            matrix[word_index[parts[0]]] = np.asarray(parts[1:], dtype="float32")
            found += 1
            if found == len(wanted):
                break
    return matrix, found, True


def build_model(vocab_size, max_len, embedding_matrix):
    # Word ids -> 100-d embedding (GloVe start, still trainable, padding masked)
    # -> spatial dropout 0.3 (drops a whole word vector; off when predicting)
    # -> bidirectional LSTM, 64 units each way, concatenated to 128.
    #    The cell output uses tanh. The input, forget, and output gates use sigmoid.
    # -> dropout 0.4 -> dense 64 with ReLU -> dropout 0.3
    # -> dense 3 with softmax (negative, neutral, positive).
    model = keras.Sequential([
        layers.Input(shape=(max_len,), dtype="int32"),
        layers.Embedding(
            vocab_size,
            EMBED_DIM,
            embeddings_initializer=keras.initializers.Constant(embedding_matrix),
            mask_zero=True,
            trainable=True,
            name="embedding",
        ),
        layers.SpatialDropout1D(0.3, name="embed_dropout"),
        layers.Bidirectional(layers.LSTM(LSTM_UNITS), name="bilstm"),
        layers.Dropout(0.4, name="bilstm_dropout"),
        layers.Dense(64, activation="relu", name="hidden"),
        layers.Dropout(0.3, name="hidden_dropout"),
        layers.Dense(len(LABELS), activation="softmax", name="sentiment"),
    ], name="sentiment_bilstm")
    # Adam uses Keras defaults β1 = 0.9 and β2 = 0.999. Loss compares the softmax
    # with the integer class id (0, 1, or 2), which is sparse categorical cross-entropy.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def set_learning_rate(model, value):
    # Update Adam's step size before the next epoch. assign() writes into the existing tensor.
    learning_rate = model.optimizer.learning_rate
    if hasattr(learning_rate, "assign"):
        learning_rate.assign(float(value))
    else:
        model.optimizer.learning_rate = float(value)


def metric_block(gold, predicted):
    # Accuracy, macro-F1, per-class precision/recall/F1, and the 3x3 confusion matrix.
    # Macro-F1 averages the three classes equally, so the large neutral class does not hide the others.
    precision, recall, f1, support = precision_recall_fscore_support(
        gold, predicted, labels=list(range(len(LABELS))), zero_division=0
    )
    matrix = confusion_matrix(gold, predicted, labels=list(range(len(LABELS))))
    return {
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_f1": float(f1_score(gold, predicted, average="macro", zero_division=0)),
        "per_class": [
            {
                "label": LABELS[index],
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in range(len(LABELS))
        ],
        "confusion_matrix": matrix.tolist(),
        "report": classification_report(
            gold, predicted, target_names=LABELS, digits=4, zero_division=0
        ),
    }


def backup_baseline():
    # Copy the previous checkpoint once, and only when the backup name does not exist yet.
    # Training later overwrites sentiment_bilstm.keras. This copy is the financial baseline.
    os.makedirs(MODEL_DIR, exist_ok=True)
    pairs = [
        ("sentiment_bilstm.keras", "baseline_financial.keras"),
        ("vocab.json", "baseline_vocab.json"),
        ("config.json", "baseline_config.json"),
    ]
    for source_name, dest_name in pairs:
        source = os.path.join(MODEL_DIR, source_name)
        dest = os.path.join(MODEL_DIR, dest_name)
        if os.path.exists(source) and not os.path.exists(dest):
            shutil.copy2(source, dest)


def exploratory(frame, token_lists):
    # Class counts, headline-length percentiles, and the 12 most common tokens per class.
    # Saved as eda.json. These counts are not model inputs.
    lengths = [len(tokens) for tokens in token_lists]
    counts = {}
    for tokens, label in zip(token_lists, frame["sentiment"]):
        bucket = counts.setdefault(label, {})
        for token in set(tokens):
            bucket[token] = bucket.get(token, 0) + 1
    top = {
        label: [word for word, _count in sorted(bucket.items(), key=lambda item: -item[1])[:12]]
        for label, bucket in counts.items()
    }
    return {
        "rows": int(len(frame)),
        "class_counts": {label: int((frame["sentiment"] == label).sum()) for label in LABELS},
        "length_mean": float(np.mean(lengths)) if lengths else 0.0,
        "length_p50": float(np.percentile(lengths, 50)) if lengths else 0.0,
        "length_p95": float(np.percentile(lengths, 95)) if lengths else 0.0,
        "top_tokens_by_class": top,
    }


def score_saved_split(model, word_index, max_len, csv_path):
    # Tokenise the locked test headlines, predict a class, and score against the human gold label.
    frame = pd.read_csv(csv_path)
    tokens = tokenize_many(frame["headline"].tolist())
    predicted = model.predict(encode(tokens, word_index, max_len), batch_size=256, verbose=0).argmax(axis=1)
    gold = frame["gold"].map({label: index for index, label in enumerate(LABELS)}).to_numpy()
    block = metric_block(gold, predicted)
    frame = frame.copy()
    frame["improved_predicted"] = [LABELS[int(index)] for index in predicted]
    return block, frame


def sarcasm_ablation(model, word_index, max_len, validation):
    """See whether the unchanged sarcasm probability helps sentiment on validation rows."""
    # One validation sample. Half is used to fit a small logistic step, half is used to score it.
    sample = validation.sample(n=min(160, len(validation)), random_state=SEED)
    try:
        # Read-only call into the existing sarcasm detector. Its files are not written.
        import sys
        sys.path.insert(0, os.path.join(ROOT, "app"))
        from sarcasm_inference import SarcasmDetector
        detector = SarcasmDetector(os.path.join(ROOT, "models"))
    except Exception as exc:
        return {"status": "skipped", "reason": str(exc)}

    # Sentiment softmax, then the sarcasm probability as a fourth number beside those three.
    probabilities = model.predict(
        encode(tokenize_many(sample["headline"].tolist()), word_index, max_len),
        batch_size=128, verbose=0,
    )
    sarcasm_probability = []
    for headline in sample["headline"]:
        sarcasm_probability.append(float(detector.predict(headline, threshold=0.5)["probability"]))
    sarcasm_probability = np.asarray(sarcasm_probability)
    gold = sample["sentiment"].map({label: index for index, label in enumerate(LABELS)}).to_numpy()
    features = np.column_stack([probabilities, sarcasm_probability])
    cut = len(sample) // 2
    if cut < 20 or len(sample) - cut < 20:
        return {"status": "skipped", "reason": "Validation sample is too small to split."}

    # Sentiment-only macro-F1 on the held-out half, then the same half with sarcasm probability added.
    sentiment_only = float(f1_score(gold[cut:], probabilities[cut:].argmax(axis=1), average="macro", zero_division=0))
    try:
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(max_iter=400)
        classifier.fit(features[:cut], gold[:cut])
        combined = float(f1_score(gold[cut:], classifier.predict(features[cut:]), average="macro", zero_division=0))
    except Exception as exc:
        return {"status": "failed", "reason": str(exc), "sentiment_only_macro_f1": sentiment_only}

    result = {
        "status": "evaluated",
        "validation_rows": int(len(sample)),
        "eval_rows": int(len(sample) - cut),
        "sentiment_only_macro_f1": sentiment_only,
        "with_sarcasm_probability_macro_f1": combined,
        "improved": bool(combined >= sentiment_only + 0.02),
        "note": (
            "Fit on one half of a validation sample and scored on the other half. "
            "The sarcasm detector was only called. Its weights and threshold were not changed. "
            "Sarcasm probability is not treated as a sentiment label."
        ),
    }
    # Keep the logistic weights only when sarcasm probability lifts macro-F1 by at least 0.02.
    if result["improved"]:
        result["coef"] = classifier.coef_.tolist()
        result["intercept"] = classifier.intercept_.tolist()
    high = sarcasm_probability[cut:] >= 0.5
    result["eval_rows_sarcasm_probability_at_least_0.5"] = int(high.sum())
    if high.sum() >= 8:
        result["sentiment_only_macro_f1_on_high_sarcasm_slice"] = float(f1_score(
            gold[cut:][high], probabilities[cut:][high].argmax(axis=1), average="macro", zero_division=0
        ))
    return result


def main():
    # Step 1. Fix the seed and refuse to train if the baseline test files are missing.
    set_seed(SEED)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for name in ("baseline_fpb_test.csv", "baseline_semeval_test.csv"):
        if not os.path.exists(os.path.join(RESULTS_DIR, name)):
            raise SystemExit("Missing " + name + ". Score the original model before replacing it.")
    backup_baseline()
    # Step 2. Load the saved headline table. force=False does not rebuild it.
    build(force=False)

    # Step 3. Train and validation only. fpb_test and semeval_test stay out of fitting.
    frame = pd.read_csv(HEADLINES_PATH)
    train_df = frame[frame["split"] == "train"].reset_index(drop=True)
    val_df = frame[frame["split"] == "validation"].reset_index(drop=True)
    print(f"train {len(train_df)}  validation {len(val_df)}")

    # Step 4. Tokenise, build the vocabulary from training headlines, pad, and load GloVe.
    train_tokens = tokenize_many(train_df["headline"].tolist())
    val_tokens = tokenize_many(val_df["headline"].tolist())
    word_index = build_vocab(train_tokens)
    max_len = choose_max_len(train_tokens)
    embedding_matrix, glove_hits, glove_used = load_glove(word_index)
    label_to_id = {label: index for index, label in enumerate(LABELS)}
    x_train = encode(train_tokens, word_index, max_len)
    x_val = encode(val_tokens, word_index, max_len)
    y_train = train_df["sentiment"].map(label_to_id).to_numpy()
    y_val = val_df["sentiment"].map(label_to_id).to_numpy()
    # Balanced class weights are computed on the training labels only.
    # A rare class (negative) gets a weight above 1 so each of its mistakes counts more.
    class_weights = compute_class_weight("balanced", classes=np.arange(len(LABELS)), y=y_train)
    class_weight = {index: float(weight) for index, weight in enumerate(class_weights)}

    with open(os.path.join(RESULTS_DIR, "eda.json"), "w", encoding="utf-8") as handle:
        json.dump(exploratory(train_df, train_tokens), handle, indent=2)

    # Step 5. Build the network. Weights are random except the embedding, which starts at GloVe.
    model = build_model(len(word_index) + 2, max_len, embedding_matrix)
    history = []
    best_loss = float("inf")
    best_weights = None
    best_epoch = 1
    wait = 0
    learning_rate = LEARNING_RATE
    stop_reason = f"Completed {EPOCHS} epochs."
    # Step 6. Epoch loop. fit() is called for one epoch at a time so the rate and the
    # stop decision can be written down after every pass.
    # Inside each fit() call Keras does the forward pass, sparse categorical cross-entropy,
    # and backpropagation with Adam. Dropout is on here. It is off later, at prediction.
    for epoch in range(1, EPOCHS + 1):
        # Record the rate this epoch actually used. The next rate is chosen after the val loss.
        used_rate = learning_rate
        set_learning_rate(model, used_rate)
        fitted = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=1,
            batch_size=BATCH_SIZE,
            class_weight=class_weight,
            verbose=2,
        )
        # Validation loss is the number early stopping watches. Macro-F1 is logged beside it.
        val_loss = float(fitted.history["val_loss"][0])
        val_pred = model.predict(x_val, batch_size=256, verbose=0).argmax(axis=1)
        val_f1 = float(f1_score(y_val, val_pred, average="macro", zero_division=0))
        decision = "Validation loss improved."
        # A new best needs val loss lower by more than 0.0001. Those weights are copied aside.
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_weights = [item.copy() for item in model.get_weights()]
            best_epoch = epoch
            wait = 0
        else:
            # No improvement: halve the learning rate (never below 1e-5) and add one to the wait.
            wait += 1
            next_rate = max(learning_rate * 0.5, 1e-5)
            decision = (
                f"Validation loss did not improve. Learning rate moves from {learning_rate:.6f} "
                f"to {next_rate:.6f}. Early-stopping wait {wait} of {PATIENCE}."
            )
            learning_rate = next_rate
            # Three misses in a row ends the loop. The saved weights are still the best epoch.
            if wait >= PATIENCE:
                decision += f" Stop. Weights return to epoch {best_epoch}."
                stop_reason = decision
        history.append({
            "epoch": epoch,
            "loss": float(fitted.history["loss"][0]),
            "accuracy": float(fitted.history["accuracy"][0]),
            "val_loss": val_loss,
            "val_accuracy": float(fitted.history["val_accuracy"][0]),
            "val_macro_f1": val_f1,
            "learning_rate": used_rate,
            "decision": decision,
        })
        print(f"epoch {epoch} val_loss {val_loss:.4f} val_macro_f1 {val_f1:.4f} {decision}")
        if wait >= PATIENCE:
            break

    # Step 7. Put the best epoch's weights back. Later epochs are discarded even if their
    # training loss was lower. Then write the checkpoint, vocabulary, and config.
    model.set_weights(best_weights)
    model.save(os.path.join(MODEL_DIR, "sentiment_bilstm.keras"))
    with open(os.path.join(MODEL_DIR, "vocab.json"), "w", encoding="utf-8") as handle:
        json.dump(word_index, handle)
    config = {
        "labels": LABELS,
        "max_len": max_len,
        "embed_dim": EMBED_DIM,
        "lstm_units": LSTM_UNITS,
        "vocab_size": len(word_index) + 2,
        "pad_id": 0,
        "oov_id": 1,
        "min_freq": MIN_FREQ,
        "seed": SEED,
        "framework": "tensorflow-keras",
        "framework_reason": (
            "PyTorch is not installed. Installing it would change numpy in the same "
            "environment the sarcasm detector uses. This BiLSTM uses the TensorFlow already installed."
        ),
        "glove_used": glove_used,
        "glove_words_found": int(glove_hits),
        "embedding_trainable": True,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "loss": "sparse_categorical_crossentropy",
        "class_weight": {LABELS[index]: weight for index, weight in class_weight.items()},
        "batch_size": BATCH_SIZE,
        "epochs_planned": EPOCHS,
        "epochs_ran": len(history),
        "best_epoch": best_epoch,
        "early_stopping_patience": PATIENCE,
        "stop_reason": stop_reason,
        "dropout": {"embedding": 0.3, "after_bilstm": 0.4, "after_dense": 0.3},
        "activations": {"lstm_output": "tanh", "lstm_gates": "sigmoid", "hidden": "relu", "output": "softmax"},
        "split": {
            "train": int(len(train_df)),
            "validation": int(len(val_df)),
            "fractions": (
                "FinancialPhraseBank keeps its original 70/15/15 split. "
                "SemEval headlines and EmoBank newspaper sentences each use their own 70/15/15 split."
            ),
        },
    }
    with open(os.path.join(MODEL_DIR, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    with open(os.path.join(RESULTS_DIR, "history.json"), "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    # Step 8. Score the two locked tests. Baseline rows were predicted by the first model
    # and saved in the CSVs. The improved rows are predicted by the weights just restored.
    comparison = {"baseline": {}, "improved": {}}
    reports = []
    error_frames = []
    for name, path in (
        ("fpb_test", os.path.join(RESULTS_DIR, "baseline_fpb_test.csv")),
        ("semeval_test", os.path.join(RESULTS_DIR, "baseline_semeval_test.csv")),
    ):
        saved = pd.read_csv(path)
        base_gold = saved["gold"].map(label_to_id).to_numpy()
        base_pred = saved["predicted"].map(label_to_id).to_numpy()
        comparison["baseline"][name] = metric_block(base_gold, base_pred)
        improved, detailed = score_saved_split(model, word_index, max_len, path)
        comparison["improved"][name] = {key: value for key, value in improved.items() if key != "report"}
        reports.append(name + "\n" + improved["report"])
        detailed["test"] = name
        # Keep up to 20 headlines where the new model disagrees with the human label.
        mistakes = detailed[detailed["gold"] != detailed["improved_predicted"]]
        error_frames.append(mistakes.head(20))
        matrix = np.asarray(improved["confusion_matrix"])
        pd.DataFrame(
            matrix,
            index=[f"gold_{label}" for label in LABELS],
            columns=[f"pred_{label}" for label in LABELS],
        ).to_csv(os.path.join(RESULTS_DIR, f"confusion_matrix_{name}.csv"))
        print(name, "baseline", round(comparison["baseline"][name]["macro_f1"], 4),
              "improved", round(improved["macro_f1"], 4))

    news_test = frame[frame["split"] == "news_test"].copy()
    if len(news_test):
        news_test["gold"] = news_test["sentiment"]
        news_path = os.path.join(RESULTS_DIR, "news_test.csv")
        news_test[["headline", "gold"]].to_csv(news_path, index=False)
        improved, detailed = score_saved_split(model, word_index, max_len, news_path)
        comparison["improved"]["news_test"] = {key: value for key, value in improved.items() if key != "report"}
        reports.append("news_test\n" + improved["report"])
        detailed["test"] = "news_test"
        error_frames.append(detailed[detailed["gold"] != detailed["improved_predicted"]].head(20))
        matrix = np.asarray(improved["confusion_matrix"])
        pd.DataFrame(
            matrix,
            index=[f"gold_{label}" for label in LABELS],
            columns=[f"pred_{label}" for label in LABELS],
        ).to_csv(os.path.join(RESULTS_DIR, "confusion_matrix_news_test.csv"))
        print("news_test", "improved", round(improved["macro_f1"], 4), "n", len(news_test))

    comparison["baseline"] = {
        name: {key: value for key, value in block.items() if key != "report"}
        for name, block in comparison["baseline"].items()
    }
    with open(os.path.join(RESULTS_DIR, "comparison.json"), "w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2)
    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(reports))
    if error_frames:
        pd.concat(error_frames, ignore_index=True).to_csv(
            os.path.join(RESULTS_DIR, "error_analysis.csv"), index=False
        )

    # Step 9. Check sarcasm probability on validation only. It is saved as a calibrator
    # only when that check improves macro-F1. Otherwise any old calibrator file is removed.
    ablation = sarcasm_ablation(model, word_index, max_len, val_df)
    with open(os.path.join(RESULTS_DIR, "sarcasm_feature_ablation.json"), "w", encoding="utf-8") as handle:
        json.dump(ablation, handle, indent=2)
    calibrator_path = os.path.join(MODEL_DIR, "sarcasm_calibrator.json")
    if ablation.get("improved") and "coef" in ablation:
        with open(calibrator_path, "w", encoding="utf-8") as handle:
            json.dump({"coef": ablation["coef"], "intercept": ablation["intercept"]}, handle)
    elif os.path.exists(calibrator_path):
        os.remove(calibrator_path)
    print("ablation", json.dumps({key: value for key, value in ablation.items() if key not in {"coef", "intercept"}}))
    print(stop_reason)

    # Step 10. Plot train loss and validation loss against epoch. Agg writes a file, no window.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.plot([row["epoch"] for row in history], [row["loss"] for row in history], label="train loss")
        axis.plot([row["epoch"] for row in history], [row["val_loss"] for row in history], label="validation loss")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Cross-entropy")
        axis.legend()
        figure.tight_layout()
        figure.savefig(os.path.join(RESULTS_DIR, "loss_curves.png"), dpi=140)
        plt.close(figure)
    except Exception as exc:
        with open(os.path.join(RESULTS_DIR, "plot_error.txt"), "w", encoding="utf-8") as handle:
            handle.write(str(exc))


if __name__ == "__main__":
    main()
