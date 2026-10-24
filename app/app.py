"""
Sarcasm Detector - Streamlit demo
Run from the project root:  streamlit run app/app.py
"""
import json
import math
import os
import html
import sys

import pandas as pd
import streamlit as st

from sarcasm_inference import SarcasmDetector

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
SENTIMENT_DIR = os.path.join(ROOT, "sentiment", "models")
_SENTIMENT_CODE = os.path.join(ROOT, "sentiment")
if _SENTIMENT_CODE not in sys.path:
    sys.path.append(_SENTIMENT_CODE)

st.set_page_config(page_title="Sarcasm Detector", page_icon="🙃", layout="centered")


@st.cache_resource(show_spinner="Loading model and NLP pipeline...")
def load_detector():
    return SarcasmDetector(MODELS_DIR)


det = load_detector()


@st.cache_resource(show_spinner="Loading sentiment model...")
def load_sentiment():
    from sentiment_inference import SentimentAnalyzer
    return SentimentAnalyzer(SENTIMENT_DIR)


def get_sentiment():
    try:
        return load_sentiment(), None
    except Exception as exc:
        return None, str(exc)


def _stage1_decision(index, best_index, learning_rates):
    wait = index - best_index
    if wait < 0:
        return "Validation loss improved, so training continues."
    if wait == 0:
        return "Lowest validation loss in stage 1. Early stopping later restores these weights."
    parts = [f"Validation loss did not beat epoch {best_index + 1}. Early-stopping wait: {wait} of 3."]
    if index + 1 < len(learning_rates) and learning_rates[index + 1] < learning_rates[index] - 1e-12:
        parts.append(
            "ReduceLROnPlateau has patience 1, so one epoch without improvement halves "
            f"the learning rate from {learning_rates[index]:.6f} to {learning_rates[index + 1]:.6f} "
            "for the next epoch."
        )
    if wait >= 3:
        parts.append(
            "Patience is 3, so training stops. The weights are put back to epoch "
            f"{best_index + 1}: training loss was still falling, but validation loss had started "
            "rising, which means the extra epochs were memorising the training headlines. "
            "The learning rate would have been halved once more, and there is no further epoch to use it."
        )
    return " ".join(parts)


def _show_epoch_log(title, rows):
    st.markdown(f"**{title}**")
    lines = []
    for row in rows:
        lines.append(f"Epoch {row['epoch']}/{row['planned']}    {row['steps']}")
        lines.append(f"accuracy: {row['accuracy']:.4f}    loss: {row['loss']:.4f}")
        lines.append(
            f"val_accuracy: {row['val_accuracy']:.4f}    val_loss: {row['val_loss']:.4f}"
        )
        lines.append(f"learning_rate: {row['learning_rate']:.6f}")
        if row.get("decision"):
            lines.append("decision: " + row["decision"])
        lines.append("")
    st.code("\n".join(lines).rstrip(), language=None)


def _chart_labels(labels):
    seen = {}
    unique = []
    for label in labels:
        text = str(label)
        seen[text] = seen.get(text, 0) + 1
        unique.append(text if seen[text] == 1 else f"{text} ({seen[text]})")
    return unique


def _bar_chart(title, note, labels, values, y_label):
    return {
        "kind": "bar",
        "title": title,
        "note": note,
        "labels": _chart_labels(labels),
        "values": [float(value) for value in values],
        "y": y_label,
    }


def _line_chart(title, note, epochs, series, y_label):
    return {
        "kind": "line",
        "title": title,
        "note": note,
        "x": [int(epoch) for epoch in epochs],
        "series": [(name, [float(value) for value in values]) for name, values in series],
        "y": y_label,
    }


def _show_charts(charts):
    for chart in charts:
        if not chart.get("values") and not chart.get("series"):
            continue
        st.markdown(f"**{chart['title']}**")
        st.caption(chart["note"])
        if chart["kind"] == "bar":
            frame = pd.DataFrame({"item": chart["labels"], "value": chart["values"]})
            st.bar_chart(
                frame, x="item", y="value",
                x_label=chart["y"], y_label="",
                horizontal=True, sort=False,
            )
        else:
            import altair as alt
            rows = []
            for name, values in chart["series"]:
                for epoch, value in zip(chart["x"], values):
                    rows.append({"epoch": epoch, "series": name, "value": value})
            present = [row["value"] for row in rows]
            low, high = min(present), max(present)
            span = high - low or abs(high) or 1.0
            drawn = alt.Chart(pd.DataFrame(rows)).mark_line(point=True).encode(
                x=alt.X("epoch:O", title="Epoch", axis=alt.Axis(labelAngle=0), sort=None),
                y=alt.Y(
                    "value:Q", title=chart["y"],
                    scale=alt.Scale(domain=[min(0.0, low), high + span * 0.08]),
                ),
                color=alt.Color("series:N", title=None),
            )
            st.altair_chart(drawn, width="stretch")


def sarcasm_pipeline_charts(result):
    """Graphs from this headline's trace and the saved sarcasm training log."""
    trace = result.get("trace") or {}
    features = result.get("features") or {}
    charts = []
    pos_rows = trace.get("pos_rows") or []
    if pos_rows:
        charts.append(_bar_chart(
            "Part-of-speech ratios for this headline",
            "Each bar is that tag's count divided by the number of tagged tokens. Same ratios as the table.",
            [row["pos"] for row in pos_rows],
            [row["ratio"] for row in pos_rows],
            "Ratio",
        ))
    hits = trace.get("vader_hits") or []
    if hits:
        charts.append(_bar_chart(
            "VADER lexicon score of each hit",
            "Only words found in the VADER lexicon. The number is the lexicon value for that word.",
            [row["word"] for row in hits],
            [row["lexicon_score"] for row in hits],
            "Lexicon score",
        ))
    if features:
        charts.append(_bar_chart(
            "VADER outputs used by the sarcasm model",
            "These are the four VADER features stored for this headline and passed into the network.",
            ["positive", "negative", "neutral", "compound"],
            [features["vader_pos"], features["vader_neg"], features["vader_neu"], features["vader_compound"]],
            "VADER score",
        ))
    tfidf_rows = trace.get("tfidf_rows") or []
    if tfidf_rows:
        charts.append(_bar_chart(
            "TF-IDF weight of each term in this headline",
            "Same tfidf values as the table, largest weight first. idf is not plotted.",
            [row["term"] for row in tfidf_rows],
            [row["tfidf"] for row in tfidf_rows],
            "tfidf",
        ))
    attention = list(result.get("attention") or [])
    words = trace.get("word_rows") or []
    labels, weights = [], []
    for index, row in enumerate(words):
        if index >= len(attention) or row.get("token") == "(pad)":
            continue
        weight = float(attention[index])
        if weight == 0.0:
            continue
        labels.append(row["token"])
        weights.append(weight)
    if weights:
        charts.append(_bar_chart(
            "Attention weight of each word",
            "Softmax attention on the real words. Padding weights are exactly 0 and are omitted. "
            f"Plotted weights sum to {sum(weights):.4f}.",
            labels, weights, "Attention",
        ))
    info_path = os.path.join(MODELS_DIR, "training_info.json")
    if os.path.exists(info_path):
        with open(info_path, encoding="utf-8") as handle:
            history = json.load(handle)["history"]
        epochs = list(range(1, len(history["loss"]) + 1))
        charts.append(_line_chart(
            "Stage 1 training loss and validation loss",
            "Stage 1, GloVe frozen. Values are the saved epoch log. Early stopping uses validation loss.",
            epochs,
            [("train loss", history["loss"]), ("validation loss", history["val_loss"])],
            "Binary cross-entropy",
        ))
        charts.append(_line_chart(
            "Stage 1 learning rate",
            "The rate used on each stage-1 epoch. It is halved after an epoch that does not improve validation loss.",
            epochs,
            [("learning rate", history["learning_rate"])],
            "Learning rate",
        ))
    charts.append(_line_chart(
        "Stage 2 training loss and validation loss",
        "Stage 2, GloVe unfrozen, two epochs. Same losses as the stage-2 epoch log. Epoch 1 was kept.",
        [1, 2],
        [("train loss", [0.2340, 0.2212]), ("validation loss", [0.2952, 0.2963])],
        "Binary cross-entropy",
    ))
    return charts


def _sentiment_charts(word_rows, probabilities, labels, history):
    charts = []
    if word_rows:
        charts.append(_bar_chart(
            "Embedding length of each kept token",
            "Length means the Euclidean size of that token's 100-number embedding. Same lengths as the table. Padding is not plotted.",
            [row[1] for row in word_rows],
            [float(row[3]) for row in word_rows],
            "Embedding length",
        ))
    if probabilities and labels:
        charts.append(_bar_chart(
            "Softmax probability of each class",
            "These are the three probabilities shown beside the sarcasm verdict. The class is the tallest bar. There is no cutoff.",
            list(labels),
            [float(probabilities[label]) for label in labels],
            "Probability",
        ))
    if history:
        charts.append(_line_chart(
            "Training loss and validation loss",
            "Each point is one epoch from the saved sentiment log. Early stopping uses validation loss, not macro-F1.",
            [row["epoch"] for row in history],
            [
                ("train loss", [row["loss"] for row in history]),
                ("validation loss", [row["val_loss"] for row in history]),
            ],
            "Cross-entropy",
        ))
        charts.append(_line_chart(
            "Validation macro-F1 by epoch",
            "Macro-F1 is logged beside validation loss. It is not the number that chooses the saved epoch.",
            [row["epoch"] for row in history],
            [("validation macro-F1", [row["val_macro_f1"] for row in history])],
            "Macro-F1",
        ))
    return charts


def render_detailed_pipeline(result, threshold):
    trace = result.get("trace") or {}
    if trace.get("error"):
        st.warning("The detailed trace could not be built: " + trace["error"])
        return

    probability = result["probability"]
    features = result["features"]
    layers = trace.get("layers") or {}
    steps = result["steps"]
    pipeline_charts = sarcasm_pipeline_charts(result)

    def _charts_named(prefix):
        _show_charts([chart for chart in pipeline_charts if chart["title"].startswith(prefix)])

    st.markdown(
        "This is the path from the headline you typed to the probability above. "
        "Dropout is off for this calculation because the network is predicting, not training."
    )

    st.markdown("#### 1. Text cleaning")
    st.markdown(
        "Each line is the headline after one cleanup rule. The last line is what the rest of the pipeline uses."
    )
    clean_df = pd.DataFrame(trace["cleaning"], columns=["Rule", "Text afterwards"])
    st.dataframe(clean_df, width="stretch", hide_index=True)

    st.markdown("#### 2. Tokens, part of speech, lemmas, stop words")
    st.markdown(
        "spaCy splits the cleaned headline and tags each token. "
        "A stop word is dropped before TF-IDF unless it is negation or an intensifier "
        "(`not`, `never`, `very`, `really`, and the rest of the keep-list), because those words carry sarcasm."
    )
    st.dataframe(pd.DataFrame(trace["tokens"]), width="stretch", hide_index=True)
    st.markdown("**Lemmas that remain, joined for TF-IDF**")
    st.code(steps["processed_text"] or "(empty)", language=None)

    st.markdown("#### 3. Part-of-speech ratios")
    st.markdown(
        f"Ratio = count of that tag ÷ {trace['pos_total']} tagged tokens. "
        "The network uses the ratio, not the raw count, so a long headline is comparable to a short one."
    )
    pos_df = pd.DataFrame(trace["pos_rows"])
    pos_df["ratio"] = pos_df["ratio"].map(lambda v: f"{v:.4f}")
    st.dataframe(pos_df, width="stretch", hide_index=True)
    _charts_named("Part-of-speech")

    st.markdown("#### 4. VADER sentiment")
    hits = trace["vader_hits"]
    if hits:
        st.markdown(
            "VADER looks each word up in a fixed sentiment lexicon. "
            "Words that are not in the lexicon contribute nothing. "
            "The compound score is VADER's own normalised sum of the hits, not a plain average."
        )
        st.dataframe(pd.DataFrame(hits), width="stretch", hide_index=True)
    else:
        st.markdown("None of the words are in the VADER lexicon, so every lexicon hit is zero.")
    gap = trace["vader_gap_max"] - trace["vader_gap_min"]
    st.markdown(
        f"Positive lexicon words = {features['n_pos_words']}. "
        f"Negative lexicon words = {features['n_neg_words']}. "
        f"Sentiment contrast = {features['n_pos_words']} > 0 and {features['n_neg_words']} > 0 "
        f"→ **{int(features['sent_contrast'])}** "
        f"({'yes' if features['sent_contrast'] else 'no'})."
    )
    if len(hits) < 2:
        st.markdown(
            "Sentiment gap = largest lexicon score − smallest lexicon score. "
            "Fewer than two lexicon hits, so both ends are the same and the gap is **0**."
        )
    else:
        st.markdown(
            f"Sentiment gap = largest lexicon score − smallest lexicon score = "
            f"{trace['vader_gap_max']:.4f} − {trace['vader_gap_min']:.4f} = **{gap:.4f}**."
        )
    st.markdown(
        f"VADER output used by the model: positive {features['vader_pos']:.4f}, "
        f"negative {features['vader_neg']:.4f}, neutral {features['vader_neu']:.4f}, "
        f"compound {features['vader_compound']:+.4f}."
    )
    _charts_named("VADER")

    st.markdown("#### 5. Intensifiers and style")
    found = trace["intensifiers_found"]
    st.markdown(
        "Intensifier count = how many cleaned tokens are in the intensifier list "
        f"→ **{features['n_intensifiers']}**"
        + (f" ({', '.join(found)})" if found else " (none in this headline)")
        + "."
    )
    style_rows = [
        ("Exclamation marks", "count of !", str(features["n_exclaim"])),
        ("Question marks", "count of ?", str(features["n_question"])),
        ("Ellipses", "count of .. or …", str(features["n_ellipsis"])),
        ("Quote marks", "count of quote characters", str(features["n_quotes"])),
        ("Contains a digit", "1 if any digit, else 0", str(features["has_digit"])),
        ("Word count", "number of cleaned words", str(features["word_count"])),
        ("Average word length", "mean number of characters per cleaned word", f"{features['avg_word_len']:.4f}"),
    ]
    st.dataframe(
        pd.DataFrame(style_rows, columns=["Feature", "Calculation", "Value"]),
        width="stretch", hide_index=True,
    )

    st.markdown("#### 6. TF-IDF, then SVD, then scaling")
    st.markdown(
        "TF-IDF was fit on the training headlines only. It uses unigrams and bigrams, "
        "keeps at most 5,000 terms, ignores terms that appear in fewer than 2 headlines, "
        "and uses sublinear term frequency `1 + log(tf)`. "
        f"This headline lights up **{trace['tfidf_nonzero']}** of those terms. "
        "`idf` is from training. `tfidf` is the weight actually stored for this headline."
    )
    if trace["tfidf_rows"]:
        tfidf_df = pd.DataFrame(trace["tfidf_rows"])
        tfidf_df["n-gram"] = tfidf_df["term"].map(_ngram_label)
        tfidf_df["tfidf"] = tfidf_df["tfidf"].map(lambda v: f"{v:.4f}")
        tfidf_df["idf"] = tfidf_df["idf"].map(lambda v: f"{v:.4f}")
        st.dataframe(tfidf_df[["term", "n-gram", "tfidf", "idf"]], width="stretch", hide_index=True)
        _charts_named("TF-IDF")
    else:
        st.markdown("No TF-IDF terms fired. The vector is all zeros.")
    st.markdown(
        "Truncated SVD then compresses those 5,000 weights into 50 numbers. "
        "StandardScaler turns every feature into `(raw − training mean) / training scale`, "
        "using the mean and scale saved from the training set. "
        "The scaled column is what enters the network."
    )
    feat_df = pd.DataFrame(trace["feature_rows"])
    if not feat_df.empty:
        for col in ("raw", "training_mean", "training_scale", "scaled"):
            feat_df[col] = feat_df[col].map(lambda v: f"{v:.4f}")
        st.dataframe(feat_df, width="stretch", hide_index=True)

    st.markdown("#### 7. Word ids, GloVe embedding, BiLSTM")
    lstm_units = layers.get("lstm_units", 64)
    embed_dim = layers.get("embed_dim", 100)
    st.markdown(
        f"The sequence branch does not use the lemmas. It splits the cleaned headline into words, "
        f"replaces each word with its training vocabulary id, and pads the list to {len(trace['word_rows'])} "
        "with id 0. A word that was never in the training vocabulary gets id 1. "
        f"Id 0 is masked, so padding never enters the LSTM or the attention scores."
    )
    st.markdown(
        f"Each id is looked up in an embedding of {embed_dim} numbers. "
        "Those vectors started as GloVe and were then fine-tuned. "
        f"Spatial dropout of {layers.get('embed_dropout')} was used while training and is off now. "
        f"A bidirectional LSTM with {lstm_units} units reads the headline both ways, "
        f"so each word becomes {lstm_units * 2} numbers. "
        f"The LSTM uses **{layers.get('lstm_activation')}** for its output and "
        f"**{layers.get('lstm_recurrent_activation')}** for its input, forget, and output gates. "
        "The lengths below are the size of those vectors, not a second score."
    )
    word_df = pd.DataFrame(trace["word_rows"])
    attention = list(result["attention"])
    word_df["attention"] = [
        f"{attention[i]:.4f}" if i < len(attention) else ""
        for i in range(len(word_df))
    ]
    for col in ("embedding_l2", "bilstm_l2"):
        if col in word_df.columns:
            word_df[col] = word_df[col].map(lambda v: "" if v == "" or pd.isna(v) else f"{float(v):.4f}")
    st.dataframe(word_df, width="stretch", hide_index=True)
    if result["unknown_words"]:
        st.markdown("Unknown words, given id 1: **" + ", ".join(result["unknown_words"]) + "**.")

    st.markdown("#### 8. Attention")
    att_units = layers.get("attention_units", 64)
    att_sum = sum(attention) if attention else 0.0
    st.markdown(
        f"Attention is additive. For each word vector `h`, the score is `u · tanh(W h + b)`, "
        f"with a tanh hidden size of {att_units}. Softmax turns the scores into the weights below, "
        f"and the weights sum to **{att_sum:.4f}**. "
        "The sentence vector is the weighted sum of the BiLSTM outputs. "
        f"Its length is **{trace.get('internals', {}).get('context_l2', 0):.4f}** "
        f"across {trace.get('internals', {}).get('context_dim', lstm_units * 2)} numbers. "
        "Padding was given a score of −1e9 before the softmax, so its weight is zero."
    )
    _charts_named("Attention")

    internals = trace.get("internals") or {}
    st.markdown("#### 9. The two branches join")
    feat_hidden = internals.get("feat_hidden") or []
    fusion_hidden = internals.get("fusion_hidden") or []
    st.markdown(
        f"The NLP feature vector (74 scaled numbers) goes through a dense layer of 32 units "
        f"with **{layers.get('feat_activation')}**. "
        f"Dropout {layers.get('feat_dropout')} was used in training and is off now. "
        f"ReLU output: {len(feat_hidden)} numbers, "
        f"**{internals.get('feat_relu_zeros', 0)}** of them are exactly 0 because ReLU clips negatives."
    )
    if feat_hidden:
        st.code(" ".join(f"{v:.3f}" for v in feat_hidden), language=None)
    st.markdown(
        f"That 32-number vector is concatenated with the {trace.get('internals', {}).get('context_dim', 128)}-number "
        "attention vector. The joined vector goes through a dense layer of 64 units with "
        f"**{layers.get('fusion_activation')}**, then dropout {layers.get('fusion_dropout')} during training only. "
        f"ReLU output: {len(fusion_hidden)} numbers, "
        f"**{internals.get('fusion_relu_zeros', 0)}** of them are 0."
    )
    if fusion_hidden:
        st.code(" ".join(f"{v:.3f}" for v in fusion_hidden), language=None)

    st.markdown("#### 10. Sigmoid and the cutoff")
    st.markdown(
        "One last dense unit with **"
        f"{layers.get('output_activation')}** maps that vector to a probability: "
        "`sigmoid(z) = 1 / (1 + e^(−z))`."
    )
    if 1e-6 < probability < 1 - 1e-6:
        logit = math.log(probability / (1 - probability))
        st.markdown(
            f"The probability is **{probability:.6f}**, so the number before the sigmoid was "
            f"`ln(p / (1 − p))` = **{logit:.4f}**."
        )
    else:
        st.markdown(
            f"The probability is **{probability:.4f}**. The sigmoid is saturated, so the value before it is very large."
        )
    shown = set()
    for cutoff, label in (
        (0.5, "default cutoff"),
        (float(det.threshold), "tuned cutoff, chosen for the best validation F1"),
    ):
        key = round(float(cutoff), 4)
        if key in shown:
            continue
        shown.add(key)
        verdict = "sarcastic" if probability >= cutoff else "not sarcastic"
        used = " This is the cutoff used for the label above." if abs(cutoff - threshold) < 1e-9 else ""
        st.markdown(
            f"`{probability:.4f} ≥ {cutoff:.2f}` is **{str(probability >= cutoff).lower()}** "
            f"→ **{verdict}** ({label}).{used}"
        )

    st.markdown("#### 11. How this network was trained")
    info_path = os.path.join(MODELS_DIR, "training_info.json")
    with open(info_path, encoding="utf-8") as handle:
        info = json.load(handle)
    history = info["history"]
    best_index = min(range(len(history["val_loss"])), key=lambda i: history["val_loss"][i])
    st.markdown(
        "Optimizer: **Adam** (β1 = 0.9, β2 = 0.999, the Keras defaults). "
        "Loss: **binary cross-entropy**. "
        "Batch size: **64**. The training split has 20,521 headlines, and 20521 ÷ 64 rounds up to **321 steps** per epoch. "
        "A stratified 10% of the training headlines (2,281) is the validation set. "
        "The 5,701 test headlines are not used for stopping, the learning rate, or the cutoff."
    )
    stage1 = []
    n_epochs = len(history["loss"])
    for i in range(n_epochs):
        stage1.append({
            "epoch": i + 1,
            "planned": 20,
            "steps": "321/321",
            "accuracy": history["accuracy"][i],
            "loss": history["loss"][i],
            "val_accuracy": history["val_accuracy"][i],
            "val_loss": history["val_loss"][i],
            "learning_rate": history["learning_rate"][i],
            "decision": _stage1_decision(i, best_index, history["learning_rate"]),
        })
    st.markdown(
        f"Stage 1 froze the GloVe vectors and trained for up to 20 epochs at learning rate 0.001. "
        f"It ran {n_epochs} epochs."
    )
    _show_epoch_log("Stage 1 epoch log", stage1)
    _charts_named("Stage 1")

    st.markdown(
        "Stage 2 unfroze the GloVe vectors so words could pick up a sarcastic use, "
        "and continued with Adam at learning rate **0.0001** for up to 4 epochs. "
        "Early stopping here has patience **1**. "
        "Epoch 1 brought validation loss from 0.3022 down to 0.2952. "
        "Epoch 2 went back up to 0.2963, so patience ran out, training stopped, "
        "and the weights returned to epoch 1 of this stage. "
        "Those weights were kept because 0.2952 is lower than the stage-1 validation loss of 0.3022. "
        "If fine-tuning had been worse, the code would have rolled all the way back to the frozen model."
    )
    stage2 = [
        {
            "epoch": 1, "planned": 4, "steps": "321/321",
            "accuracy": 0.9039, "loss": 0.2340, "val_accuracy": 0.8772, "val_loss": 0.2952,
            "learning_rate": 1e-4,
            "decision": "Validation loss improved from 0.3022 to 0.2952. These are the weights that were kept.",
        },
        {
            "epoch": 2, "planned": 4, "steps": "321/321",
            "accuracy": 0.9088, "loss": 0.2212, "val_accuracy": 0.8794, "val_loss": 0.2963,
            "learning_rate": 1e-4,
            "decision": "Validation loss rose from 0.2952 to 0.2963. Patience is 1, so fine-tuning stops and epoch 1 is restored.",
        },
    ]
    _show_epoch_log("Stage 2 epoch log (GloVe unfrozen)", stage2)
    _charts_named("Stage 2")
    st.markdown(
        "The cutoff was chosen the same way: on the validation set, F1 at 0.50 was 0.8726 and F1 at 0.36 was 0.8791, "
        "so **0.36** was saved. On the untouched test set the saved model scores "
        "F1 0.8590 at 0.50 and F1 0.8655 at 0.36."
    )


def _sentiment_cleaning_rows(text):
    """Same cleanup as sentiment/text.py prepare(), shown one rule at a time."""
    import re
    from text import _URL_RE

    raw = str(text)
    quotes = raw.replace("\u2019", "'").replace("\u2018", "'")
    no_url = _URL_RE.sub(" ", quotes)
    no_html = re.sub(r"<[^>]+>", " ", no_url)
    collapsed = re.sub(r"\s+", " ", no_html).strip()
    return [
        ("Headline as typed", raw),
        ("Straighten curly quotes", quotes),
        ("Remove URLs", no_url),
        ("Remove HTML tags", no_html),
        ("Collapse spaces. Case, ! and ? stay", collapsed),
    ], collapsed


def _sentiment_token_rows(prepared):
    """One row per spaCy token, with the keep or drop decision sentiment/text.py makes."""
    from text import CONTENT_POS, CONTRAST, _is_negation, _nlp

    if not prepared:
        return []
    rows = []
    negate_next = False
    for token in _nlp()(prepared):
        pos = token.pos_
        lemma = token.lemma_.lower().strip()
        if token.text in {"!", "?"}:
            rows.append((token.text, pos, token.text, token.text, "Kept. ! and ? are their own tokens, and they clear negation."))
            negate_next = False
            continue
        if token.is_space or (token.is_punct and token.text not in {"!", "?"}):
            rows.append((token.text, pos, lemma, "", "Dropped. Other punctuation and spaces are not tokens, and they clear negation."))
            negate_next = False
            continue
        if _is_negation(token):
            rows.append((token.text, pos, lemma, "not", "Kept as not. The next content word is marked not_<lemma>."))
            negate_next = True
            continue
        lower = token.text.lower()
        if lemma in CONTRAST or lower in CONTRAST:
            rows.append((token.text, pos, lemma or lower, lemma or lower, "Kept. Contrast words stay, and they clear the negation flag."))
            negate_next = False
            continue
        if pos not in CONTENT_POS:
            rows.append((token.text, pos, lemma, "", f"Dropped. POS {pos} is not a content tag (adjective, adverb, verb, noun, proper noun, interjection, number)."))
            continue
        if token.is_stop:
            rows.append((token.text, pos, lemma, "", "Dropped. It is a content tag, but spaCy lists it as a stop word."))
            continue
        if not lemma:
            rows.append((token.text, pos, "", "", "Dropped. The lemma is empty."))
            continue
        kept = []
        reason = "Kept as the lowercase lemma."
        if token.text.isupper() and token.text.isalpha() and len(token.text) > 2:
            kept.append("allcaps")
            reason = "The word is ALL-CAPS and longer than 2 letters, so allcaps is added, then the lemma."
        if negate_next:
            kept.append("not_" + lemma)
            negate_next = False
            reason = "The previous word was negation, so the lemma is stored as not_" + lemma + " and the plain lemma is not stored."
        else:
            kept.append(lemma)
        rows.append((token.text, pos, lemma, " ".join(kept), reason))
    return rows


def _sentiment_probe(analyzer):
    """Read embedding, BiLSTM, ReLU, and softmax for one headline. Dropout stays off."""
    probe = getattr(analyzer, "_ui_probe", None)
    if probe is None:
        from tensorflow.keras import Model
        model = analyzer.model
        probe = Model(
            model.inputs,
            [
                model.get_layer("embedding").output,
                model.get_layer("bilstm").output,
                model.get_layer("hidden").output,
                model.get_layer("sentiment").output,
            ],
        )
        analyzer._ui_probe = probe
    return probe


def render_sentiment_pipeline(text, sentiment):
    """NLP and deep-learning steps for the sentiment model on this headline."""
    import numpy as np

    if sentiment.get("error"):
        st.warning("The sentiment pipeline could not be built: " + sentiment["error"])
        return
    analyzer, sentiment_error = get_sentiment()
    if analyzer is None:
        st.warning("The sentiment pipeline could not be built: " + (sentiment_error or "model unavailable"))
        return

    config_path = os.path.join(SENTIMENT_DIR, "config.json")
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    ablation_path = os.path.join(ROOT, "sentiment", "results", "sarcasm_feature_ablation.json")
    ablation = {}
    if os.path.exists(ablation_path):
        with open(ablation_path, encoding="utf-8") as handle:
            ablation = json.load(handle)

    labels = list(analyzer.labels)
    probabilities = sentiment["probabilities"]
    chosen = sentiment["sentiment"]
    only_f1 = ablation.get("sentiment_only_macro_f1")
    with_f1 = ablation.get("with_sarcasm_probability_macro_f1")
    if ablation.get("improved") is False and only_f1 is not None and with_f1 is not None:
        sarcasm_note = (
            f"A separate check did try that. On {ablation.get('eval_rows', 80)} held-out validation headlines, "
            f"sentiment alone scored macro-F1 **{only_f1:.4f}**. Adding the sarcasm probability scored "
            f"macro-F1 **{with_f1:.4f}**. The extra step was worse, so it was not saved. "
            "There is no sarcasm calibrator, and the three probabilities above are the sentiment softmax only."
        )
    elif analyzer.calibrator is not None:
        sarcasm_note = (
            "A saved calibrator then mixes that sarcasm probability with the three sentiment probabilities. "
            "That step is on for this model."
        )
    else:
        sarcasm_note = (
            "No sarcasm calibrator is loaded, so the three probabilities above are the sentiment softmax only."
        )
    st.markdown(
        "The Analyse button scores sarcasm first, then sentiment. "
        "Sentiment does **not** read the sarcasm probability. It reads this same headline on its own. "
        + sarcasm_note
    )
    st.markdown(
        "Dropout is off for the numbers below because the network is predicting, not training. "
        "This model does not use VADER, TF-IDF, attention, or a sigmoid cutoff. "
        "The class is whichever of the three softmax probabilities is largest."
    )

    st.markdown("#### 1. Text cleaning")
    st.markdown(
        "Sentiment keeps capital letters, exclamation marks, and question marks. "
        "Only URLs, HTML, and extra spaces are removed. The last line is what the tagger sees."
    )
    cleaning, prepared = _sentiment_cleaning_rows(text)
    st.dataframe(pd.DataFrame(cleaning, columns=["Rule", "Text afterwards"]), width="stretch", hide_index=True)

    st.markdown("#### 2. Part of speech, lemmas, negation")
    st.markdown(
        "spaCy tags the cleaned headline. Parser and named-entity steps are off. "
        "Content tags kept: adjective, adverb, verb, noun, proper noun, interjection, number. "
        "A negation (`not`, `no`, `never`, `n't`, and the rest of that list) is stored as `not`, "
        "and the next content lemma is stored as `not_<lemma>` instead of the plain lemma. "
        "Contrast words (`but`, `however`, `although`, `though`, `yet`) are kept and clear that flag."
    )
    token_rows = _sentiment_token_rows(prepared)
    if token_rows:
        st.dataframe(
            pd.DataFrame(token_rows, columns=["Token", "POS", "Lemma", "Kept as", "Why"]),
            width="stretch", hide_index=True,
        )
    else:
        st.markdown("Nothing was left after cleaning.")
    from text import tokenize
    kept_tokens = tokenize(text)
    st.markdown("**Tokens the embedding actually reads, in order**")
    st.code(" ".join(kept_tokens) if kept_tokens else "(empty)", language=None)

    st.markdown("#### 3. Word ids and padding")
    max_len = int(analyzer.max_len)
    ids = [analyzer.word_index.get(token, config.get("oov_id", 1)) for token in kept_tokens][:max_len]
    unknown = [token for token in kept_tokens[:max_len] if token not in analyzer.word_index]
    st.markdown(
        f"Each kept token is replaced by its id in the training vocabulary "
        f"({config.get('vocab_size', len(analyzer.word_index) + 2)} ids, including padding and unknown). "
        f"A word seen in fewer than {config.get('min_freq', 2)} training headlines, or never seen, gets id "
        f"**{config.get('oov_id', 1)}**. The list is cut at {max_len} and padded on the right with id "
        f"**{config.get('pad_id', 0)}**. Id 0 is masked, so padding does not enter the LSTM. "
        f"This headline uses **{len(ids)}** tokens and **{max_len - len(ids)}** padding positions."
    )
    sequence = analyzer.encode(text)
    emb, bilstm, hidden, softmax = _sentiment_probe(analyzer)([sequence], training=False)
    emb = np.asarray(emb.numpy() if hasattr(emb, "numpy") else emb)
    bilstm = np.asarray(bilstm.numpy() if hasattr(bilstm, "numpy") else bilstm)[0]
    hidden = np.asarray(hidden.numpy() if hasattr(hidden, "numpy") else hidden)[0]
    softmax = np.asarray(softmax.numpy() if hasattr(softmax, "numpy") else softmax)[0]
    word_rows = []
    for index, token in enumerate(kept_tokens[:max_len]):
        word_id = int(sequence[0, index])
        vector = emb[0, index]
        word_rows.append((index, token, word_id, f"{float(np.linalg.norm(vector)):.4f}"))
    if word_rows:
        st.dataframe(
            pd.DataFrame(word_rows, columns=["Position", "Token", "Id", "Embedding length"]),
            width="stretch", hide_index=True,
        )
    if unknown:
        st.markdown("Unknown tokens, given id 1: **" + ", ".join(unknown) + "**.")
    _show_charts(_sentiment_charts(word_rows, None, None, None))

    st.markdown("#### 4. GloVe embedding and the bidirectional LSTM")
    embed_dim = int(config.get("embed_dim", 100))
    lstm_units = int(config.get("lstm_units", 64))
    glove_hits = config.get("glove_words_found")
    st.markdown(
        f"Each id is looked up as {embed_dim} numbers. "
        f"Those vectors started as GloVe"
        + (f" ({glove_hits} vocabulary words were found in glove.6B.100d)" if glove_hits is not None else "")
        + ", and the embedding stayed trainable, so training could move them. "
        f"Spatial dropout of {config.get('dropout', {}).get('embedding', 0.3)} drops whole word vectors during training and is off now. "
        f"A bidirectional LSTM uses {lstm_units} units forward and {lstm_units} backward. "
        f"The two directions are concatenated, so the sentence vector has **{lstm_units * 2}** numbers. "
        "The LSTM output uses **tanh**. The input, forget, and output gates use **sigmoid**. "
        "This LSTM returns the final state only, not one vector per word, and it has no attention layer. "
        f"Length of this headline's BiLSTM vector: **{float(np.linalg.norm(bilstm)):.4f}**."
    )
    st.code(" ".join(f"{value:.3f}" for value in bilstm), language=None)

    st.markdown("#### 5. ReLU hidden layer")
    relu_zeros = int(np.sum(hidden <= 0.0))
    st.markdown(
        f"The {lstm_units * 2} BiLSTM numbers go through dropout "
        f"{config.get('dropout', {}).get('after_bilstm', 0.4)} during training only, "
        f"then a dense layer of {len(hidden)} units with **ReLU**: `max(0, z)`. "
        f"**{relu_zeros}** of the {len(hidden)} outputs are 0 because ReLU clips negatives. "
        f"Dropout {config.get('dropout', {}).get('after_dense', 0.3)} sits after this layer in training and is off now."
    )
    st.code(" ".join(f"{value:.3f}" for value in hidden), language=None)

    st.markdown("#### 6. Softmax and the chosen class")
    kernel, bias = analyzer.model.get_layer("sentiment").get_weights()
    logits = hidden @ np.asarray(kernel) + np.asarray(bias)
    shifted = logits - float(np.max(logits))
    exp_shifted = np.exp(shifted)
    shown_softmax = exp_shifted / float(np.sum(exp_shifted))
    st.markdown(
        "The last dense layer has 3 units and **softmax**. "
        "For logits `z`, `softmax(z)_i = exp(z_i − max z) / Σ exp(z_j − max z)`. "
        "Subtracting the max keeps the exponential from overflowing. The three results sum to 1. "
        "There is no cutoff: the predicted class is the largest probability. "
        f"Loss during training was **{config.get('loss', 'sparse_categorical_crossentropy')}**, "
        "which compares this distribution with the integer class id (0 negative, 1 neutral, 2 positive)."
    )
    softmax_rows = []
    for index, label in enumerate(labels):
        softmax_rows.append((
            label,
            f"{float(logits[index]):.4f}",
            f"{float(shown_softmax[index]):.4f}",
            f"{float(probabilities[label]):.4f}",
            "yes" if label == chosen else "",
        ))
    st.dataframe(
        pd.DataFrame(
            softmax_rows,
            columns=["Class", "Logit z", "Softmax", "Shown above", "Chosen"],
        ),
        width="stretch", hide_index=True,
    )
    st.markdown(
        f"Largest probability is **{chosen}** at **{probabilities[chosen]:.4f}**, "
        "which is the sentiment shown beside the sarcasm verdict."
    )
    _show_charts(_sentiment_charts(None, probabilities, labels, None))

    st.markdown("#### 7. How this network was trained")
    history_path = os.path.join(ROOT, "sentiment", "results", "history.json")
    history = []
    if os.path.exists(history_path):
        with open(history_path, encoding="utf-8") as handle:
            history = json.load(handle)
    train_rows = int(config.get("split", {}).get("train", 0))
    val_rows = int(config.get("split", {}).get("validation", 0))
    batch_size = int(config.get("batch_size", 32))
    steps = math.ceil(train_rows / batch_size) if batch_size else 0
    weights = config.get("class_weight") or {}
    weight_text = ", ".join(f"{label} {float(weights[label]):.3f}" for label in labels if label in weights)
    st.markdown(
        "Optimizer: **Adam** (β1 = 0.9, β2 = 0.999, the Keras defaults). "
        f"Starting learning rate: **{float(config.get('learning_rate', 0.001)):.3f}**. "
        f"Batch size: **{batch_size}**. "
        f"The training split has {train_rows} headlines, and {train_rows} ÷ {batch_size} "
        f"{'is' if train_rows % batch_size == 0 else 'rounds up to'} **{steps} steps** per epoch. "
        f"Validation has {val_rows} headlines. "
        "The Financial PhraseBank test, the SemEval test, and the newspaper test are not used for stopping or the learning rate. "
        f"Class weights, from the training labels only: {weight_text}. "
        "A weight above 1 makes a mistake on that class count more. Negative is the rare class, so its weight is the largest. "
        f"Up to {config.get('epochs_planned', 20)} epochs were allowed. "
        f"Early stopping watches validation loss with patience {config.get('early_stopping_patience', 3)}. "
        "A miss also halves the learning rate, and the rate is not taken below 0.00001."
    )
    if history:
        lines = []
        for row in history:
            lines.append(f"Epoch {row['epoch']}/{config.get('epochs_planned', 20)}    {steps}/{steps}")
            lines.append(f"accuracy: {row['accuracy']:.4f}    loss: {row['loss']:.4f}")
            lines.append(
                f"val_accuracy: {row['val_accuracy']:.4f}    val_loss: {row['val_loss']:.4f}    "
                f"val_macro_f1: {row['val_macro_f1']:.4f}"
            )
            lines.append(f"learning_rate: {row['learning_rate']:.6f}")
            lines.append("decision: " + row["decision"])
            lines.append("")
        st.markdown("**Sentiment epoch log**")
        st.code("\n".join(lines).rstrip(), language=None)
        _show_charts(_sentiment_charts(None, None, None, history))
        best = min(history, key=lambda row: row["val_loss"])
        last = history[-1]
        st.markdown(
            f"It ran **{len(history)}** epochs. The lowest validation loss is epoch **{best['epoch']}** "
            f"at **{best['val_loss']:.4f}** (val macro-F1 {best['val_macro_f1']:.4f}). "
            + config.get("stop_reason", "")
        )
        if last["epoch"] != best["epoch"] and last.get("val_macro_f1", 0) > best.get("val_macro_f1", 0):
            st.markdown(
                f"Epoch {last['epoch']} val macro-F1 was {last['val_macro_f1']:.4f}, "
                f"which is higher than epoch {best['epoch']} at {best['val_macro_f1']:.4f}. "
                "The restored weights are still epoch "
                f"{best['epoch']}, because the rule is lowest validation loss, not highest F1. "
                "Later epochs were still fitting the training headlines after validation loss had stopped falling."
            )


def _show_linguistic_tables(result):
    chunks = result["chunks"]
    entities = result["entities"]
    semantics = result["semantics"]
    if result.get("truncated"):
        st.caption("Only the first 6,000 characters were analysed.")
    st.markdown("**POS-based chunking**")
    st.caption(
        "A noun phrase is a determiner or pronoun, adjectives or numbers, then nouns. "
        "A verb phrase is auxiliaries and adverbs around a verb. The pattern column is the part-of-speech tags."
    )
    if chunks:
        st.dataframe(pd.DataFrame(chunks), width="stretch", hide_index=True)
    else:
        st.caption("No noun phrase or verb phrase matched.")
    st.markdown("**Named entities**")
    if entities:
        st.dataframe(pd.DataFrame(entities), width="stretch", hide_index=True)
    else:
        st.caption("No named entity was found.")
    st.markdown("**Semantic roles**")
    st.caption(
        "Each row is one sentence. The predicate is the dependency root. "
        "Subject and object are the root's arguments. Negation and adjective modifiers come from the same parse."
    )
    if semantics:
        st.dataframe(pd.DataFrame(semantics), width="stretch", hide_index=True)
    else:
        st.caption("No sentence was found.")


def render_sentiment_linguistics(text):
    """Display-only. Does not call either model."""
    from linguistics import (
        extract_docx,
        extract_pdf,
        extract_webpage_text,
        linguistic_analysis,
    )

    st.markdown(
        "These tables describe the wording. They are **not** passed into the sarcasm model "
        "or the sentiment model. Both models still score only the headline in the box above."
    )
    st.markdown("#### This headline")
    try:
        _show_linguistic_tables(linguistic_analysis(text))
    except Exception as exc:
        st.warning(f"The headline could not be analysed. {exc}")

    st.markdown("#### PDF, Word, or webpage")
    st.caption(
        "The extracted document stays in this section. It does not replace the headline "
        "and it is not sent to either model."
    )
    upload = st.file_uploader("PDF or Word (.docx)", type=["pdf", "docx"], key="sentiment_upload")
    if upload is not None:
        file_id = f"{upload.name}:{upload.size}"
        if st.session_state.get("sentiment_file_id") != file_id:
            st.session_state.sentiment_file_id = file_id
            try:
                data = upload.getvalue()
                if upload.name.lower().endswith(".docx"):
                    extracted = extract_docx(data)
                else:
                    extracted = extract_pdf(data)
                st.session_state.sentiment_document = {"source": upload.name, "text": extracted}
                st.session_state.sentiment_document_error = ""
            except Exception as exc:
                st.session_state.sentiment_document = None
                st.session_state.sentiment_document_error = str(exc)
    page_url = st.text_input(
        "Webpage URL for the article body",
        key="sentiment_page_url",
        placeholder="https://",
    )
    if st.button("Extract webpage text", disabled=not str(page_url).strip()):
        try:
            extracted = extract_webpage_text(page_url.strip())
            st.session_state.sentiment_document = {"source": page_url.strip(), "text": extracted}
            st.session_state.sentiment_document_error = ""
        except Exception as exc:
            st.session_state.sentiment_document = None
            st.session_state.sentiment_document_error = str(exc)
    if st.session_state.get("sentiment_document_error"):
        st.warning(st.session_state.sentiment_document_error)
    document = st.session_state.get("sentiment_document") or {}
    if document.get("text"):
        st.markdown(f"**Extracted from** {document.get('source', 'the file')}")
        st.text_area("Extracted text", value=document["text"][:12000], height=180, disabled=True)
        try:
            _show_linguistic_tables(linguistic_analysis(document["text"]))
        except Exception as exc:
            st.warning(f"The extracted text could not be analysed. {exc}")


def _ngram_label(term):
    n_words = len(str(term).split())
    return {1: "unigram", 2: "bigram", 3: "trigram"}.get(n_words, f"{n_words}-gram")


def _text_table(headers, rows):
    data = [[str(cell) for cell in row] for row in rows]
    widths = [len(str(header)) for header in headers]
    for row in data:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def format_row(row):
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    rule = format_row(["-" * width for width in widths])
    body = [format_row(row) for row in data]
    return "\n".join([format_row([str(header) for header in headers]), rule] + body)


def _epoch_log_text(title, rows):
    lines = [title, ""]
    for row in rows:
        lines.append(f"Epoch {row['epoch']}/{row['planned']}    {row['steps']}")
        lines.append(f"accuracy: {row['accuracy']:.4f}    loss: {row['loss']:.4f}")
        lines.append(f"val_accuracy: {row['val_accuracy']:.4f}    val_loss: {row['val_loss']:.4f}")
        lines.append(f"learning_rate: {row['learning_rate']:.6f}")
        if row.get("decision"):
            lines.append("decision: " + row["decision"])
        lines.append("")
    return "\n".join(lines).rstrip()


def pipeline_report(result, threshold):
    """Plain-text copy of the detailed pipeline, for the download button."""
    trace = result.get("trace") or {}
    if trace.get("error"):
        return "The detailed pipeline could not be built: " + trace["error"]

    probability = result["probability"]
    features = result["features"]
    layers = trace.get("layers") or {}
    steps = result["steps"]
    verdict = "sarcastic" if probability >= threshold else "not sarcastic"
    lines = [
        "DETAILED PIPELINE",
        "Headline: " + result.get("headline", ""),
        f"Probability: {probability:.6f}",
        f"Label at cutoff {threshold:.2f}: {verdict}",
        "",
        "This is the path from the headline to that probability.",
        "Dropout is off for this calculation because the network is predicting, not training.",
        "",
        "1. Text cleaning",
        "Each line is the headline after one cleanup rule.",
        "",
        _text_table(["Rule", "Text afterwards"], trace["cleaning"]),
        "",
        "2. Tokens, part of speech, lemmas, stop words",
        "A stop word is dropped before TF-IDF unless it is negation or an intensifier.",
        "",
        _text_table(
            ["token", "pos", "lemma", "decision"],
            [[row["token"], row["pos"], row["lemma"], row["decision"]] for row in trace["tokens"]],
        ),
        "",
        "Lemmas that remain, joined for TF-IDF:",
        steps["processed_text"] or "(empty)",
        "",
        "3. Part-of-speech ratios",
        f"Ratio = count of that tag / {trace['pos_total']} tagged tokens.",
        "",
        _text_table(
            ["pos", "count", "ratio", "calculation"],
            [[row["pos"], row["count"], f"{row['ratio']:.4f}", row["calculation"]] for row in trace["pos_rows"]],
        ),
        "",
        "4. VADER sentiment",
    ]
    hits = trace["vader_hits"]
    if hits:
        lines += [
            "VADER looks each word up in a fixed sentiment lexicon.",
            "The compound score is VADER's own normalised sum of the hits, not a plain average.",
            "",
            _text_table(
                ["word", "lexicon_score"],
                [[row["word"], f"{row['lexicon_score']:.4f}"] for row in hits],
            ),
        ]
    else:
        lines.append("None of the words are in the VADER lexicon, so every lexicon hit is zero.")
    gap = trace["vader_gap_max"] - trace["vader_gap_min"]
    lines += [
        "",
        f"Positive lexicon words = {features['n_pos_words']}. Negative lexicon words = {features['n_neg_words']}.",
        f"Sentiment contrast = {features['n_pos_words']} > 0 and {features['n_neg_words']} > 0 "
        f"-> {int(features['sent_contrast'])} ({'yes' if features['sent_contrast'] else 'no'}).",
    ]
    if len(hits) < 2:
        lines.append("Sentiment gap = 0, because fewer than two lexicon hits were found.")
    else:
        lines.append(
            f"Sentiment gap = {trace['vader_gap_max']:.4f} - {trace['vader_gap_min']:.4f} = {gap:.4f}."
        )
    found = trace["intensifiers_found"]
    lines += [
        f"VADER output: positive {features['vader_pos']:.4f}, negative {features['vader_neg']:.4f}, "
        f"neutral {features['vader_neu']:.4f}, compound {features['vader_compound']:+.4f}.",
        "",
        "5. Intensifiers and style",
        f"Intensifier count = {features['n_intensifiers']}"
        + (f" ({', '.join(found)})" if found else " (none in this headline)")
        + ".",
        "",
        _text_table(
            ["Feature", "Calculation", "Value"],
            [
                ("Exclamation marks", "count of !", features["n_exclaim"]),
                ("Question marks", "count of ?", features["n_question"]),
                ("Ellipses", "count of .. or …", features["n_ellipsis"]),
                ("Quote marks", "count of quote characters", features["n_quotes"]),
                ("Contains a digit", "1 if any digit, else 0", features["has_digit"]),
                ("Word count", "number of cleaned words", features["word_count"]),
                ("Average word length", "mean characters per cleaned word", f"{features['avg_word_len']:.4f}"),
            ],
        ),
        "",
        "6. TF-IDF, then SVD, then scaling",
        "TF-IDF was fit on the training headlines only. It uses unigrams and bigrams,",
        "keeps at most 5,000 terms, ignores terms in fewer than 2 headlines,",
        "and uses sublinear term frequency 1 + log(tf).",
        f"This headline lights up {trace['tfidf_nonzero']} of those terms.",
        "",
    ]
    if trace["tfidf_rows"]:
        lines.append(_text_table(
            ["term", "n-gram", "tfidf", "idf"],
            [[row["term"], _ngram_label(row["term"]), f"{row['tfidf']:.4f}", f"{row['idf']:.4f}"] for row in trace["tfidf_rows"]],
        ))
    else:
        lines.append("No TF-IDF terms fired. The vector is all zeros.")
    lines += [
        "",
        "Truncated SVD compresses those 5,000 weights into 50 numbers.",
        "StandardScaler turns every feature into (raw - training mean) / training scale.",
        "",
    ]
    if trace["feature_rows"]:
        lines.append(_text_table(
            ["feature", "raw", "training_mean", "training_scale", "scaled"],
            [[
                row["feature"], f"{row['raw']:.4f}", f"{row['training_mean']:.4f}",
                f"{row['training_scale']:.4f}", f"{row['scaled']:.4f}",
            ] for row in trace["feature_rows"]],
        ))

    lstm_units = layers.get("lstm_units", 64)
    embed_dim = layers.get("embed_dim", 100)
    attention = list(result["attention"])
    word_rows = []
    for i, row in enumerate(trace["word_rows"]):
        word_rows.append([
            row["position"], row["token"], row["word_id"], row["known"], row["role"],
            f"{attention[i]:.4f}" if i < len(attention) else "",
            f"{row['embedding_l2']:.4f}" if "embedding_l2" in row else "",
            f"{row['bilstm_l2']:.4f}" if "bilstm_l2" in row else "",
        ])
    internals = trace.get("internals") or {}
    feat_hidden = internals.get("feat_hidden") or []
    fusion_hidden = internals.get("fusion_hidden") or []
    lines += [
        "",
        "7. Word ids, GloVe embedding, BiLSTM",
        f"Each word becomes an id, then a {embed_dim}-number GloVe embedding that was fine-tuned.",
        f"Padding id 0 is masked. Unknown words get id 1.",
        f"Spatial dropout {layers.get('embed_dropout')} is off now.",
        f"Bidirectional LSTM, {lstm_units} units each way, output activation {layers.get('lstm_activation')},",
        f"gate activation {layers.get('lstm_recurrent_activation')}.",
        "",
        _text_table(
            ["position", "token", "word_id", "known", "role", "attention", "embedding_l2", "bilstm_l2"],
            word_rows,
        ),
    ]
    if result["unknown_words"]:
        lines.append("Unknown words, given id 1: " + ", ".join(result["unknown_words"]))
    lines += [
        "",
        "8. Attention",
        "Score for each word vector h: u · tanh(W h + b), then softmax.",
        f"Tanh hidden size: {layers.get('attention_units', 64)}. Weights sum to {sum(attention) if attention else 0:.4f}.",
        f"Sentence-vector length: {internals.get('context_l2', 0):.4f} "
        f"across {internals.get('context_dim', lstm_units * 2)} numbers.",
        "",
        "9. The two branches join",
        f"74 scaled NLP numbers -> dense 32, activation {layers.get('feat_activation')}.",
        f"Dropout {layers.get('feat_dropout')} is off now. ReLU zeros: {internals.get('feat_relu_zeros', 0)} of {len(feat_hidden)}.",
        " ".join(f"{value:.3f}" for value in feat_hidden),
        f"Concatenated with the {internals.get('context_dim', 128)}-number attention vector,",
        f"then dense 64, activation {layers.get('fusion_activation')}. Dropout {layers.get('fusion_dropout')} is off now.",
        f"ReLU zeros: {internals.get('fusion_relu_zeros', 0)} of {len(fusion_hidden)}.",
        " ".join(f"{value:.3f}" for value in fusion_hidden),
        "",
        "10. Sigmoid and the cutoff",
        f"Final dense unit activation: {layers.get('output_activation')}.",
        "sigmoid(z) = 1 / (1 + e^(-z)).",
    ]
    if 1e-6 < probability < 1 - 1e-6:
        logit = math.log(probability / (1 - probability))
        lines.append(f"Probability = {probability:.6f}, so z = ln(p / (1 - p)) = {logit:.4f}.")
    else:
        lines.append(f"Probability = {probability:.4f}. The sigmoid is saturated, so z is very large.")
    shown = set()
    for cutoff, label in (
        (0.5, "default cutoff"),
        (float(det.threshold), "tuned cutoff, chosen for the best validation F1"),
    ):
        key = round(float(cutoff), 4)
        if key in shown:
            continue
        shown.add(key)
        cutoff_verdict = "sarcastic" if probability >= cutoff else "not sarcastic"
        used = " This is the cutoff used for the label." if abs(cutoff - threshold) < 1e-9 else ""
        lines.append(
            f"{probability:.4f} >= {cutoff:.2f} is {str(probability >= cutoff).lower()} "
            f"-> {cutoff_verdict} ({label}).{used}"
        )

    info_path = os.path.join(MODELS_DIR, "training_info.json")
    with open(info_path, encoding="utf-8") as handle:
        history = json.load(handle)["history"]
    best_index = min(range(len(history["val_loss"])), key=lambda i: history["val_loss"][i])
    stage1 = []
    for i in range(len(history["loss"])):
        stage1.append({
            "epoch": i + 1, "planned": 20, "steps": "321/321",
            "accuracy": history["accuracy"][i], "loss": history["loss"][i],
            "val_accuracy": history["val_accuracy"][i], "val_loss": history["val_loss"][i],
            "learning_rate": history["learning_rate"][i],
            "decision": _stage1_decision(i, best_index, history["learning_rate"]),
        })
    stage2 = [
        {
            "epoch": 1, "planned": 4, "steps": "321/321",
            "accuracy": 0.9039, "loss": 0.2340, "val_accuracy": 0.8772, "val_loss": 0.2952,
            "learning_rate": 1e-4,
            "decision": "Validation loss improved from 0.3022 to 0.2952. These are the weights that were kept.",
        },
        {
            "epoch": 2, "planned": 4, "steps": "321/321",
            "accuracy": 0.9088, "loss": 0.2212, "val_accuracy": 0.8794, "val_loss": 0.2963,
            "learning_rate": 1e-4,
            "decision": "Validation loss rose from 0.2952 to 0.2963. Patience is 1, so fine-tuning stops and epoch 1 is restored.",
        },
    ]
    lines += [
        "",
        "11. How this network was trained",
        "Optimizer: Adam (beta1 = 0.9, beta2 = 0.999). Loss: binary cross-entropy. Batch size: 64.",
        "20,521 training headlines / 64 = 321 steps per epoch. Validation set: 2,281. Test set: 5,701, left untouched.",
        "",
        f"Stage 1 froze GloVe and trained up to 20 epochs at learning rate 0.001. It ran {len(stage1)} epochs.",
        "",
        _epoch_log_text("Stage 1 epoch log", stage1),
        "",
        "Stage 2 unfroze GloVe at learning rate 0.0001, patience 1.",
        "Epoch 1 lowered validation loss from 0.3022 to 0.2952 and those weights were kept.",
        "Epoch 2 rose to 0.2963, so fine-tuning stopped and epoch 1 was restored.",
        "",
        _epoch_log_text("Stage 2 epoch log (GloVe unfrozen)", stage2),
        "",
        "Validation F1 was 0.8726 at cutoff 0.50 and 0.8791 at 0.36, so 0.36 was saved.",
        "Test F1 is 0.8590 at 0.50 and 0.8655 at 0.36.",
        "",
    ]
    return "\n".join(lines)


def sentiment_pipeline_report(text, sentiment):
    """Plain-text copy of the detailed sentiment pipeline, for download."""
    import numpy as np
    from text import tokenize

    if sentiment.get("error"):
        return "The sentiment pipeline could not be built: " + sentiment["error"], []
    analyzer, sentiment_error = get_sentiment()
    if analyzer is None:
        return "The sentiment pipeline could not be built: " + (sentiment_error or "model unavailable"), []

    with open(os.path.join(SENTIMENT_DIR, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    ablation = {}
    ablation_path = os.path.join(ROOT, "sentiment", "results", "sarcasm_feature_ablation.json")
    if os.path.exists(ablation_path):
        with open(ablation_path, encoding="utf-8") as handle:
            ablation = json.load(handle)

    labels = list(analyzer.labels)
    probabilities = sentiment["probabilities"]
    chosen = sentiment["sentiment"]
    only_f1 = ablation.get("sentiment_only_macro_f1")
    with_f1 = ablation.get("with_sarcasm_probability_macro_f1")
    if ablation.get("improved") is False and only_f1 is not None and with_f1 is not None:
        sarcasm_note = (
            f"A separate check did try that. On {ablation.get('eval_rows', 80)} held-out validation headlines, "
            f"sentiment alone scored macro-F1 {only_f1:.4f}. Adding the sarcasm probability scored "
            f"macro-F1 {with_f1:.4f}. The extra step was worse, so it was not saved. "
            "There is no sarcasm calibrator, and the three probabilities are the sentiment softmax only."
        )
    elif analyzer.calibrator is not None:
        sarcasm_note = (
            "A saved calibrator then mixes that sarcasm probability with the three sentiment probabilities."
        )
    else:
        sarcasm_note = "No sarcasm calibrator is loaded, so the three probabilities are the sentiment softmax only."

    cleaning, prepared = _sentiment_cleaning_rows(text)
    token_rows = _sentiment_token_rows(prepared)
    kept_tokens = tokenize(text)
    max_len = int(analyzer.max_len)
    unknown = [token for token in kept_tokens[:max_len] if token not in analyzer.word_index]
    sequence = analyzer.encode(text)
    emb, bilstm, hidden, _softmax = _sentiment_probe(analyzer)([sequence], training=False)
    emb = np.asarray(emb.numpy() if hasattr(emb, "numpy") else emb)
    bilstm = np.asarray(bilstm.numpy() if hasattr(bilstm, "numpy") else bilstm)[0]
    hidden = np.asarray(hidden.numpy() if hasattr(hidden, "numpy") else hidden)[0]
    word_rows = []
    for index, token in enumerate(kept_tokens[:max_len]):
        word_rows.append((index, token, int(sequence[0, index]), f"{float(np.linalg.norm(emb[0, index])):.4f}"))
    kernel, bias = analyzer.model.get_layer("sentiment").get_weights()
    logits = hidden @ np.asarray(kernel) + np.asarray(bias)
    shifted = logits - float(np.max(logits))
    exp_shifted = np.exp(shifted)
    shown_softmax = exp_shifted / float(np.sum(exp_shifted))
    embed_dim = int(config.get("embed_dim", 100))
    lstm_units = int(config.get("lstm_units", 64))
    glove_hits = config.get("glove_words_found")
    relu_zeros = int(np.sum(hidden <= 0.0))
    lines = [
        "DETAILED SENTIMENT PIPELINE",
        "Headline: " + text.strip(),
        f"Sentiment: {chosen} ({probabilities[chosen]:.4f})",
        "",
        "The Analyse button scores sarcasm first, then sentiment.",
        "Sentiment does not read the sarcasm probability. It reads this same headline on its own.",
        sarcasm_note,
        "",
        "Dropout is off for this calculation because the network is predicting, not training.",
        "This model does not use VADER, TF-IDF, attention, or a sigmoid cutoff.",
        "The class is whichever of the three softmax probabilities is largest.",
        "",
        "1. Text cleaning",
        "Capital letters, exclamation marks, and question marks stay. URLs, HTML, and extra spaces are removed.",
        "",
        _text_table(["Rule", "Text afterwards"], cleaning),
        "",
        "2. Part of speech, lemmas, negation",
        "Content tags kept: adjective, adverb, verb, noun, proper noun, interjection, number.",
        "Negation is stored as not, and the next content lemma is stored as not_<lemma>.",
        "Contrast words (but, however, although, though, yet) are kept and clear that flag.",
        "",
    ]
    if token_rows:
        lines.append(_text_table(["Token", "POS", "Lemma", "Kept as", "Why"], token_rows))
    else:
        lines.append("Nothing was left after cleaning.")
    lines += [
        "",
        "Tokens the embedding reads, in order:",
        " ".join(kept_tokens) if kept_tokens else "(empty)",
        "",
        "3. Word ids and padding",
        f"Vocabulary size {config.get('vocab_size', len(analyzer.word_index) + 2)}, including padding and unknown.",
        f"A word seen in fewer than {config.get('min_freq', 2)} training headlines gets id {config.get('oov_id', 1)}.",
        f"The list is cut at {max_len} and padded on the right with id {config.get('pad_id', 0)}.",
        f"This headline uses {len(kept_tokens[:max_len])} tokens and {max_len - len(kept_tokens[:max_len])} padding positions.",
        "",
        _text_table(["Position", "Token", "Id", "Embedding length"], word_rows),
    ]
    if unknown:
        lines.append("Unknown tokens, given id 1: " + ", ".join(unknown))
    lines += [
        "",
        "4. GloVe embedding and the bidirectional LSTM",
        f"Each id is {embed_dim} numbers, started from GloVe"
        + (f" ({glove_hits} vocabulary words found in glove.6B.100d)" if glove_hits is not None else "")
        + ", and the embedding stayed trainable.",
        f"Spatial dropout {config.get('dropout', {}).get('embedding', 0.3)} is off now.",
        f"Bidirectional LSTM: {lstm_units} forward and {lstm_units} backward, concatenated to {lstm_units * 2}.",
        "LSTM output uses tanh. Input, forget, and output gates use sigmoid.",
        "The LSTM returns the final state only. There is no attention layer.",
        f"Length of this BiLSTM vector: {float(np.linalg.norm(bilstm)):.4f}.",
        " ".join(f"{value:.3f}" for value in bilstm),
        "",
        "5. ReLU hidden layer",
        f"Dropout {config.get('dropout', {}).get('after_bilstm', 0.4)} is off now, then dense {len(hidden)} with ReLU: max(0, z).",
        f"ReLU zeros: {relu_zeros} of {len(hidden)}.",
        f"Dropout {config.get('dropout', {}).get('after_dense', 0.3)} after this layer is off now.",
        " ".join(f"{value:.3f}" for value in hidden),
        "",
        "6. Softmax and the chosen class",
        "softmax(z)_i = exp(z_i - max z) / sum exp(z_j - max z).",
        f"Training loss: {config.get('loss', 'sparse_categorical_crossentropy')} (0 negative, 1 neutral, 2 positive).",
        "",
        _text_table(
            ["Class", "Logit z", "Softmax", "Shown above", "Chosen"],
            [
                (
                    label,
                    f"{float(logits[index]):.4f}",
                    f"{float(shown_softmax[index]):.4f}",
                    f"{float(probabilities[label]):.4f}",
                    "yes" if label == chosen else "",
                )
                for index, label in enumerate(labels)
            ],
        ),
        "",
        f"Largest probability is {chosen} at {probabilities[chosen]:.4f}.",
        "",
        "7. How this network was trained",
    ]
    history = []
    history_path = os.path.join(ROOT, "sentiment", "results", "history.json")
    if os.path.exists(history_path):
        with open(history_path, encoding="utf-8") as handle:
            history = json.load(handle)
    train_rows = int(config.get("split", {}).get("train", 0))
    val_rows = int(config.get("split", {}).get("validation", 0))
    batch_size = int(config.get("batch_size", 32))
    steps = math.ceil(train_rows / batch_size) if batch_size else 0
    weights = config.get("class_weight") or {}
    weight_text = ", ".join(f"{label} {float(weights[label]):.3f}" for label in labels if label in weights)
    lines += [
        "Optimizer: Adam (beta1 = 0.9, beta2 = 0.999).",
        f"Starting learning rate: {float(config.get('learning_rate', 0.001)):.3f}. Batch size: {batch_size}.",
        f"Training headlines: {train_rows}. Steps per epoch: {steps}. Validation headlines: {val_rows}.",
        "The Financial PhraseBank test, the SemEval test, and the newspaper test are not used for stopping or the learning rate.",
        f"Class weights: {weight_text}.",
        f"Planned epochs: {config.get('epochs_planned', 20)}. Early stopping patience: {config.get('early_stopping_patience', 3)}.",
        "A miss halves the learning rate, and the rate is not taken below 0.00001.",
        "",
    ]
    if history:
        epoch_rows = []
        for row in history:
            epoch_rows.append(
                f"Epoch {row['epoch']}/{config.get('epochs_planned', 20)}    {steps}/{steps}\n"
                f"accuracy: {row['accuracy']:.4f}    loss: {row['loss']:.4f}\n"
                f"val_accuracy: {row['val_accuracy']:.4f}    val_loss: {row['val_loss']:.4f}    "
                f"val_macro_f1: {row['val_macro_f1']:.4f}\n"
                f"learning_rate: {row['learning_rate']:.6f}\n"
                f"decision: {row['decision']}"
            )
        lines.append("Sentiment epoch log")
        lines.append("\n\n".join(epoch_rows))
        best = min(history, key=lambda row: row["val_loss"])
        last = history[-1]
        lines += [
            "",
            f"It ran {len(history)} epochs. Lowest validation loss is epoch {best['epoch']} "
            f"at {best['val_loss']:.4f} (val macro-F1 {best['val_macro_f1']:.4f}).",
            config.get("stop_reason", ""),
        ]
        if last["epoch"] != best["epoch"] and last.get("val_macro_f1", 0) > best.get("val_macro_f1", 0):
            lines.append(
                f"Epoch {last['epoch']} val macro-F1 was {last['val_macro_f1']:.4f}, "
                f"higher than epoch {best['epoch']} at {best['val_macro_f1']:.4f}. "
                f"The restored weights are still epoch {best['epoch']}, because the rule is lowest validation loss."
            )
    lines.append("")
    return "\n".join(lines), _sentiment_charts(word_rows, probabilities, labels, history)


def _charts_svg(charts):
    parts = []
    for chart in charts or []:
        if chart["kind"] == "bar" and chart["values"]:
            parts.append(_bar_svg(chart))
        elif chart["kind"] == "line" and chart["series"]:
            parts.append(_line_svg(chart))
    return "\n".join(parts)


def _bar_svg(chart):
    labels, values = chart["labels"], chart["values"]
    top = max(values) or 1.0
    row_h, width, left = 22, 680, 140
    height = 36 + row_h * len(labels)
    rows = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" role="img">',
        f'<text x="0" y="16" font-family="Georgia,serif" font-size="14">{html.escape(chart["title"])}</text>',
    ]
    for index, (label, value) in enumerate(zip(labels, values)):
        y = 28 + index * row_h
        bar = (width - left - 70) * (value / top)
        rows.append(
            f'<text x="0" y="{y + 12}" font-family="Consolas,monospace" font-size="11">{html.escape(label)}</text>'
            f'<rect x="{left}" y="{y}" width="{bar:.1f}" height="14" fill="#3b6ea5"/>'
            f'<text x="{left + bar + 6:.1f}" y="{y + 12}" font-family="Consolas,monospace" font-size="11">{value:.4f}</text>'
        )
    rows.append("</svg>")
    return "\n".join(rows)


def _line_svg(chart):
    width, height, left, bottom = 680, 260, 48, 36
    series = chart["series"]
    epochs = chart["x"]
    present = [value for _, values in series for value in values]
    high = max(present)
    span = high - min(present) or abs(high) or 1.0
    y_max = high + span * 0.08
    plot_w, plot_h = width - left - 16, height - bottom - 28
    colors = ["#1f4e79", "#7eb6d9", "#c47b3a"]
    rows = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" role="img">',
        f'<text x="0" y="16" font-family="Georgia,serif" font-size="14">{html.escape(chart["title"])}</text>',
    ]
    def xy(index, value):
        x = left + (plot_w * index / max(len(epochs) - 1, 1))
        y = 24 + plot_h * (1 - (value / y_max if y_max else 0))
        return x, y
    for series_index, (name, values) in enumerate(series):
        points = " ".join(f"{xy(i, value)[0]:.1f},{xy(i, value)[1]:.1f}" for i, value in enumerate(values))
        color = colors[series_index % len(colors)]
        rows.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>')
        rows.append(
            f'<text x="{left + series_index * 160}" y="{height - 8}" font-family="Consolas,monospace" '
            f'font-size="11" fill="{color}">{html.escape(name)}</text>'
        )
    for index, epoch in enumerate(epochs):
        x, _ = xy(index, 0)
        rows.append(
            f'<text x="{x:.1f}" y="{height - 22}" font-family="Consolas,monospace" font-size="11" '
            f'text-anchor="middle">{epoch}</text>'
        )
    rows.append(
        f'<text x="4" y="28" font-family="Consolas,monospace" font-size="11">{y_max:.4f}</text>'
        f'<text x="4" y="{24 + plot_h:.0f}" font-family="Consolas,monospace" font-size="11">0</text>'
    )
    rows.append("</svg>")
    return "\n".join(rows)


def _html_document(title, body, charts=None):
    safe_title = html.escape(title)
    return (
        "<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<title>{safe_title}</title><style>"
        "body{font-family:Georgia,serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#222;}"
        "h1{font-size:1.4rem;} pre{white-space:pre-wrap;font-family:Consolas,monospace;font-size:13px;line-height:1.45;}"
        "</style></head><body>"
        f"<h1>{safe_title}</h1>{_charts_svg(charts)}<pre>{html.escape(body)}</pre></body></html>"
    ).encode("utf-8")


def _pdf_escape(line):
    return line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf_chart_streams(chart):
    """One or more PDF content streams. Bar values are printed so the page can be checked."""
    streams = []
    if chart["kind"] == "bar" and chart["values"]:
        labels, values = chart["labels"], chart["values"]
        top = max(abs(value) for value in values) or 1.0
        for start in range(0, len(labels), 22):
            chunk_l = labels[start:start + 22]
            chunk_v = values[start:start + 22]
            commands = ["BT", "/F1 12 Tf", "40 810 Td", f"({_pdf_escape(chart['title'][:90])}) Tj", "ET"]
            for index, (label, value) in enumerate(zip(chunk_l, chunk_v)):
                y = 780 - index * 28
                width = 360 * (abs(value) / top) if top else 0
                commands.append("0.231 0.431 0.647 rg")
                commands.append(f"150 {y} {width:.2f} 12 re f")
                commands.append("0 0 0 rg")
                commands.append("BT")
                commands.append(f"/F1 9 Tf 40 {y + 2} Td ({_pdf_escape(str(label)[:22])}) Tj ET")
                commands.append("BT")
                commands.append(f"/F1 9 Tf 520 {y + 2} Td ({value:.4f}) Tj ET")
            streams.append("\n".join(commands).encode("latin-1", "replace"))
    elif chart["kind"] == "line" and chart["series"]:
        present = [value for _, values in chart["series"] for value in values]
        high = max(present)
        span = high - min(present) or abs(high) or 1.0
        y_max = high + span * 0.08
        commands = [
            "BT", "/F1 12 Tf", "40 810 Td", f"({_pdf_escape(chart['title'][:90])}) Tj", "ET",
            "BT", "/F1 9 Tf", "40 790 Td",
            f"({_pdf_escape('y from 0 to ' + format(y_max, '.4f'))}) Tj", "ET",
        ]
        colors = [(0.122, 0.306, 0.475), (0.494, 0.714, 0.851)]
        epochs = chart["x"]
        for series_index, (name, values) in enumerate(chart["series"]):
            red, green, blue = colors[series_index % len(colors)]
            commands.append(f"{red:.3f} {green:.3f} {blue:.3f} RG")
            commands.append("1.5 w")
            for index, value in enumerate(values):
                x = 70 + (460 * index / max(len(epochs) - 1, 1))
                y = 120 + 560 * ((value / y_max) if y_max else 0)
                commands.append(f"{x:.2f} {y:.2f} {'m' if index == 0 else 'l'}")
            commands.append("S")
            commands.append("BT")
            commands.append(
                f"/F1 9 Tf {70 + series_index * 180} 90 Td ({_pdf_escape(name)}) Tj ET"
            )
        for index, epoch in enumerate(epochs):
            x = 70 + (460 * index / max(len(epochs) - 1, 1))
            commands.append("BT")
            commands.append(f"/F1 8 Tf {x - 4:.2f} 100 Td ({epoch}) Tj ET")
        streams.append("\n".join(commands).encode("latin-1", "replace"))
    return streams


def _text_to_pdf(text, charts=None):
    """Multi-page PDF using the built-in Helvetica font. No extra package."""
    safe = (
        text.replace("\u2014", "-").replace("\u2013", "-").replace("\u2212", "-")
        .replace("\u2265", ">=").replace("\u2264", "<=")
        .replace("\u00b7", " | ").replace("\u2026", "...")
        .replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", "\"").replace("\u201d", "\"")
        .replace("\u03b2", "beta").replace("\u03a3", "sum").replace("\u00f7", "/")
    )
    wrapped = []
    for raw in safe.splitlines():
        raw = raw.encode("latin-1", "replace").decode("latin-1")
        if not raw:
            wrapped.append("")
            continue
        while len(raw) > 95:
            cut = raw.rfind(" ", 0, 95)
            if cut < 40:
                cut = 95
            wrapped.append(raw[:cut])
            raw = raw[cut:].lstrip()
        wrapped.append(raw)
    per_page = 58
    pages = [wrapped[i:i + per_page] for i in range(0, len(wrapped), per_page)] or [[""]]

    streams = []
    for page_lines in pages:
        commands = ["BT", "/F1 10 Tf", "48 800 Td", "13 TL"]
        for line in page_lines:
            commands.append(f"({_pdf_escape(line)}) Tj")
            commands.append("T*")
        commands.append("ET")
        streams.append("\n".join(commands).encode("latin-1"))
    for chart in charts or []:
        streams.extend(_pdf_chart_streams(chart))

    page_ids = []
    content_ids = []
    next_id = 4
    for _stream in streams:
        page_ids.append(next_id)
        content_ids.append(next_id + 1)
        next_id += 2
    font_id = 3
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    bodies = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("ascii"),
        font_id: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    for page_id, content_id, stream in zip(page_ids, content_ids, streams):
        bodies[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            f"/Contents {content_id} 0 R /Resources << /Font << /F1 {font_id} 0 R >> >> >>"
        ).encode("ascii")
        bodies[content_id] = b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream"

    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_id in range(1, next_id):
        offsets.append(len(output))
        output += f"{object_id} 0 obj\n".encode("ascii") + bodies[object_id] + b"\nendobj\n"
    xref_at = len(output)
    output += f"xref\n0 {next_id}\n".encode("ascii")
    output += b"0000000000 65535 f \n"
    for object_id in range(1, next_id):
        output += f"{offsets[object_id]:010d} 00000 n \n".encode("ascii")
    output += (
        f"trailer << /Size {next_id} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF".encode("ascii")
    )
    return bytes(output)


EXAMPLES = [
    "area man thrilled to spend entire weekend doing taxes",
    "nation's dads unveil bold new plan to adjust thermostat",
    "senate passes bipartisan infrastructure bill",
    "local woman just loves being stuck in traffic for 3 hours",
    "study finds regular exercise linked to better sleep",
]

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    use_tuned = st.toggle(f"Use tuned threshold ({det.threshold:.2f})", value=False,
                          help="Off = standard 0.50 cut-off. On = threshold tuned on validation F1.")
    threshold = det.threshold if use_tuned else 0.5
    st.caption(f"Predict sarcastic when probability ≥ {threshold:.2f}")
    st.divider()
    st.markdown("**Model:** GloVe + BiLSTM + attention, fused with VADER sentiment, "
                "POS, style and TF-IDF features.")
    st.markdown("**Trained on:** News Headlines Sarcasm dataset (The Onion vs HuffPost).")

tab_demo, tab_results = st.tabs(["🔍 Try it", "📊 Model performance"])

# ------------------------------------------------------------------ demo tab
with tab_demo:
    st.title("🙃 Sarcasm Detector")
    st.write("Type a news-style headline. The model works best on short headlines, "
             "since that is what it was trained on.")

    if "text" not in st.session_state:
        st.session_state.text = EXAMPLES[0]
    if "applied_example" not in st.session_state:
        st.session_state.applied_example = "—"
    choice = st.selectbox("Or pick an example", ["—"] + EXAMPLES)
    if choice != st.session_state.applied_example:
        st.session_state.applied_example = choice
        if choice != "—":
            st.session_state.text = choice

    news_url = st.text_input(
        "News article URL",
        key="news_url",
        placeholder="https://",
        help="Extracts the heading into the headline box. You can edit it before Analyse. Both models then score that same text.",
    )
    if st.button("Extract heading", disabled=not str(news_url).strip()):
        try:
            from linguistics import extract_news_heading
            heading = extract_news_heading(news_url.strip())
            st.session_state.text = heading[:300]
            st.session_state.heading_error = ""
            st.session_state.heading_note = (
                "The heading was shortened to 300 characters to fit the headline box."
                if len(heading) > 300 else ""
            )
        except Exception as exc:
            st.session_state.heading_error = str(exc)
            st.session_state.heading_note = ""
    if st.session_state.get("heading_error"):
        st.warning(st.session_state.heading_error)
    elif st.session_state.get("heading_note"):
        st.caption(st.session_state.heading_note)

    text = st.text_area("Headline", key="text", height=80, max_chars=300)

    st.markdown(
        """
        <style>
        div[data-testid="column"] .stButton > button,
        div[data-testid="column"] .stDownloadButton > button,
        div[data-testid="column"] div[data-testid="stPopover"] > button {
            width: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    analyse_col, download_col = st.columns(2)
    if analyse_col.button("Analyse", type="primary", width="stretch") and text.strip():
        st.session_state.analysis = det.predict(text, threshold=threshold)
        st.session_state.analysis_text = text
        analyzer, sentiment_error = get_sentiment()
        if analyzer is None:
            st.session_state.sentiment = {
                "error": sentiment_error or "Sentiment model is unavailable.",
            }
        else:
            try:
                st.session_state.sentiment = analyzer.predict(text)
            except Exception as exc:
                st.session_state.sentiment = {"error": str(exc)}

    report_ready = (
        st.session_state.get("analysis") is not None
        and st.session_state.get("analysis_text") == text
    )
    if report_ready:
        report_result = dict(st.session_state.analysis)
        report_result["headline"] = text.strip()
        with download_col.popover("Download the pipeline", width="stretch"):
            pipeline_choice = st.selectbox(
                "Pipeline",
                [
                    "Download sarcasm detection pipeline",
                    "Download sentiment pipeline",
                ],
            )
            if pipeline_choice == "Download sarcasm detection pipeline":
                report_body = pipeline_report(report_result, threshold)
                report_charts = sarcasm_pipeline_charts(report_result)
                report_title = "Sarcasm detection pipeline"
                report_stem = "sarcasm_detection_pipeline"
            else:
                report_body, report_charts = sentiment_pipeline_report(
                    text, st.session_state.get("sentiment") or {}
                )
                report_title = "Sentiment pipeline"
                report_stem = "sentiment_pipeline"
            st.download_button(
                "Download PDF",
                data=_text_to_pdf(report_body, report_charts),
                file_name=report_stem + ".pdf",
                mime="application/pdf",
                width="stretch",
                key="download_" + report_stem + "_pdf",
            )
            st.download_button(
                "Download HTML",
                data=_html_document(report_title, report_body, report_charts),
                file_name=report_stem + ".html",
                mime="text/html",
                width="stretch",
                key="download_" + report_stem + "_html",
            )
    else:
        download_col.button(
            "Download the pipeline",
            width="stretch",
            disabled=True,
            help="Analyse the headline first. The file is the detailed pipeline for that headline.",
        )

    if report_ready:
        r = dict(st.session_state.analysis)
        r["threshold"] = threshold
        r["is_sarcastic"] = r["probability"] >= threshold
        p = r["probability"]

        # --- verdict. Sarcasm text is unchanged. Sentiment is a separate model.
        sentiment = st.session_state.get("sentiment") or {}
        sarcasm_col, sentiment_col = st.columns(2)
        with sarcasm_col:
            if r["is_sarcastic"]:
                st.error(f"### 🙃 Sarcastic  ·  {p:.0%} confidence")
            else:
                st.success(f"### 🙂 Not sarcastic  ·  {1 - p:.0%} confidence")
        with sentiment_col:
            if sentiment.get("error"):
                st.warning("Sentiment model could not score this headline.")
            else:
                name = sentiment["sentiment"]
                confidence = sentiment["confidence"]
                probabilities = sentiment["probabilities"]
                heading = f"### {name.capitalize()} sentiment  ·  {confidence:.0%} confidence"
                if name == "negative":
                    st.error(heading)
                elif name == "positive":
                    st.success(heading)
                else:
                    st.info(heading)
                st.caption(
                    "Separate sentiment model. "
                    f"Negative {probabilities['negative']:.0%} · "
                    f"Neutral {probabilities['neutral']:.0%} · "
                    f"Positive {probabilities['positive']:.0%}"
                )
        st.progress(p, text=f"Sarcasm probability: {p:.3f}")

        # --- attention highlight
        st.subheader("Where the model looked")
        st.caption("Darker = more attention from the BiLSTM attention layer.")
        att = r["attention"]
        top = max(att) if att else 1.0
        spans = []
        for word, a in zip(r["tokens"], att):
            alpha = 0.12 + 0.88 * (a / top)
            spans.append(
                f"<span style='background: rgba(221,132,82,{alpha:.2f}); padding:4px 7px; "
                f"margin:3px; border-radius:6px; display:inline-block; color:#111' "
                f"title='attention {a:.3f}'>{html.escape(word)}</span>")
        st.markdown("<div style='line-height:2.4'>" + "".join(spans) + "</div>",
                    unsafe_allow_html=True)
        if r["unknown_words"]:
            st.caption("Words the model never saw in training: " + ", ".join(r["unknown_words"]))

        # --- NLP signals
        st.subheader("NLP signals")
        f = r["features"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("VADER compound", f"{f['vader_compound']:+.2f}")
        c2.metric("Positive words", f["n_pos_words"])
        c3.metric("Negative words", f["n_neg_words"])
        c4.metric("Sentiment contrast", "Yes" if f["sent_contrast"] else "No",
                  help="Positive and negative words in the same headline - a classic sarcasm cue.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Intensifiers", f["n_intensifiers"])
        c2.metric("Adjective ratio", f"{f['pos_adj_ratio']:.0%}")
        c3.metric("Adverb ratio", f"{f['pos_adv_ratio']:.0%}")
        c4.metric("! / ?", f"{f['n_exclaim']} / {f['n_question']}")

        # --- pipeline steps
        with st.expander("See the preprocessing pipeline"):
            s = r["steps"]
            st.markdown("**1. Cleaned text** (lowercase, contractions expanded, URLs removed)")
            st.code(s["clean_text"], language=None)
            st.markdown("**2. Tokens + POS tags** (spaCy)")
            st.dataframe(pd.DataFrame({"token": s["spacy_tokens"],
                                       "POS": s["pos_tags"].split()}).T,
                         width="stretch")
            st.markdown("**3. Lemmatized, stop words removed** (input to TF-IDF)")
            st.code(s["processed_text"] or "(empty)", language=None)
            st.markdown("**4. Word ids fed to the BiLSTM**")
            st.code(" ".join(r["tokens"]), language=None)

        with st.expander("Detailed pipeline"):
            render_detailed_pipeline(r, threshold)

        with st.expander("Detailed sentiment pipeline"):
            render_sentiment_pipeline(text, sentiment)

    with st.expander("Sentiment document and linguistic analysis"):
        render_sentiment_linguistics(text)

# --------------------------------------------------------------- results tab
with tab_results:
    st.header("Test-set results")
    results_csv = os.path.join(RESULTS_DIR, "results_final.csv")
    if os.path.exists(results_csv):
        res = pd.read_csv(results_csv, index_col=0)
        st.dataframe(res.style.format("{:.4f}").highlight_max(axis=0, color="#fde2cf"),
                     width="stretch")
    else:
        st.info("results/results_final.csv was not found.")

    for img, caption in [("model_comparison.png", "F1 by model"),
                         ("confusion_matrices.png", "Confusion matrices"),
                         ("roc_curves.png", "ROC curves"),
                         ("attention_sarcastic.png", "Attention on detected sarcasm"),
                         ("feature_importance.png", "Which inputs the model relies on"),
                         ("learning_curves.png", "Training curves")]:
        img_path = os.path.join(FIGURES_DIR, img)
        if os.path.exists(img_path):
            st.image(img_path, caption=caption, width="stretch")
