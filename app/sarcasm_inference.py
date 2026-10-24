"""
Sarcasm detection - inference pipeline.

Re-creates EXACTLY the preprocessing used in training (Phases 3-5) so a new
headline is turned into the same two inputs the hybrid model was trained on:
  Branch A: padded word-id sequence   (clean_text -> vocab -> pad)
  Branch B: scaled NLP feature vector (VADER + POS + style + TF-IDF/SVD)

Needs these files in the models folder:
  hybrid_final.keras, nlp_feature_pipeline.joblib, dl_vocab.pkl, training_info.json
"""
import os, re, json, pickle

import numpy as np
import joblib
import keras
from keras import layers, ops
import spacy
import contractions
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ---------------------------------------------------------------- model layer
# Must match Phase 7 (cell 49) so Keras can load the saved model.
@keras.saving.register_keras_serializable()
class AttentionPooling(layers.Layer):
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.supports_masking = True

    def build(self, input_shape):
        d = input_shape[-1]
        self.W = self.add_weight(name="W", shape=(d, self.units), initializer="glorot_uniform")
        self.b = self.add_weight(name="b", shape=(self.units,), initializer="zeros")
        self.u = self.add_weight(name="u", shape=(self.units, 1), initializer="glorot_uniform")

    def call(self, h, mask=None):
        score = ops.squeeze(ops.matmul(ops.tanh(ops.matmul(h, self.W) + self.b), self.u), -1)
        if mask is not None:
            score = ops.where(mask, score, ops.full_like(score, -1e9))
        alpha = ops.softmax(score, axis=-1)
        return ops.sum(h * ops.expand_dims(alpha, -1), axis=1), alpha

    def compute_mask(self, inputs, mask=None):
        return [None, None]

    def get_config(self):
        return {**super().get_config(), "units": self.units}


# ------------------------------------------------------ Phase 3: preprocessing
URL_RE = re.compile(r"https?://\S+|www\.\S+")
KEEP_WORDS = {"not", "no", "nor", "never", "so", "very", "just", "too", "only",
              "more", "most", "again", "all", "against", "same", "few", "over",
              "off", "what", "why", "how", "really", "such", "own"}
KEEP_PUNCT = {"!", "?", "..."}


def clean_text(text):
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = contractions.fix(text)
    text = re.sub(r"\.{2,}|…", " ... ", text)
    text = re.sub(r"([!?])", r" \1 ", text)
    text = re.sub(r"[^a-z0-9\s!?.']", " ", text)
    text = re.sub(r"(?<!\.)\.(?!\.)", " ", text)
    text = re.sub(r"(?<![a-z])'|'(?![a-z])", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def cleaning_steps(text):
    """Same operations as clean_text, kept as a list so the UI can show each one."""
    cur = str(text)
    steps = [("Original", cur)]
    cur = cur.lower()
    steps.append(("Lowercase", cur))
    cur = URL_RE.sub(" ", cur)
    steps.append(("URLs removed", cur))
    cur = re.sub(r"<.*?>", " ", cur)
    steps.append(("HTML tags removed", cur))
    cur = cur.replace("’", "'").replace("‘", "'")
    steps.append(("Curly quotes normalised", cur))
    cur = contractions.fix(cur)
    steps.append(("Contractions expanded", cur))
    cur = re.sub(r"\.{2,}|…", " ... ", cur)
    steps.append(("Ellipsis kept as its own token", cur))
    cur = re.sub(r"([!?])", r" \1 ", cur)
    steps.append(("! and ? separated so they are kept", cur))
    cur = re.sub(r"[^a-z0-9\s!?.']", " ", cur)
    steps.append(("Every other character removed", cur))
    cur = re.sub(r"(?<!\.)\.(?!\.)", " ", cur)
    steps.append(("Single periods removed", cur))
    cur = re.sub(r"(?<![a-z])'|'(?![a-z])", " ", cur)
    steps.append(("Stray apostrophes removed", cur))
    cur = re.sub(r"\s+", " ", cur).strip()
    steps.append(("Extra spaces collapsed", cur))
    return steps


# --------------------------------------------------- Phase 4: NLP features
INTENSIFIERS = {"so", "very", "really", "totally", "absolutely", "completely", "extremely",
                "literally", "truly", "incredibly", "utterly", "just", "definitely", "super"}
POS_OF_INTEREST = ["ADJ", "ADV", "INTJ", "VERB", "NOUN", "PROPN", "PRON", "NUM"]


def pos_features(pos_string, clean):
    tags = pos_string.split()
    n = max(len(tags), 1)
    feats = {f"pos_{p.lower()}_ratio": tags.count(p) / n for p in POS_OF_INTEREST}
    feats["n_intensifiers"] = sum(w in INTENSIFIERS for w in clean.split())
    return feats


def style_features(text):
    words = text.split()
    return {
        "n_exclaim": text.count("!"),
        "n_question": text.count("?"),
        "n_ellipsis": len(re.findall(r"\.{2,}|…", text)),
        "n_quotes": len(re.findall(r"['\"‘’“”](?![a-z])|(?<![a-z])['\"‘’“”]", text)),
        "has_digit": int(bool(re.search(r"\d", text))),
        "word_count": len(words),
        "avg_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
    }


# ------------------------------------------------------------- the detector
class SarcasmDetector:
    def __init__(self, artifact_dir="."):
        p = lambda f: os.path.join(artifact_dir, f)

        self.model = keras.models.load_model(p("hybrid_final.keras"))
        self.att_model = keras.Model(self.model.inputs,
                                     self.model.get_layer("attention").output[1])

        pipe = joblib.load(p("nlp_feature_pipeline.joblib"))
        self.tfidf, self.svd, self.scaler = pipe["tfidf"], pipe["svd"], pipe["scaler"]
        self.feature_names = pipe["feature_names"]
        self.hand_names = [n for n in self.feature_names if not n.startswith("svd_")]

        with open(p("dl_vocab.pkl"), "rb") as f:
            vocab = pickle.load(f)
        self.word_index, self.max_len = vocab["word_index"], vocab["max_len"]
        self.index_word = {i: w for w, i in self.word_index.items()}

        self.threshold = 0.5
        if os.path.exists(p("training_info.json")):
            with open(p("training_info.json")) as f:
                self.threshold = json.load(f).get("best_threshold", 0.5)

        try:
            from nltk.corpus import stopwords
            stops = stopwords.words("english")
        except LookupError:
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords
            stops = stopwords.words("english")
        self.custom_stops = set(stops) - KEEP_WORDS

        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        self.vader = SentimentIntensityAnalyzer()
        self.probe = self._build_probe()

    # ---- Phase 3: spaCy tokens, POS tags, lemmas, stop words
    def preprocess(self, headline):
        clean = clean_text(headline)
        doc = self.nlp(clean)
        spacy_tokens = [t.text for t in doc if not t.is_space]
        pos = " ".join(t.pos_ for t in doc if not t.is_space)
        tokens = []
        for t in doc:
            if t.is_space:
                continue
            if t.is_punct:
                if t.text in KEEP_PUNCT:
                    tokens.append(t.text)
                continue
            lemma = t.lemma_.lower()
            if lemma in self.custom_stops or t.text in self.custom_stops or lemma.startswith("'"):
                continue
            tokens.append(lemma)
        return clean, " ".join(tokens), pos, spacy_tokens

    # ---- Phase 4: sentiment features (VADER on the raw headline)
    def sentiment_features(self, text):
        s = self.vader.polarity_scores(text)
        words = re.findall(r"[a-z']+", text.lower())
        word_scores = [self.vader.lexicon[w] for w in words if w in self.vader.lexicon]
        n_pos = sum(1 for v in word_scores if v > 0)
        n_neg = sum(1 for v in word_scores if v < 0)
        return {
            "vader_pos": s["pos"], "vader_neg": s["neg"],
            "vader_neu": s["neu"], "vader_compound": s["compound"],
            "n_pos_words": n_pos, "n_neg_words": n_neg,
            "sent_contrast": int(n_pos > 0 and n_neg > 0),
            "sent_gap": max(word_scores, default=0) - min(word_scores, default=0),
        }

    # ---- Phase 4: full Branch B vector
    def feature_vector(self, raw, clean, processed, pos):
        hand = {**self.sentiment_features(raw), **pos_features(pos, clean), **style_features(raw)}
        hand_vec = np.array([[hand[n] for n in self.hand_names]], dtype="float64")
        svd_vec = self.svd.transform(self.tfidf.transform([processed]))
        return self.scaler.transform(np.hstack([hand_vec, svd_vec])), hand

    # ---- Phase 5: Branch A sequence
    def sequence(self, clean):
        tokens = clean.replace("'s", " 's").split()
        ids = [self.word_index.get(w, 1) for w in tokens][: self.max_len]
        padded = np.zeros((1, self.max_len), dtype="int32")
        padded[0, : len(ids)] = ids
        return padded, tokens[: self.max_len]

    def _build_probe(self):
        try:
            return keras.Model(
                self.model.inputs,
                [
                    self.model.get_layer("glove_embedding").output,
                    self.model.get_layer("bilstm").output,
                    self.model.get_layer("attention").output[0],
                    self.model.get_layer("feat_dense").output,
                    self.model.get_layer("fusion_dense").output,
                ],
            )
        except Exception:
            return None

    def _layer_facts(self):
        facts = {}
        try:
            lstm = self.model.get_layer("bilstm").layer
            facts["lstm_units"] = int(lstm.units)
            facts["lstm_activation"] = lstm.activation.__name__
            facts["lstm_recurrent_activation"] = lstm.recurrent_activation.__name__
        except Exception:
            facts["lstm_units"] = 64
            facts["lstm_activation"] = "tanh"
            facts["lstm_recurrent_activation"] = "sigmoid"
        for key, layer_name, attr in (
            ("embed_dropout", "embed_dropout", "rate"),
            ("feat_dropout", "feat_dropout", "rate"),
            ("fusion_dropout", "fusion_dropout", "rate"),
        ):
            try:
                facts[key] = float(getattr(self.model.get_layer(layer_name), attr))
            except Exception:
                facts[key] = None
        for key, layer_name in (("feat_activation", "feat_dense"),
                                ("fusion_activation", "fusion_dense"),
                                ("output_activation", "sarcasm_prob")):
            try:
                facts[key] = self.model.get_layer(layer_name).activation.__name__
            except Exception:
                facts[key] = None
        try:
            facts["attention_units"] = int(self.model.get_layer("attention").units)
        except Exception:
            facts["attention_units"] = 64
        try:
            cfg = self.model.get_layer("glove_embedding").get_config()
            facts["vocab_size"] = int(cfg.get("input_dim"))
            facts["embed_dim"] = int(cfg.get("output_dim"))
        except Exception:
            facts["vocab_size"] = None
            facts["embed_dim"] = 100
        return facts

    def _trace(self, original, raw, clean, processed, pos, seq, tokens, feats, readable):
        steps = cleaning_steps(original)
        if steps[-1][1] != clean:
            steps.append(("Text passed forward", clean))

        doc = self.nlp(clean)
        token_rows = []
        for t in doc:
            if t.is_space:
                continue
            if t.is_punct:
                if t.text in KEEP_PUNCT:
                    decision = "Kept. ! ? and ... are tone marks, not thrown away."
                else:
                    decision = "Dropped. Other punctuation is removed before TF-IDF."
                token_rows.append({
                    "token": t.text, "pos": t.pos_, "lemma": t.lemma_, "decision": decision,
                })
                continue
            lemma = t.lemma_.lower()
            if lemma in self.custom_stops or t.text in self.custom_stops or lemma.startswith("'"):
                decision = "Dropped. It is a stop word, and it is not on the keep-list (negation and intensifiers stay)."
            else:
                decision = "Kept. Lemma goes into the TF-IDF text."
            token_rows.append({
                "token": t.text, "pos": t.pos_, "lemma": lemma, "decision": decision,
            })

        tags = pos.split()
        n_tags = max(len(tags), 1)
        pos_rows = [{
            "pos": p,
            "count": tags.count(p),
            "ratio": tags.count(p) / n_tags,
            "calculation": f"{tags.count(p)} / {n_tags}",
        } for p in POS_OF_INTEREST]

        scored = []
        for w in re.findall(r"[a-z']+", raw.lower()):
            if w in self.vader.lexicon:
                scored.append({"word": w, "lexicon_score": float(self.vader.lexicon[w])})
        scores = [row["lexicon_score"] for row in scored]
        intensifiers_found = [w for w in clean.split() if w in INTENSIFIERS]

        tf = self.tfidf.transform([processed])
        names = self.tfidf.get_feature_names_out()
        coo = tf.tocoo()
        tfidf_rows = []
        ngram_names = {1: "unigram", 2: "bigram", 3: "trigram"}
        for col, weight in sorted(zip(coo.col, coo.data), key=lambda pair: -pair[1]):
            term = str(names[col])
            n_words = len(term.split())
            tfidf_rows.append({
                "term": term,
                "n-gram": ngram_names.get(n_words, f"{n_words}-gram"),
                "tfidf": float(weight),
                "idf": float(self.tfidf.idf_[col]),
            })

        svd_vec = self.svd.transform(tf).ravel()
        raw_values = [float(readable[name]) for name in self.hand_names] + [float(v) for v in svd_vec]
        scaled = [float(v) for v in np.asarray(feats).ravel()]
        means = [float(v) for v in self.scaler.mean_]
        scales = [float(v) for v in self.scaler.scale_]
        feature_rows = []
        if len(self.feature_names) == len(raw_values) == len(scaled) == len(means) == len(scales):
            for name, raw_v, mean, scale, scaled_v in zip(
                self.feature_names, raw_values, means, scales, scaled
            ):
                feature_rows.append({
                    "feature": name,
                    "raw": raw_v,
                    "training_mean": mean,
                    "training_scale": scale,
                    "scaled": scaled_v,
                })

        ids = [int(i) for i in seq.ravel()]
        word_rows = []
        for pos_i, word_id in enumerate(ids):
            if pos_i < len(tokens):
                word = tokens[pos_i]
                known = word in self.word_index
            else:
                word = ""
                known = False
            word_rows.append({
                "position": pos_i + 1,
                "token": word if word else "(pad)",
                "word_id": word_id,
                "known": known if word else False,
                "role": "word" if word else "padding, masked out of the LSTM and attention",
            })

        internals = {}
        if self.probe is not None:
            x = [ops.convert_to_tensor(seq), ops.convert_to_tensor(feats.astype("float32"))]
            emb, lstm_out, context, feat_h, fusion_h = self.probe(x, training=False)
            emb = np.asarray(ops.convert_to_numpy(emb))[0]
            lstm_out = np.asarray(ops.convert_to_numpy(lstm_out))[0]
            context = np.asarray(ops.convert_to_numpy(context))[0]
            feat_h = np.asarray(ops.convert_to_numpy(feat_h))[0]
            fusion_h = np.asarray(ops.convert_to_numpy(fusion_h))[0]
            for row, vec_e, vec_h in zip(word_rows, emb, lstm_out):
                if row["token"] == "(pad)":
                    row["embedding_l2"] = 0.0
                    row["bilstm_l2"] = 0.0
                else:
                    row["embedding_l2"] = float(np.linalg.norm(vec_e))
                    row["bilstm_l2"] = float(np.linalg.norm(vec_h))
            internals = {
                "context_l2": float(np.linalg.norm(context)),
                "context_dim": int(context.shape[-1]),
                "feat_hidden": [float(v) for v in feat_h],
                "feat_relu_zeros": int(np.sum(np.asarray(feat_h) == 0)),
                "fusion_hidden": [float(v) for v in fusion_h],
                "fusion_relu_zeros": int(np.sum(np.asarray(fusion_h) == 0)),
            }

        return {
            "cleaning": steps,
            "tokens": token_rows,
            "pos_rows": pos_rows,
            "pos_total": n_tags,
            "vader_hits": scored,
            "vader_gap_max": float(max(scores, default=0)),
            "vader_gap_min": float(min(scores, default=0)),
            "intensifiers_found": intensifiers_found,
            "tfidf_rows": tfidf_rows,
            "tfidf_nonzero": len(tfidf_rows),
            "svd_values": [float(v) for v in svd_vec],
            "feature_rows": feature_rows,
            "word_rows": word_rows,
            "internals": internals,
            "layers": self._layer_facts(),
        }

    def predict(self, headline, threshold=None):
        # Training headlines were all lowercase, so lowercase user input to match
        original = str(headline).strip()
        raw = original.lower()
        clean, processed, pos, spacy_tokens = self.preprocess(raw)
        seq, tokens = self.sequence(clean)
        feats, readable = self.feature_vector(raw, clean, processed, pos)

        # Direct call (not .predict) is faster for one input and avoids retracing
        x = [ops.convert_to_tensor(seq), ops.convert_to_tensor(feats.astype("float32"))]
        prob = float(ops.convert_to_numpy(self.model(x, training=False))[0, 0])
        alpha = ops.convert_to_numpy(self.att_model(x, training=False))[0][: len(tokens)]
        t = self.threshold if threshold is None else threshold
        try:
            trace = self._trace(original, raw, clean, processed, pos, seq, tokens, feats, readable)
        except Exception as exc:
            trace = {"error": str(exc)}
        return {
            "probability": prob,
            "is_sarcastic": prob >= t,
            "threshold": t,
            "tokens": tokens,
            "attention": alpha.tolist(),
            "unknown_words": [w for w in tokens if w not in self.word_index],
            "features": readable,
            "steps": {"clean_text": clean, "processed_text": processed,
                      "pos_tags": pos, "spacy_tokens": spacy_tokens},
            "trace": trace,
        }


if __name__ == "__main__":
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    det = SarcasmDetector(os.path.join(_root, "models"))
    for h in ["area man thrilled to spend entire weekend doing taxes",
              "senate passes bipartisan infrastructure bill"]:
        r = det.predict(h)
        print(f"{r['probability']:.3f}  {'SARCASTIC' if r['is_sarcastic'] else 'not sarcastic':<14} {h}")
