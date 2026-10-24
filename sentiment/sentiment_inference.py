"""Load the trained sentiment BiLSTM and score one headline.

This module does not import or modify the sarcasm detector.
`sarcasm_probability` is accepted so a caller can pass the sarcasm model's
probability. It changes the sentiment result only when a checked calibrator
file is present. With no calibrator, the sentiment result is unchanged.
"""
import json
import os

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from tensorflow import keras

from text import tokenize


class SentimentAnalyzer:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        # Step 1. Read the label order, pad length, and the word -> id map saved at training.
        with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as handle:
            self.config = json.load(handle)
        with open(os.path.join(model_dir, "vocab.json"), encoding="utf-8") as handle:
            raw_index = json.load(handle)
        self.word_index = {word: int(index) for word, index in raw_index.items()}
        self.labels = list(self.config["labels"])
        self.max_len = int(self.config["max_len"])
        # Step 2. Load the BiLSTM checkpoint. This is prediction only; weights are not updated.
        self.model = keras.models.load_model(os.path.join(model_dir, "sentiment_bilstm.keras"))
        # Step 3. A calibrator exists only if sarcasm probability helped on the validation check.
        calibrator_path = os.path.join(model_dir, "sarcasm_calibrator.json")
        self.calibrator = None
        if os.path.exists(calibrator_path):
            with open(calibrator_path, encoding="utf-8") as handle:
                self.calibrator = json.load(handle)

    def encode(self, text):
        # Same rules as training: tokenise, map to ids, unknown words become 1, pad with 0.
        sequence = np.zeros((1, self.max_len), dtype="int32")
        ids = [self.word_index.get(token, 1) for token in tokenize(text)][: self.max_len]
        if ids:
            sequence[0, : len(ids)] = ids
        return sequence

    def _distribution(self, text):
        # training=False turns dropout off. The last layer is a 3-way softmax.
        probabilities = self.model(self.encode(text), training=False).numpy()[0]
        probabilities = np.asarray(probabilities, dtype="float64")
        # Clip and renormalise so a zero probability cannot break a later log.
        probabilities = np.clip(probabilities, 1e-8, None)
        probabilities = probabilities / probabilities.sum()
        return probabilities

    def predict(self, text, sarcasm_probability=None):
        # Softmax first. The sarcasm probability is applied only when a calibrator was saved.
        probabilities = self._distribution(text)
        if sarcasm_probability is not None and self.calibrator is not None:
            probabilities = _apply_calibrator(probabilities, float(sarcasm_probability), self.calibrator)
        # The predicted class is the largest of the three probabilities.
        chosen = int(np.argmax(probabilities))
        return {
            "sentiment": self.labels[chosen],
            "confidence": float(probabilities[chosen]),
            "probabilities": {
                label: float(probabilities[index]) for index, label in enumerate(self.labels)
            },
        }


def _apply_calibrator(probabilities, sarcasm_probability, calibrator):
    """Logistic step fitted on held-out headlines. Absent unless it improved them."""
    # No coefficients means the file is not a fitted calibrator, so the softmax stands.
    if "coef" not in calibrator:
        return probabilities
    # Four inputs: negative, neutral, positive, and the sarcasm probability.
    features = np.array([
        probabilities[0], probabilities[1], probabilities[2], float(sarcasm_probability),
    ], dtype="float64")
    coef = np.asarray(calibrator["coef"], dtype="float64")
    intercept = np.asarray(calibrator["intercept"], dtype="float64")
    logits = coef @ features + intercept
    # Subtract the max before exp so a large logit does not overflow. Then softmax.
    logits = logits - logits.max()
    exp = np.exp(logits)
    return exp / exp.sum()


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    analyzer = SentimentAnalyzer(os.path.join(here, "models"))
    for headline in [
        "company profit rose sharply after the new contract",
        "the firm filed for bankruptcy after sales collapsed",
        "the annual general meeting will be held on tuesday",
        "the test to predict breast cancer relapse is approved",
    ]:
        result = analyzer.predict(headline)
        print(f"{result['sentiment']:8} {result['confidence']:.3f}  {headline}")
