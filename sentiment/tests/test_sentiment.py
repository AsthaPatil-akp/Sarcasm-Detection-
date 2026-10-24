"""Tests for the sentiment module only. They do not load the sarcasm model."""
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from data_lib import _drop_near_test_leaks, coarse_valence, emobank_news_polarity, norm_key
from linguistics import (
    body_from_html,
    extract_docx,
    extract_pdf,
    heading_from_html,
    linguistic_analysis,
)
from text import tokenize


class PreprocessTests(unittest.TestCase):
    # "not a good result!" must keep not, mark the next content word, and keep the exclamation.
    def test_negation_is_kept_and_marked_on_the_next_content_word(self):
        tokens = tokenize("This is not a good result!")
        self.assertIn("not", tokens)
        self.assertIn("not_good", tokens)
        self.assertIn("!", tokens)
        self.assertNotIn("good", tokens)
        self.assertNotIn("is", tokens)

    # "never" is on the negation list, so it becomes "not" and marks the following lemma.
    def test_never_is_a_negation(self):
        tokens = tokenize("The patients were never happy")
        self.assertIn("not", tokens)
        self.assertTrue(any(token.startswith("not_") for token in tokens))

    # Question marks are sentiment cues and must survive tokenising.
    def test_question_mark_is_kept(self):
        self.assertIn("?", tokenize("Is the bill approved?"))


class LabelAndLeakTests(unittest.TestCase):
    # SemEval edges: -50 is negative, just above -50 is neutral, 50 and above are positive.
    def test_official_valence_bins(self):
        self.assertEqual(coarse_valence(-50), "negative")
        self.assertEqual(coarse_valence(-49.9), "neutral")
        self.assertEqual(coarse_valence(0), "neutral")
        self.assertEqual(coarse_valence(50), "positive")
        self.assertEqual(coarse_valence(100), "positive")

    def test_emobank_newspaper_bins_leave_the_middle_unlabeled(self):
        self.assertEqual(emobank_news_polarity(2.5), "negative")
        self.assertEqual(emobank_news_polarity(2.51), "")
        self.assertEqual(emobank_news_polarity(3.0), "")
        self.assertEqual(emobank_news_polarity(3.49), "")
        self.assertEqual(emobank_news_polarity(3.5), "positive")

    # A train headline that nearly copies a test headline is dropped. The test row stays.
    def test_near_duplicate_of_a_test_headline_leaves_the_test_row(self):
        frame = pd.DataFrame([
            {"headline": "Senate passes the infrastructure bill today", "sentiment": "neutral", "split": "semeval_test"},
            {"headline": "Senate passes the infrastructure bill today in full", "sentiment": "positive", "split": "train"},
            {"headline": "Local team wins the championship game", "sentiment": "positive", "split": "train"},
        ])
        kept, dropped = _drop_near_test_leaks(frame, threshold=0.6)
        self.assertGreaterEqual(dropped, 1)
        self.assertIn("semeval_test", set(kept["split"]))
        self.assertNotIn(
            norm_key("Senate passes the infrastructure bill today in full"),
            set(kept.loc[kept["split"] == "train", "headline"].map(norm_key)),
        )

    # The saved table has no target-level labels, and no train key equals a test key.
    def test_saved_headline_table_has_no_target_labels_and_no_test_leak(self):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "headlines.csv")
        if not os.path.exists(path):
            self.skipTest("headlines.csv is built by data_lib")
        frame = pd.read_csv(path)
        self.assertNotIn("target", set(frame["label_level"]))
        test_keys = set(frame.loc[frame["split"].isin(["fpb_test", "semeval_test", "news_test"]), "headline"].map(norm_key))
        train_keys = set(frame.loc[frame["split"] == "train", "headline"].map(norm_key))
        self.assertTrue(test_keys.isdisjoint(train_keys))


class ModelTests(unittest.TestCase):
    # Vocabulary comes only from the lists passed in. An unseen word encodes as id 1.
    def test_vocab_is_built_from_the_given_headlines_only(self):
        from train import build_vocab, encode
        index = build_vocab([["profit", "rose"], ["profit", "rose"], ["loss", "fell"]])
        self.assertNotIn("secret", index)
        self.assertGreater(index["profit"], 1)
        encoded = encode([["profit", "secret"]], index, 4)
        self.assertEqual(int(encoded[0, 0]), index["profit"])
        self.assertEqual(int(encoded[0, 1]), 1)

    # A tiny BiLSTM can train one epoch, save, load, and ignore sarcasm probability when no calibrator exists.
    def test_checkpoint_save_load_and_sarcasm_fallback(self):
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        from tensorflow import keras
        from tensorflow.keras import layers
        from sentiment_inference import SentimentAnalyzer

        labels = ["negative", "neutral", "positive"]
        index = {"profit": 2, "rose": 3, "loss": 4, "fell": 5}
        max_len = 6
        matrix = np.random.normal(0, 0.05, (len(index) + 2, 16)).astype("float32")
        matrix[0] = 0.0
        model = keras.Sequential([
            layers.Input(shape=(max_len,), dtype="int32"),
            layers.Embedding(len(index) + 2, 16, embeddings_initializer=keras.initializers.Constant(matrix), mask_zero=True),
            layers.Bidirectional(layers.LSTM(4)),
            layers.Dense(3, activation="softmax"),
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        x = np.array([[2, 3, 0, 0, 0, 0], [4, 5, 0, 0, 0, 0], [2, 3, 0, 0, 0, 0], [4, 5, 0, 0, 0, 0]], dtype="int32")
        y = np.array([2, 0, 2, 0])
        # Two epochs on four rows. The second loss should not blow up compared with the first.
        first = model.fit(x, y, epochs=1, verbose=0).history["loss"][0]
        second = model.fit(x, y, epochs=1, verbose=0).history["loss"][0]
        self.assertLessEqual(second, first + 1.0)

        folder = tempfile.mkdtemp()
        try:
            model.save(os.path.join(folder, "sentiment_bilstm.keras"))
            import json
            with open(os.path.join(folder, "vocab.json"), "w", encoding="utf-8") as handle:
                json.dump(index, handle)
            with open(os.path.join(folder, "config.json"), "w", encoding="utf-8") as handle:
                json.dump({"labels": labels, "max_len": max_len}, handle)
            analyzer = SentimentAnalyzer(folder)
            alone = analyzer.predict("profit rose")
            with_sarcasm = analyzer.predict("profit rose", sarcasm_probability=0.99)
            self.assertIn(alone["sentiment"], labels)
            self.assertEqual(set(alone["probabilities"]), set(labels))
            self.assertAlmostEqual(sum(alone["probabilities"].values()), 1.0, places=4)
            self.assertEqual(alone["sentiment"], with_sarcasm["sentiment"])
            self.assertAlmostEqual(alone["confidence"], with_sarcasm["confidence"], places=5)
        finally:
            shutil.rmtree(folder)


class LinguisticsTests(unittest.TestCase):
    # The shared headline comes from the article title, and a publisher suffix on <title> is dropped.
    def test_heading_prefers_open_graph_and_strips_a_site_suffix(self):
        page = """
        <html><head>
          <meta property="og:title" content="Senate passes the infrastructure bill"/>
          <title>Ignored title | Reuters</title>
        </head><body><h1>Ignored h1</h1></body></html>
        """
        self.assertEqual(heading_from_html(page), "Senate passes the infrastructure bill")
        titled = "<html><head><title>Local plant to cut 200 jobs | Reuters</title></head><body></body></html>"
        self.assertEqual(heading_from_html(titled), "Local plant to cut 200 jobs")

    # Webpage extraction keeps article paragraphs and leaves navigation and scripts out.
    def test_webpage_body_keeps_paragraphs_only(self):
        page = """
        <html><body>
          <nav><p>Home World Business</p></nav>
          <article>
            <p>The company reported a sharp rise in profit after the morning session.</p>
            <script>var secret = 1;</script>
            <p>Too short</p>
          </article>
        </body></html>
        """
        body = body_from_html(page)
        self.assertIn("sharp rise in profit", body)
        self.assertNotIn("secret", body)
        self.assertNotIn("Home World", body)

    # A .docx is a zip of Word XML. Paragraph text is the only thing extracted.
    def test_docx_paragraphs_are_read(self):
        import io
        import zipfile
        xml = (
            "<?xml version='1.0' encoding='UTF-8'?>"
            "<w:document xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'>"
            "<w:body><w:p><w:r><w:t>Profit rose sharply.</w:t></w:r></w:p></w:body></w:document>"
        )
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("word/document.xml", xml)
        self.assertEqual(extract_docx(buffer.getvalue()), "Profit rose sharply.")

    # The PDF text layer is what gets extracted. A page with no text layer is refused.
    def test_pdf_text_layer_is_read(self):
        content = b"BT /F1 12 Tf 20 80 Td (Profit rose sharply.) Tj ET"
        pieces = [
            b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n",
            b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n",
            b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 144]/Contents 4 0 R"
            b"/Resources<</Font<</F1 5 0 R>>>>>>endobj\n",
            b"4 0 obj<</Length %d>>stream\n%s\nendstream\nendobj\n" % (len(content), content),
            b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n",
        ]
        pdf = bytearray(b"%PDF-1.4\n")
        offsets = [0]
        for piece in pieces:
            offsets.append(len(pdf))
            pdf += piece
        xref = len(pdf)
        pdf += b"xref\n0 6\n0000000000 65535 f \n"
        for offset in offsets[1:]:
            pdf += f"{offset:010d} 00000 n \n".encode("ascii")
        pdf += f"trailer<</Size 6/Root 1 0 R>>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
        self.assertIn("Profit rose sharply", extract_pdf(bytes(pdf)))

    # POS chunks, entities, and semantic roles are labels about the sentence, not model features.
    def test_chunks_entities_and_roles_describe_the_sentence(self):
        result = linguistic_analysis("Apple opened a new store in London but did not raise prices.")
        noun_phrases = [row["chunk"] for row in result["chunks"] if row["kind"] == "noun phrase"]
        self.assertTrue(any("store" in phrase for phrase in noun_phrases))
        labels = {row["label"] for row in result["entities"]}
        self.assertIn("ORG", labels)
        self.assertIn("GPE", labels)
        self.assertTrue(result["semantics"])
        self.assertEqual(result["semantics"][0]["negated"], "yes")

    # Tokenising still marks negation after the full spaCy model has been loaded for display.
    def test_display_parser_does_not_change_sentiment_tokens(self):
        linguistic_analysis("Apple opened a store in London.")
        tokens = tokenize("This is not a good result!")
        self.assertIn("not_good", tokens)
        self.assertNotIn("good", tokens)


if __name__ == "__main__":
    unittest.main()
