"""Sentiment preprocessing.

This file is not used by the sarcasm detector. Capital letters, exclamation
marks, question marks, and negation words are kept because they change sentiment.
"""
import re

import spacy

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_NLP = None

# Content words are kept. Function words are dropped unless they mark negation or contrast.
CONTENT_POS = {"ADJ", "ADV", "VERB", "NOUN", "PROPN", "INTJ", "NUM"}
NEGATIONS = {
    "not", "no", "never", "nor", "neither", "n't", "cannot", "cant", "without",
    "nobody", "nothing", "nowhere", "hardly", "barely", "n't",
    "dont", "doesn't", "didn't", "won't", "isn't", "aren't", "wasn't", "weren't",
    "don't", "doesn't", "didn't", "won't", "isn't", "aren't",
}
CONTRAST = {"but", "however", "although", "though", "yet"}


def _nlp():
    # Load spaCy once. Parser and named-entity steps are off; only the tagger and lemmatiser run.
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    return _NLP


def prepare(text):
    """Strip URLs and HTML. Leave case and ! ? in place for the tagger."""
    # Curly quotes become straight quotes so contractions match the negation list.
    text = str(text).replace("\u2019", "'").replace("\u2018", "'")
    text = _URL_RE.sub(" ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_negation(token):
    # Match the surface form, the lemma, or a contracted n't such as "wasn't".
    lower = token.text.lower()
    lemma = token.lemma_.lower()
    return lower in NEGATIONS or lemma in NEGATIONS or lower.endswith("n't")


def tokens_from_doc(doc):
    """Lemma tokens, with negation marked on the next content word."""
    tokens = []
    # negate_next is set by "not" / "never" and consumed by the following content lemma.
    negate_next = False
    for token in doc:
        # Keep ! and ? as their own tokens. They reset negation so it does not cross them.
        if token.text in {"!", "?"}:
            tokens.append(token.text)
            negate_next = False
            continue
        # Other punctuation and spaces are dropped, and they also reset negation.
        if token.is_space or (token.is_punct and token.text not in {"!", "?"}):
            negate_next = False
            continue
        # Negation is kept as the token "not", and the next content word will be marked.
        if _is_negation(token):
            tokens.append("not")
            negate_next = True
            continue
        lemma = token.lemma_.lower().strip()
        lower = token.text.lower()
        # Contrast words (but, however, ...) are kept and clear the negation flag.
        if lemma in CONTRAST or lower in CONTRAST:
            tokens.append(lemma or lower)
            negate_next = False
            continue
        # Drop function words: determiners, prepositions, and other non-content tags.
        if token.pos_ not in CONTENT_POS:
            continue
        if token.is_stop:
            continue
        if not lemma:
            continue
        # An ALL-CAPS content word longer than 2 letters adds an extra "allcaps" token.
        if token.text.isupper() and token.text.isalpha() and len(token.text) > 2:
            tokens.append("allcaps")
        # "not good" becomes the tokens not, not_good. The plain lemma is not also stored.
        if negate_next:
            tokens.append("not_" + lemma)
            negate_next = False
        else:
            tokens.append(lemma)
    return tokens


def tokenize(text):
    # One headline: clean, tag with spaCy, then apply the token rules above.
    prepared = prepare(text)
    if not prepared:
        return []
    return tokens_from_doc(_nlp()(prepared))


def tokenize_many(texts):
    # Same rules as tokenize(), batched through spaCy so training does not tag one row at a time.
    prepared = [prepare(text) for text in texts]
    if not prepared:
        return []
    docs = _nlp().pipe(prepared, batch_size=256)
    return [tokens_from_doc(doc) if item else [] for doc, item in zip(docs, prepared)]
