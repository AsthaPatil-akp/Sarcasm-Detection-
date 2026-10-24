"""Download FinancialPhraseBank and normalise it to headline + sentiment.

The labels are human annotations (Malo et al., 2014), not model-generated
labels. Sentences_50Agree keeps every sentence where at least half of the
annotators agreed. That is the largest of the four published agreement cuts.
"""
import io
import json
import os
import zipfile
from urllib.request import Request, urlopen

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
CSV_PATH = os.path.join(DATA_DIR, "financial_phrasebank.csv")
REPORT_PATH = os.path.join(DATA_DIR, "data_report.json")

ZIP_URL = (
    "https://huggingface.co/datasets/financial_phrasebank/resolve/main/data/"
    "FinancialPhraseBank-v1.0.zip"
)
TXT_URL = (
    "https://raw.githubusercontent.com/maxwellsarpong/"
    "NLP-financial-text-processing-dataset/master/Sentences_50Agree.txt"
)
LABELS = ("negative", "neutral", "positive")
TARGET_NAME = "sentences_50agree.txt"


def _fetch(url):
    # Download the raw bytes. The caller decides whether they are a zip or plain text.
    request = Request(url, headers={"User-Agent": "sarcasmnlpdl-sentiment/1.0"})
    with urlopen(request, timeout=120) as response:
        return response.read()


def _read_phrasebank_text(payload, source_name):
    # The published file is latin-1. Try utf-8 first, then latin-1.
    for encoding in ("utf-8", "latin-1"):
        try:
            return payload.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {source_name}")


def _extract_50agree(payload):
    """Return the 50% agreement file from a zip, or the payload itself if it is already text."""
    # A zip starts with the bytes PK. Pull out Sentences_50Agree.txt and ignore the other cuts.
    if payload[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = archive.namelist()
            match = [name for name in names if name.lower().endswith(TARGET_NAME)]
            if not match:
                raise FileNotFoundError(
                    "Sentences_50Agree.txt was not in the zip. Members: " + ", ".join(names[:20])
                )
            member = match[0]
            return archive.read(member), member
    return payload, "Sentences_50Agree.txt"


def parse_phrasebank(text):
    # Each kept line is "sentence@label". The label is the text after the last @.
    rows = []
    skipped = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "@" not in line:
            skipped += 1
            continue
        sentence, label = line.rsplit("@", 1)
        sentence = sentence.strip()
        label = label.strip().lower()
        if not sentence or label not in LABELS:
            skipped += 1
            continue
        rows.append({"headline": sentence, "sentiment": label})
    return rows, skipped


def normalise(rows):
    # Drop blank rows, unknown labels, and any sentence that was given two different labels.
    frame = pd.DataFrame(rows)
    before = len(frame)
    frame["headline"] = frame["headline"].astype(str).str.strip()
    frame["sentiment"] = frame["sentiment"].astype(str).str.strip().str.lower()
    frame = frame.replace({"": pd.NA}).dropna(subset=["headline", "sentiment"])
    frame = frame[frame["sentiment"].isin(LABELS)]
    frame["key"] = frame["headline"].str.lower()
    conflicts = int((frame.groupby("key")["sentiment"].nunique() > 1).sum())
    conflict_keys = set(frame.groupby("key")["sentiment"].nunique().loc[lambda s: s > 1].index)
    frame = frame[~frame["key"].isin(conflict_keys)]
    frame = frame.drop_duplicates(subset=["key"], keep="first")
    frame = frame.drop(columns=["key"]).reset_index(drop=True)
    report = {
        "rows_parsed": before,
        "conflicts_removed": conflicts,
        "rows_after_cleaning": int(len(frame)),
        "label_counts": {label: int((frame["sentiment"] == label).sum()) for label in LABELS},
    }
    return frame, report


def download(force=False):
    os.makedirs(RAW_DIR, exist_ok=True)
    # Reuse the CSV already on disk unless force=True asks for a fresh download.
    if os.path.exists(CSV_PATH) and not force:
        frame = pd.read_csv(CSV_PATH)
        with open(REPORT_PATH, encoding="utf-8") as handle:
            report = json.load(handle)
        return frame, report

    # Step 1. Prefer the Hugging Face zip. If that fails, fetch the plain Sentences_50Agree.txt.
    source = ZIP_URL
    try:
        payload = _fetch(ZIP_URL)
        raw, member = _extract_50agree(payload)
        source = ZIP_URL + " -> " + member
    except Exception as zip_error:
        payload = _fetch(TXT_URL)
        raw, member = payload, "Sentences_50Agree.txt"
        source = TXT_URL + f" (zip failed: {zip_error})"

    # Step 2. Decode, keep a local copy of the source text, then parse and drop conflicts.
    text, encoding = _read_phrasebank_text(raw, member)
    raw_path = os.path.join(RAW_DIR, "Sentences_50Agree.txt")
    with open(raw_path, "w", encoding="utf-8") as handle:
        handle.write(text)

    rows, skipped = parse_phrasebank(text)
    frame, report = normalise(rows)
    report.update({
        "dataset": "FinancialPhraseBank",
        "citation": "Malo, Sinha, Korhonen, Wallenius, Takala (2014)",
        "agreement": "Sentences_50Agree: at least 50% of human annotators agreed",
        "license": "CC BY-NC-SA 3.0",
        "source": source,
        "encoding": encoding,
        "malformed_lines_skipped": skipped,
        "columns": ["headline", "sentiment"],
        "label_meaning": (
            "Investor viewpoint on a financial-news sentence: "
            "positive, negative, or neutral effect on the stock price. "
            "Sentiment that is not financially relevant was annotated as neutral."
        ),
    })
    # Step 3. Write financial_phrasebank.csv. data_lib reads this file when it builds headlines.
    frame.to_csv(CSV_PATH, index=False)
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return frame, report


if __name__ == "__main__":
    data, info = download(force=True)
    print(json.dumps(info, indent=2))
    print(data.head(3).to_string(index=False))
