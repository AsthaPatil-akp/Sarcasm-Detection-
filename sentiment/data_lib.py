"""Build the sentiment tables.

Headline and sentence labels are kept apart from target-level labels.
NewsMTSC polarity is sentiment toward a mentioned person, so those rows
are stored for audit and are not used as headline training labels.
"""
import html
import json
import os
import random
import re

import pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
HEADLINES_PATH = os.path.join(DATA_DIR, "headlines.csv")
TARGETS_PATH = os.path.join(DATA_DIR, "target_sentiment.csv")
REPORT_PATH = os.path.join(DATA_DIR, "data_report.json")
FPB_PATH = os.path.join(DATA_DIR, "financial_phrasebank.csv")

LABELS = ("negative", "neutral", "positive")
LOCKED_TEST_SPLITS = {"fpb_test", "semeval_test", "news_test"}
TOKEN_RE = re.compile(r"[a-z0-9']+")
INSTANCE_RE = re.compile(r'<instance id="(\d+)">(.*?)</instance>', re.DOTALL)

# SemEval-2007 Task 14 coarse valence bins, from the task README:
# [-100, -50] negative, (-50, 50) neutral, [50, 100] positive.
REJECTED_SOURCES = [
    {
        "name": "dnakhla/us-news-headlines-enriched",
        "rows": 143142,
        "reason": (
            "Sentiment labels are produced by cardiffnlp/twitter-roberta-base-sentiment-latest. "
            "They are model output, not human labels, so they are not used as ground truth."
        ),
    },
    {
        "name": "MASHXD/investing_financial_news_headlines",
        "rows": 13259,
        "reason": (
            "The card does not say whether the labels are human, market-direction, or automatic. "
            "Undocumented labels are not used as ground truth."
        ),
    },
    {
        "name": "Okcan/news-headlines-classifier",
        "rows": 350,
        "reason": "Too small, and the card does not identify who assigned the sentiment labels.",
    },
]


def norm_tokens(text):
    # Lowercase word pieces used only for matching headlines, not as model tokens.
    return TOKEN_RE.findall(str(text).lower())


def norm_key(text):
    # One string key so the same headline from two sources can be compared.
    return " ".join(norm_tokens(text))


def group_key(text):
    # First four words. SemEval headlines that share this prefix stay in one split.
    return " ".join(norm_tokens(text)[:4])


def jaccard(left, right):
    # Overlap of two token sets: shared words divided by the union. 1 means the same set.
    if not left or not right:
        return 0.0
    shared = len(left & right)
    return shared / len(left | right)


def coarse_valence(value):
    """Map a mean human valence in [-100, 100] with the SemEval coarse bins."""
    if value <= -50:
        return "negative"
    if value >= 50:
        return "positive"
    return "neutral"


def emobank_news_polarity(value):
    """Map EmoBank's 1–5 valence. 3 is the neutral midpoint.

    Newspaper ratings sit in a narrow band around 3. The middle band is left
    unlabeled. At or below 2.5 is the lower half of the negative side. At or
    above 3.5 is the upper half of the positive side.
    """
    if value <= 2.5:
        return "negative"
    if value >= 3.5:
        return "positive"
    return ""


def original_fpb_splits(frame):
    """The split the first sentiment model was trained and tested on."""
    # 70% train, then the remaining 30% is cut in half: 15% validation, 15% fpb_test.
    # random_state 42 and stratify keep this the same split as the first model.
    train_df, held = train_test_split(
        frame, test_size=0.30, random_state=42, stratify=frame["sentiment"]
    )
    val_df, test_df = train_test_split(
        held, test_size=0.50, random_state=42, stratify=held["sentiment"]
    )
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "validation"
    test_df["split"] = "fpb_test"
    return pd.concat([train_df, val_df, test_df], ignore_index=True)


def _load_affective():
    # SemEval-2007 headlines from the XML, then valence from annotators F1–F3 and M1–M3.
    xml_path = os.path.join(RAW_DIR, "affectivetext_test.xml")
    with open(xml_path, encoding="utf-8") as handle:
        xml = handle.read()
    headlines = {
        int(item_id): html.unescape(re.sub(r"\s+", " ", text).strip())
        for item_id, text in INSTANCE_RE.findall(xml)
    }
    scores = {item_id: [] for item_id in headlines}
    for name in ("F1", "F2", "F3", "M1", "M2", "M3"):
        path = os.path.join(RAW_DIR, f"affective_{name}.txt")
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) < 8:
                    continue
                item_id = int(parts[0])
                if item_id in scores:
                    scores[item_id].append(float(parts[7]))
    rows = []
    for item_id, text in headlines.items():
        values = scores.get(item_id) or []
        if not values:
            continue
        # Mean of the human valence scores, then the official coarse bins in coarse_valence().
        mean_valence = sum(values) / len(values)
        rows.append({
            "headline": text,
            "sentiment": coarse_valence(mean_valence),
            "sarcasm": "unknown",
            "literal_sentiment": "",
            "topic": "general_news",
            "source": "SemEval-2007 Task 14 Affective Text",
            "label_source": "human_valence_mean_of_6_annotators_then_official_coarse_bins",
            "label_level": "headline",
            "valence_mean": round(mean_valence, 4),
            "n_annotators": len(values),
        })
    return pd.DataFrame(rows)


def _load_newsmtsc():
    # Target sentiment toward a named person. Stored for audit, not joined into headline training.
    rows = []
    polarity_map = {-1: "negative", 0: "neutral", 1: "positive"}
    for split_name, filename in (
        ("train", "newsmtsc_rw_train.jsonl"),
        ("validation", "newsmtsc_rw_dev.jsonl"),
        ("test", "newsmtsc_rw_test.jsonl"),
    ):
        path = os.path.join(RAW_DIR, filename)
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                polarity = int(item["polarity"])
                if polarity not in polarity_map:
                    continue
                rows.append({
                    "headline": str(item["sentence"]).strip(),
                    "sentiment": polarity_map[polarity],
                    "sarcasm": "unknown",
                    "literal_sentiment": "",
                    "topic": "politics",
                    "source": "NewsMTSC",
                    "label_source": "human_five_coders_target_dependent",
                    "label_level": "target",
                    "mention": item.get("mention", ""),
                    "published_split": split_name,
                })
    return pd.DataFrame(rows)


def _fpb_rows():
    # Financial sentences plus the original split. label_level "sentence" marks them as sentences.
    frame = pd.read_csv(FPB_PATH)
    split = original_fpb_splits(frame)
    split["sarcasm"] = "unknown"
    split["literal_sentiment"] = ""
    split["topic"] = "business"
    split["source"] = "FinancialPhraseBank"
    split["label_source"] = "human_annotators_at_least_50_percent_agreement"
    split["label_level"] = "sentence"
    split["valence_mean"] = pd.NA
    split["n_annotators"] = pd.NA
    return split


def _assign_group_splits(frame, test_name):
    """Keep headlines that share a 4-token prefix in the same split."""
    work = frame.copy()
    work["group"] = work["headline"].map(group_key)
    groups = {}
    for group_id, chunk in work.groupby("group", sort=False):
        groups[group_id] = chunk.index.tolist()
    # Shuffle groups with seed 42, then give each group to the split that is furthest under its target.
    order = list(groups)
    random.Random(42).shuffle(order)
    total = len(work)
    targets = {"train": 0.70 * total, "validation": 0.15 * total, test_name: 0.15 * total}
    counts = {name: 0 for name in targets}
    split_of = {}
    for group_id in order:
        choice = max(targets, key=lambda name: targets[name] - counts[name])
        counts[choice] += len(groups[group_id])
        for index in groups[group_id]:
            split_of[index] = choice
    work["split"] = work.index.map(split_of)
    return work.drop(columns=["group"])


def _assign_semeval_splits(frame):
    return _assign_group_splits(frame, "semeval_test")


def _load_emobank_newspaper():
    """MASC newspaper sentences from EmoBank. SemEval headlines in that corpus are not loaded again."""
    text_path = os.path.join(RAW_DIR, "emobank.csv")
    meta_path = os.path.join(RAW_DIR, "emobank_meta.tsv")
    if not os.path.exists(text_path) or not os.path.exists(meta_path):
        return pd.DataFrame()
    bank = pd.read_csv(text_path)
    meta = pd.read_csv(meta_path, sep="\t")
    merged = bank.merge(meta, on="id", how="inner")
    news = merged[merged["category"] == "newspaper"].copy()
    news["sentiment"] = news["V"].map(emobank_news_polarity)
    news = news[news["sentiment"] != ""].copy()
    rows = pd.DataFrame({
        "headline": news["text"].map(lambda text: re.sub(r"\s+", " ", str(text)).strip()),
        "sentiment": news["sentiment"].tolist(),
        "sarcasm": "unknown",
        "literal_sentiment": "",
        "topic": "general_news",
        "source": "EmoBank newspaper",
        "label_source": "human_reader_and_writer_valence_mean_bins_2.5_and_3.5",
        "label_level": "sentence",
        "valence_mean": news["V"].round(4).tolist(),
        "n_annotators": pd.NA,
    })
    rows = rows[rows["headline"].str.len() > 0].reset_index(drop=True)
    if rows.empty:
        return rows
    return _assign_group_splits(rows, "news_test")


def _drop_near_test_leaks(frame, threshold=0.8):
    """Drop train/validation rows that are near-copies of a locked test headline."""
    test_mask = frame["split"].isin(LOCKED_TEST_SPLITS)
    test_tokens = [
        (set(norm_tokens(text)), norm_tokens(text)[:1])
        for text in frame.loc[test_mask, "headline"]
    ]
    buckets = {}
    for tokens, first in test_tokens:
        key = first[0] if first else ""
        buckets.setdefault(key, []).append(tokens)

    keep = []
    dropped = 0
    for row in frame.itertuples(index=False):
        # Locked test rows are never dropped. Only train and validation can be near-copies.
        if row.split in LOCKED_TEST_SPLITS:
            keep.append(True)
            continue
        tokens = norm_tokens(row.headline)
        probe = set(tokens)
        first = tokens[0] if tokens else ""
        leaked = any(jaccard(probe, other) >= threshold for other in buckets.get(first, []))
        if leaked:
            dropped += 1
            keep.append(False)
        else:
            keep.append(True)
    kept = frame.loc[keep].reset_index(drop=True)
    return kept, dropped


def _exact_dedupe(frame):
    """Drop conflicting or repeated headlines. Locked test rows stay."""
    work = frame.copy()
    work["key"] = work["headline"].map(norm_key)
    work = work[work["key"].str.len() > 0]
    # A key with two different labels is a conflict. Drop the non-test copies of that key.
    locked = work["split"].isin(LOCKED_TEST_SPLITS)
    conflict_keys = set(
        work.groupby("key")["sentiment"].nunique().loc[lambda series: series > 1].index
    )
    drop_conflict = work["key"].isin(conflict_keys) & ~locked
    work = work[~drop_conflict]
    locked_rows = work[work["split"].isin(LOCKED_TEST_SPLITS)]
    other = work[~work["split"].isin(LOCKED_TEST_SPLITS)]
    other = other.drop_duplicates(subset=["key"], keep="first")
    # A train or validation row whose key matches a locked test headline is removed.
    other = other[~other["key"].isin(set(locked_rows["key"]))]
    work = pd.concat([other, locked_rows], ignore_index=True)
    work = work.drop(columns=["key"])
    return work.reset_index(drop=True), len(conflict_keys)


def build(force=False):
    # Reuse headlines.csv when it is already built, unless force=True asks for a rebuild.
    if os.path.exists(HEADLINES_PATH) and os.path.exists(REPORT_PATH) and not force:
        headlines = pd.read_csv(HEADLINES_PATH)
        targets = pd.read_csv(TARGETS_PATH)
        with open(REPORT_PATH, encoding="utf-8") as handle:
            report = json.load(handle)
        return headlines, targets, report

    # Step 1. Financial sentences with their original split, SemEval headlines with their own split.
    fpb = _fpb_rows()
    semeval = _load_affective()
    semeval = _assign_semeval_splits(semeval)
    newspaper = _load_emobank_newspaper()
    # Step 2. NewsMTSC stays in a separate table. It is not concatenated into headlines.
    targets = _load_newsmtsc()

    # Step 3. Stack headline sources, then drop conflicts and near-copies of the test sets.
    parts = [fpb, semeval]
    if len(newspaper):
        parts.append(newspaper)
    headlines = pd.concat(parts, ignore_index=True)
    before = len(headlines)
    headlines, conflicts = _exact_dedupe(headlines)
    headlines, near_drops = _drop_near_test_leaks(headlines)

    # Step 4. Write the headline table, the audit-only target table, and data_report.json.
    os.makedirs(DATA_DIR, exist_ok=True)
    headlines.to_csv(HEADLINES_PATH, index=False)
    targets.to_csv(TARGETS_PATH, index=False)

    def counts(frame):
        return {label: int((frame["sentiment"] == label).sum()) for label in LABELS}

    report = {
        "headline_rows": int(len(headlines)),
        "headline_rows_before_dedupe": int(before),
        "cross_source_label_conflicts_removed": int(conflicts),
        "near_duplicates_of_test_removed_from_train": int(near_drops),
        "class_counts": counts(headlines),
        "class_counts_by_split": {
            split: counts(chunk) for split, chunk in headlines.groupby("split")
        },
        "split_sizes": {split: int(len(chunk)) for split, chunk in headlines.groupby("split")},
        "sources": {
            "FinancialPhraseBank": {
                "licence": "CC BY-NC-SA 3.0",
                "label_provenance": "Human annotators with a finance background. At least 50% agreement.",
                "label_level": "Investor-view sentiment of a financial-news sentence, not a general headline.",
                "rows": int((headlines["source"] == "FinancialPhraseBank").sum()),
            },
            "SemEval-2007 Task 14": {
                "licence": "Shared-task data redistributed by MichiganNLP/Affective-Text-Individual-Annotations. No separate licence file is in that mirror.",
                "label_provenance": (
                    "Six human annotators gave valence from -100 to 100. "
                    "The mean is mapped with the task's own coarse bins: "
                    "<= -50 negative, >= 50 positive, otherwise neutral."
                ),
                "label_level": "Sentiment of the whole news headline.",
                "rows": int((headlines["source"] == "SemEval-2007 Task 14 Affective Text").sum()),
            },
            "EmoBank newspaper": {
                "licence": "EmoBank v1.4, Buechel and Hahn, EACL 2017. MASC newspaper sentences.",
                "label_provenance": (
                    "Human reader and writer valence on a 1-5 scale, averaged in emobank.csv. "
                    "Negative if valence <= 2.5, positive if valence >= 3.5. "
                    "The band between those cuts is not assigned a class."
                ),
                "label_level": "Newspaper sentence, not a headline and not a finance label.",
                "rows": int((headlines["source"] == "EmoBank newspaper").sum()),
            },
            "NewsMTSC": {
                "licence": "MIT",
                "citation": "Hamborg and Donnay, EACL 2021",
                "label_provenance": "Five human coders. Kept only where the coders agreed.",
                "label_level": "Sentiment toward a named person in a political news sentence. Not a headline label.",
                "rows_not_used_for_headline_training": int(len(targets)),
                "unique_sentences": int(targets["headline"].nunique()),
            },
        },
        "rejected_sources": REJECTED_SOURCES,
        "sarcasm_examples": (
            "No downloadable news-headline set was found that pairs a human sarcasm mark "
            "with a human sentiment label. Sarcasm is recorded as unknown. "
            "The project's sarcasm headlines are not assigned a sentiment, because a sarcasm "
            "mark is not evidence of negative sentiment."
        ),
        "size_limit": (
            "Added MASC newspaper sentences from EmoBank whose human valence is at or below 2.5 "
            "or at or above 3.5. The SemEval headlines inside EmoBank are not loaded a second time. "
            "Larger headline files found online use a pretrained sentiment model or do not document "
            "who labeled them, so they were not added."
        ),
        "columns": [
            "headline", "sentiment", "sarcasm", "literal_sentiment",
            "topic", "source", "label_source", "label_level", "split",
        ],
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return headlines, targets, report


if __name__ == "__main__":
    table, target_table, info = build(force=True)
    print(json.dumps({
        "headline_rows": info["headline_rows"],
        "split_sizes": info["split_sizes"],
        "class_counts": info["class_counts"],
        "class_counts_by_split": info["class_counts_by_split"],
        "near_drops": info["near_duplicates_of_test_removed_from_train"],
        "conflicts": info["cross_source_label_conflicts_removed"],
        "target_rows": int(len(target_table)),
    }, indent=2))
