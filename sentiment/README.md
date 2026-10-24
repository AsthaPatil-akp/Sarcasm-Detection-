# Sentiment module

This folder is separate from the sarcasm detector. Training reads `embeddings/glove.6B.100d.txt` and does not write it. It does not write `models/`, `results/`, `data/sarcasm_base.csv`, or `app/sarcasm_inference.py`.

The app still shows this model's label beside the sarcasm verdict. The sarcasm verdict comes from the original sarcasm model. The detailed sarcasm pipeline is unchanged.

There is no git repository in this project, so no branch was created.

## Why this is not a PyTorch model

The requested network is a BiLSTM with trainable embeddings, dropout, and a softmax classifier. That network is trained here with TensorFlow and Keras, which were already installed. PyTorch is not installed. Installing it in this same Python environment can replace NumPy and break the sarcasm detector, and the disk had under 2 GB free. The training loop, loss, and early stopping are in `train.py`.

## Dataset

Human labels only. Rows that a pretrained sentiment model had labeled, and rows whose labeling method was not documented, were not used.

| Source | What the label means | Rows kept | Licence |
| --- | --- | ---: | --- |
| FinancialPhraseBank, Sentences_50Agree | Human investor-view sentiment of a financial-news sentence. At least 50% of annotators agreed. | 4,836 before the split filters | CC BY-NC-SA 3.0 |
| SemEval-2007 Task 14 Affective Text | Human valence of a general news headline, from six annotators. The mean valence uses the task's own coarse bins: at or below -50 negative, at or above 50 positive, otherwise neutral. | about 976 after duplicate checks | Shared-task data via the MichiganNLP annotation mirror. That mirror has no separate licence file. |
| NewsMTSC (Hamborg and Donnay, EACL 2021) | Human sentiment toward a named person in a political news sentence. Five coders. | 9,885 target rows,  stored in `data/target_sentiment.csv` | MIT |

NewsMTSC is not headline sentiment, so it is not used to train the headline classifier.

Rejected:

- `dnakhla/us-news-headlines-enriched` (143,142 headlines). Labels come from `cardiffnlp/twitter-roberta-base-sentiment-latest`.
- `MASHXD/investing_financial_news_headlines` (13,259). The card does not say who assigned the labels.
- `Okcan/news-headlines-classifier` (350). Too small, and the labeler is not identified.

No downloadable news-headline set was found that has both a human sarcasm mark and a human sentiment label. The project's sarcasm headlines were not given a sentiment, because sarcasm is not evidence of a negative label. The `sarcasm` and `literal_sentiment` fields are therefore unknown or blank.

Usable headline and sentence rows after duplicate removal: **5,812**. That is short of 20,000–30,000. The missing rows would have been automatic or undocumented labels.

Class counts on those 5,812 rows: negative 841, neutral 3,487, positive 1,484. Balance comes from class weights, not from copying minority rows.

| split | rows |
| --- | ---: |
| train | 4,064 |
| validation | 872 |
| FinancialPhraseBank test (the original test headlines) | 726 |
| SemEval headline test | 150 |

Exact duplicate headlines with disagreeing labels were removed (4 keys). A train or validation headline with token-overlap of at least 0.8 against a test headline was removed (7 rows). The vocabulary is built from the training headlines only.

Audit columns in `data/headlines.csv`: `headline`, `sentiment`, `sarcasm`, `literal_sentiment`, `topic`, `source`, `label_source`, plus `label_level` and `split`.

## Preprocessing

Implemented in `text.py`. The sarcasm cleaner is not called.

1. URLs and HTML tags are removed. Case, `!`, and `?` stay until tagging.
2. spaCy tokenization, part of speech, and lemmatization (`en_core_web_sm`).
3. Determiners, pronouns, and other function words are dropped.
4. Negation words (`not`, `no`, `never`, `n't`, and the rest of the list) are kept. The next content word is stored as `not_<lemma>`, so `not good` does not become `good`.
5. Contrast words such as `but` are kept.
6. A fully capitalized word adds an `allcaps` token.
7. Training-only vocabulary. A token must appear in at least 2 training headlines. Padding id 0. Unknown id 1. Sequences are padded to 26, the 95th percentile of training lengths, capped at 60.

`results/eda.json` has the training class counts, headline lengths, and frequent lemmas per class.

## Architecture and training

- Embedding, 100 dimensions, initialised from GloVe (3,576 of 3,884 rows matched) and then trained. Spatial dropout 0.3.
- Bidirectional LSTM, 64 units each way. Output activation tanh. Gates use sigmoid.
- Dropout 0.4.
- Dense 64, ReLU, dropout 0.3.
- Dense 3, softmax. Order: negative, neutral, positive.
- Loss: sparse categorical cross-entropy, with class weights negative 2.269, neutral 0.560, positive 1.293.
- Optimiser: Adam, learning rate 0.001.
- Batch size 32. Seed 42.
- Up to 20 epochs. Early stopping watches validation loss, patience 3, and restores the best weights. One epoch without improvement halves the learning rate, down to 1e-5.

The run used 7 epochs. Epoch 4 had the lowest validation loss (0.6849) and validation macro-F1 0.6708. Epochs 5–7 did not beat that loss, so training stopped and epoch 4 was restored. Epoch 7's validation macro-F1 was 0.6783, slightly higher, but selection uses validation loss, so those weights were not kept.

| epoch | accuracy | loss | val accuracy | val loss | val macro-F1 | learning rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.4555 | 1.0232 | 0.5539 | 0.9109 | 0.4992 | 0.001 |
| 2 | 0.5453 | 0.8883 | 0.6388 | 0.8175 | 0.5943 | 0.001 |
| 3 | 0.6243 | 0.7768 | 0.6606 | 0.7755 | 0.6197 | 0.001 |
| 4 | 0.6772 | 0.6850 | 0.7190 | 0.6849 | 0.6708 | 0.001 |
| 5 | 0.7338 | 0.5841 | 0.7099 | 0.6896 | 0.6683 | 0.0005 |
| 6 | 0.7645 | 0.4966 | 0.7099 | 0.6933 | 0.6777 | 0.00025 |
| 7 | 0.7810 | 0.4711 | 0.7064 | 0.6928 | 0.6783 | 0.000125 |

The loss plot is `results/loss_curves.png`.

## Baseline versus improved

The baseline is the previous financial-only BiLSTM, scored before its weights were replaced. Both models see the same test headlines. The baseline weights are copied to `models/baseline_financial.keras`.

FinancialPhraseBank test (726 sentences). The new model is slightly worse here.

| | Accuracy | Macro-F1 | Negative F1 | Neutral F1 | Positive F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.763 | 0.732 | 0.703 | 0.824 | 0.668 |
| Improved | 0.755 | 0.723 | 0.678 | 0.818 | 0.673 |

SemEval general-news headlines (150 headlines). The new model is better here.

| | Accuracy | Macro-F1 | Negative F1 | Neutral F1 | Positive F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.527 | 0.423 | 0.384 | 0.649 | 0.238 |
| Improved | 0.620 | 0.532 | 0.581 | 0.693 | 0.323 |

Positive headlines in that test number only 19, and the positive F1 is still 0.323. The gain is mostly on negative and neutral headlines.

## Sarcasm probability as an extra feature

On 160 validation headlines the unchanged sarcasm detector was called and its probability was added as a feature to a logistic regression, fit on half and scored on the other half.

- Sentiment probabilities alone: macro-F1 0.737
- With the sarcasm probability: macro-F1 0.643

The extra feature made the held-out half worse, so it is not used. `predict(text)` and `predict(text, sarcasm_probability=...)` return the same sentiment when no calibrator file is present. Only 13 of the 80 evaluation rows had sarcasm probability at or above 0.5, which is too few for a sarcasm-specific claim. There is no human sarcasm-plus-sentiment test set in this project.

## Errors that remain

- "Sales fell abroad but increased in Finland." Gold neutral, predicted negative. The contrast is not solved.
- "'Stomp' steps to No. 1 at box office." Gold positive, predicted neutral.
- "It projected revenue of $2.2 billion to $2.3 billion, slightly higher than analyst estimates." Gold positive, predicted negative.

## How to retrain

From the project root:

```
py -3.12 sentiment/train.py
```

`train.py` rebuilds nothing that already exists under `sentiment/data/` unless you run:

```
py -3.12 sentiment/data_lib.py
```

Tests:

```
py -3.12 -m unittest sentiment.tests.test_sentiment -v
```

Restart Streamlit after training so it loads the new weights.

## Limitations

- 5,812 human-labeled rows, mostly financial sentences, is a small set. General-news coverage is the 1,000 SemEval headlines, and the headline test has only 150 of them.
- SemEval's neutral band is wide, so a headline can read as bad news and still be neutral under the official bins.
- NewsMTSC cannot be treated as a headline label. Mixing it in would teach person-level sentiment.
- The financial test got slightly worse. The general-headline test got better. Those are the measured results.
- Irony such as praise next to a bankruptcy is still not a solved case, and it was not given a gold label here.
