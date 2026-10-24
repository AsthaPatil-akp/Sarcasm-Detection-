# Sarcasm detection

A Streamlit app that scores one news headline with two separate models.

The sarcasm model says whether the headline is sarcastic. The sentiment model says whether the wording is negative, neutral, or positive. Both models read the same headline. The sentiment model does not receive the sarcasm probability, and the linguistic tables in the sentiment section are not model inputs.

## Run the app

From the project root, with Python 3.12:

```
py -3.12 -m streamlit run app/app.py
```

Then open http://localhost:8501.

Type a headline, paste one, or extract a news heading from a URL. Edit the text if you want, then click **Analyse**. Both models score that text.

The sarcasm cutoff defaults to **0.50**. The sidebar toggle **Use tuned threshold (0.36)** uses the cutoff that had the best F1 on the sarcasm validation set. The probability does not change when the cutoff changes. Only the sarcastic / not-sarcastic label does.

**Download the pipeline** writes a PDF or HTML report for the sarcasm path or the sentiment path. The charts in those files use the same numbers as the on-screen tables.

## What is in the app

- Sarcasm verdict, probability, attention highlight, and a step-by-step sarcasm pipeline.
- A separate sentiment label with class probabilities and its own pipeline.
- News-heading extraction from a URL into the shared headline box.
- In the sentiment section only: PDF and Word text extraction, webpage body extraction, part-of-speech chunks, named entities, and semantic roles. Those tables describe the wording. They are not passed into either model.

## Sarcasm model

Trained on the News Headlines Dataset for Sarcasm Detection (The Onion versus HuffPost), 28,503 headlines.

| Split | Headlines |
| --- | ---: |
| Train | 20,521 |
| Validation | 2,281 |
| Test, never used for fitting | 5,701 |

Each headline becomes two inputs.

**Word sequence.** Lowercasing, URL and HTML removal, contraction expansion, and light punctuation cleaning. Words become ids from a 23,871-word vocabulary, padded or truncated to 16. Unknown words use id 1. Id 0 is padding and is masked. Each id is a 100-dimensional GloVe vector. A bidirectional LSTM (64 units each way) reads the sequence. Bahdanau attention pools it into one 128-number vector.

**NLP features.** 74 scaled numbers: VADER lexicon scores, part-of-speech ratios, intensifier and style counts, and a 50-dimensional SVD compression of TF-IDF unigrams and bigrams. spaCy supplies tags and lemmas. Named-entity recognition and dependency parsing are turned off on this spaCy load. A dense layer of 32 ReLU units compresses the 74 numbers.

The two vectors are joined (160 numbers), passed through a dense layer of 64 ReLU units, and a sigmoid outputs the sarcasm probability. Dropout is on during training and off when the app scores a headline.

Training had two stages. Stage 1 froze GloVe and trained with Adam at learning rate 0.001. Validation loss was best at epoch 7 (0.3022). Stage 2 unfroze GloVe at learning rate 0.0001. Epoch 1 of that stage reached validation loss 0.2952, and those weights are the saved model `models/hybrid_final.keras`.

On the 5,701 test headlines, ROC AUC is 0.9428 at either cutoff.

| Cutoff | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| 0.50 | 0.8677 | 0.8713 | 0.8469 | 0.8590 |
| 0.36 | 0.8684 | 0.8420 | 0.8904 | 0.8655 |

0.36 is the best point on a validation search from 0.30 to 0.70 in steps of 0.02. Validation F1 was 0.8726 at 0.50 and 0.8791 at 0.36. The test set was not used to choose it.

A logistic-regression baseline on the same test set scored accuracy 0.7899 and AUC 0.8779.

## Sentiment model

A second network, in `sentiment/`. It does not load the sarcasm weights and it does not use VADER, TF-IDF, or attention.

Labels are human labels only. Rows labeled by another sentiment model were not used.

| Source | What the label is | Rows in the training table |
| --- | --- | ---: |
| Financial PhraseBank, Sentences_50Agree | Investor-view sentiment of a financial sentence. At least 50% of annotators agreed. Licence CC BY-NC-SA 3.0. | 4,827 |
| SemEval-2007 Task 14 Affective Text | Mean valence of a news headline from six annotators. Negative at or below −50, positive at or above 50, otherwise neutral. | 985 |
| EmoBank newspaper sentences | Reader and writer valence on a 1–5 scale. Negative at or below 2.5, positive at or above 3.5. The band between those cuts is not given a class. | 155 |

NewsMTSC is stored in `sentiment/data/target_sentiment.csv` for audit. It is sentiment toward a named person, so it is not used as a headline label.

After duplicate checks the table has 5,967 rows: 4,173 train, 895 validation, and three locked tests (726 Financial PhraseBank, 150 SemEval, 23 newspaper). The tests are not used for fitting or early stopping.

Preprocessing keeps content lemmas, marks negation as `not_<lemma>`, and keeps contrast words. The vocabulary is built from training text only (3,981 tokens, minimum count 2). Sequences are padded to 26.

The network is an embedding of 100 dimensions, initialised from GloVe (3,666 of 3,981 rows matched) and then trained, a bidirectional LSTM of 64 units each way that returns its final state only, dropout, a dense layer of 64 ReLU units, and a 3-way softmax. Loss is sparse categorical cross-entropy with class weights. Adam starts at learning rate 0.001. Early stopping restored epoch 6, where validation loss was 0.7167.

| Locked test | Rows | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| Financial PhraseBank | 726 | 0.738 | 0.728 |
| SemEval headlines | 150 | 0.580 | 0.506 |
| EmoBank newspaper | 23 | 0.652 | 0.481 |

The newspaper test is small and has no neutral sentences, so that macro-F1 is a weak estimate. The original financial-only checkpoint is still in `sentiment/models/baseline_financial.keras`. On the Financial PhraseBank test that older model scored accuracy 0.763 and macro-F1 0.732.

Adding the sarcasm probability as an extra sentiment feature made a validation check worse, so it is not used.

## Project layout

```
app/                  Streamlit app and sarcasm inference
models/               Sarcasm checkpoint, vocabulary, and NLP pipeline
data/                 Sarcasm headlines
embeddings/           GloVe file (not in git) and the saved embedding matrix
features/             Saved training matrices
notebooks/            Sarcasm training notebook
results/              Sarcasm test metrics and figures
sentiment/            Sentiment data, training, and inference
```

## GloVe file

`embeddings/glove.6B.100d.txt` is not in this repository. It is about 331 MB, and GitHub rejects files over 100 MB. The saved Keras models already contain their embedding weights, so the app scores headlines without that file. Download `glove.6B.100d.txt` into `embeddings/` before retraining either model.

`models/hybrid_final.keras` is 76 MB. GitHub accepts it and warns that it is above the 50 MB size GitHub recommends.

## Retrain sentiment

From the project root, after the GloVe file is in place:

```
py -3.12 sentiment/train.py
```

Restart Streamlit afterwards so it loads the new weights. `sentiment/train.py` does not modify the sarcasm checkpoint.

## Python packages

Python 3.12. The app uses TensorFlow and Keras, Streamlit, spaCy (`en_core_web_sm`), NLTK stop words, VADER, `contractions`, scikit-learn, pandas, NumPy, requests, Beautiful Soup, and pypdf. PyTorch is not required. Installing it into this same environment can change NumPy and break the sarcasm model.
