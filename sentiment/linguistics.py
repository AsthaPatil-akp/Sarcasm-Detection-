"""Extra reading tools for the sentiment view.

News-heading extraction fills the shared headline box. PDF, Word, and webpage
body extraction, POS chunking, named entities, and semantic roles are display
only. Nothing in this file is read by either model's predict() path.
"""
import io
import re
import zipfile
from html import unescape
from urllib.parse import urlparse
from xml.etree import ElementTree

import requests
from bs4 import BeautifulSoup

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_FETCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; educational-news-reader/1.0)",
}
_ANALYSIS_NLP = None
_ANALYSIS_LIMIT = 6000


def _tidy(text):
    text = unescape(str(text or ""))
    text = text.replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def _fetch(url):
    cleaned = str(url or "").strip()
    parsed = urlparse(cleaned)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Enter a full http or https URL.")
    try:
        response = requests.get(cleaned, headers=_FETCH_HEADERS, timeout=12)
    except requests.Timeout as exc:
        raise ValueError("The site took too long to answer.") from exc
    except requests.RequestException as exc:
        raise ValueError(f"Could not open that URL. {exc}") from exc
    if response.status_code >= 400:
        raise ValueError(f"The site returned HTTP {response.status_code}.")
    response.encoding = response.encoding or "utf-8"
    return response.text


def _strip_site_suffix(title):
    """Drop a short publisher suffix from a <title>, such as ' | Reuters'."""
    for separator in (" | ", " – ", " — ", " - "):
        if separator not in title:
            continue
        left, right = title.rsplit(separator, 1)
        if len(left.split()) >= 3 and len(right.split()) <= 6 and len(right) <= 40:
            return left.strip()
    return title


def heading_from_html(html_text):
    """Pick the article heading. Open Graph title wins, then Twitter, then <title>, then h1."""
    soup = BeautifulSoup(html_text or "", "html.parser")
    for attrs in (
        {"property": "og:title"},
        {"name": "twitter:title"},
        {"property": "twitter:title"},
    ):
        tag = soup.find("meta", attrs=attrs)
        content = _tidy(tag.get("content")) if tag is not None else ""
        if content:
            return content
    if soup.title and soup.title.string:
        title = _strip_site_suffix(_tidy(soup.title.string))
        if title:
            return title
    heading = soup.find("h1")
    if heading is not None:
        text = _tidy(heading.get_text(" ", strip=True))
        if text:
            return text
    raise ValueError("No news heading was found on that page.")


def extract_news_heading(url):
    return heading_from_html(_fetch(url))


def body_from_html(html_text):
    """Article paragraphs. Scripts, navigation, and the heading tags are left out."""
    soup = BeautifulSoup(html_text or "", "html.parser")
    for tag in soup.find_all(["script", "style", "noscript", "nav", "footer", "header", "form", "aside"]):
        tag.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup
    paragraphs = []
    for paragraph in root.find_all("p"):
        text = _tidy(paragraph.get_text(" ", strip=True))
        if len(text) >= 40:
            paragraphs.append(text)
    if not paragraphs and root is not None:
        text = _tidy(root.get_text(" ", strip=True))
        if text:
            paragraphs.append(text)
    if not paragraphs:
        raise ValueError("No article text was found on that page.")
    return "\n\n".join(paragraphs)


def extract_webpage_text(url):
    return body_from_html(_fetch(url))


def extract_docx(data):
    """Read paragraph text from a .docx file. The file is a zip of Word XML."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError("That file is not a Word .docx document.") from exc
    try:
        xml_bytes = archive.read("word/document.xml")
    except KeyError as exc:
        raise ValueError("That Word file has no document text.") from exc
    root = ElementTree.fromstring(xml_bytes)
    paragraphs = []
    for paragraph in root.iter(_W + "p"):
        parts = []
        for node in paragraph.iter(_W + "t"):
            if node.text:
                parts.append(node.text)
            if node.tail:
                parts.append(node.tail)
        line = _tidy("".join(parts))
        if line:
            paragraphs.append(line)
    if not paragraphs:
        raise ValueError("That Word file has no extractable text.")
    return "\n".join(paragraphs)


def extract_pdf(data):
    """Read the text layer of a PDF. Scanned pages with no text layer cannot be read."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ValueError("PDF reading is unavailable because pypdf is not installed.") from exc
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:
        raise ValueError("That file could not be read as a PDF.") from exc
    pages = []
    for page in reader.pages:
        raw = page.extract_text() or ""
        text = re.sub(r"[ \t]+\n", "\n", raw)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if text:
            pages.append(text)
    if not pages:
        raise ValueError(
            "This PDF has no extractable text. A scan without a text layer cannot be read here."
        )
    return "\n\n".join(pages)


def _analysis_nlp():
    # Separate from sentiment/text.py, which loads the same model with the parser and NER off.
    global _ANALYSIS_NLP
    if _ANALYSIS_NLP is None:
        import spacy
        _ANALYSIS_NLP = spacy.load("en_core_web_sm")
    return _ANALYSIS_NLP


def _pos_chunks(doc):
    """Chunk by part-of-speech pattern, not by the dependency parser.

    Noun phrase: optional determiner or pronoun, then adjectives or numbers, ending in nouns.
    Verb phrase: auxiliaries and adverbs around one or more verbs.
    """
    tokens = [token for token in doc if not token.is_space and token.pos_ != "PUNCT"]
    chunks = []
    index = 0
    while index < len(tokens):
        start = index
        if tokens[index].pos_ in {"DET", "PRON"}:
            index += 1
        while index < len(tokens) and tokens[index].pos_ in {"ADJ", "NUM"}:
            index += 1
        noun_end = index
        while noun_end < len(tokens) and tokens[noun_end].pos_ in {"NOUN", "PROPN"}:
            noun_end += 1
        if noun_end > index:
            span = tokens[start:noun_end]
            chunks.append({
                "chunk": " ".join(token.text for token in span),
                "kind": "noun phrase",
                "pattern": " ".join(token.pos_ for token in span),
            })
            index = noun_end
            continue
        index = start
        if tokens[index].pos_ in {"AUX", "ADV", "PART", "VERB"}:
            saw_verb = False
            while index < len(tokens) and tokens[index].pos_ in {"AUX", "ADV", "PART", "VERB"}:
                saw_verb = saw_verb or tokens[index].pos_ == "VERB"
                index += 1
            if saw_verb and index > start:
                span = tokens[start:index]
                chunks.append({
                    "chunk": " ".join(token.text for token in span),
                    "kind": "verb phrase",
                    "pattern": " ".join(token.pos_ for token in span),
                })
                continue
        index = start + 1
    return chunks


def _semantic_rows(doc):
    """Predicate, subject, object, negation, and adjectival modifiers from the dependency parse."""
    rows = []
    for sentence in doc.sents:
        root = sentence.root
        subjects = [
            token.text for token in root.children
            if token.dep_ in {"nsubj", "nsubjpass", "csubj"}
        ]
        objects = [
            token.text for token in root.children
            if token.dep_ in {"dobj", "obj", "attr", "acomp", "oprd"}
        ]
        negated = any(token.dep_ == "neg" for token in sentence)
        modifiers = [
            f"{token.text} modifies {token.head.text}"
            for token in sentence
            if token.dep_ == "amod"
        ]
        rows.append({
            "sentence": sentence.text.strip(),
            "predicate": root.lemma_,
            "subject": ", ".join(subjects),
            "object": ", ".join(objects),
            "negated": "yes" if negated else "no",
            "modifiers": "; ".join(modifiers),
        })
    return rows


def linguistic_analysis(text):
    """POS chunks, named entities, and semantic roles. Not a model input."""
    raw = str(text or "").strip()
    if not raw:
        return {"chunks": [], "entities": [], "semantics": [], "truncated": False}
    truncated = len(raw) > _ANALYSIS_LIMIT
    doc = _analysis_nlp()(raw[:_ANALYSIS_LIMIT])
    import spacy
    entities = [
        {
            "text": entity.text,
            "label": entity.label_,
            "meaning": spacy.explain(entity.label_) or entity.label_,
        }
        for entity in doc.ents
    ]
    return {
        "chunks": _pos_chunks(doc),
        "entities": entities,
        "semantics": _semantic_rows(doc),
        "truncated": truncated,
    }
