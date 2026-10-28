"""
Dependency-free TF-IDF matching scikit-learn's TfidfVectorizer defaults
(lowercase, token pattern \b\w\w+\b, smooth idf, l2 norm), so quality metrics
stay identical without loading scikit-learn (~50 MB) on small hosts.
ENGLISH_STOP_WORDS is scikit-learn's list, copied verbatim.
"""

import math
import re
from collections import Counter
from typing import List, Optional, Sequence, Tuple

_TOKEN = re.compile(r"(?u)\b\w\w+\b")

ENGLISH_STOP_WORDS = frozenset([
    "a", "about", "above", "across", "after", "afterwards", "again", "against", "all", "almost",
    "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst",
    "amount", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere",
    "are", "around", "as", "at", "back", "be", "became", "because", "become", "becomes", "becoming",
    "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between",
    "beyond", "bill", "both", "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
    "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during",
    "each", "eg", "eight", "either", "eleven", "else", "elsewhere", "empty", "enough", "etc",
    "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen",
    "fifty", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found",
    "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have",
    "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself",
    "him", "himself", "his", "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least",
    "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more",
    "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely",
    "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor",
    "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto",
    "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part",
    "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming",
    "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six",
    "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere",
    "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves",
    "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon",
    "these", "they", "thick", "thin", "third", "this", "those", "though", "three", "through",
    "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve",
    "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well",
    "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas",
    "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who",
    "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet",
    "you", "your", "yours", "yourself", "yourselves",
])


def _analyze(text: str, stop_words: Optional[frozenset], ngram_range: Tuple[int, int]) -> List[str]:
    tokens = _TOKEN.findall(text.lower())
    if stop_words:
        tokens = [t for t in tokens if t not in stop_words]
    lo, hi = ngram_range
    if (lo, hi) == (1, 1):
        return tokens
    grams = []
    for n in range(lo, hi + 1):
        grams.extend(" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    return grams


def tfidf_vectors(
    corpus: Sequence[str],
    stop_words: Optional[str] = None,
    ngram_range: Tuple[int, int] = (1, 1),
    max_features: Optional[int] = None,
) -> List[dict]:
    """Sparse l2-normalised TF-IDF vectors ({term: weight}) for each document."""
    sw = ENGLISH_STOP_WORDS if stop_words == "english" else None
    counts = [Counter(_analyze(doc, sw, ngram_range)) for doc in corpus]
    df: Counter = Counter()
    for c in counts:
        df.update(c.keys())
    if not df:
        raise ValueError("empty vocabulary")
    vocab = set(df)
    if max_features is not None and len(vocab) > max_features:
        total: Counter = Counter()
        for c in counts:
            total.update(c)
        vocab = {t for t, _ in sorted(total.items(), key=lambda kv: (-kv[1], kv[0]))[:max_features]}
    n = len(corpus)
    idf = {t: math.log((1 + n) / (1 + df[t])) + 1.0 for t in vocab}
    vectors = []
    for c in counts:
        vec = {t: tf * idf[t] for t, tf in c.items() if t in vocab}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        vectors.append({t: v / norm for t, v in vec.items()} if norm else {})
    return vectors


def cosine(a: dict, b: dict) -> float:
    if len(a) > len(b):
        a, b = b, a
    return float(sum(v * b.get(t, 0.0) for t, v in a.items()))
