# Python STL
from typing import Generator, List
# 3rd Party
from nltk.tokenize import word_tokenize


def clean(s : str) -> str:
    s = s.lower()
    s = " ".join(word_tokenize(s))
    return s

def get_tokenized_samples(f) -> Generator[List[str],None,None]:
    lines = map(lambda l: l.strip(), f)
    lines = map(lambda l: " ".join(l.split()[1:]), lines)
    lines = map(clean, lines)
    return lines
