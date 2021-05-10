# Python STL
from typing import Generator, List
# 3rd Party
from nltk.tokenize import word_tokenize
import torch


def clean(s : str) -> str:
    s = s.lower()
    s = " ".join(word_tokenize(s))
    return s

def get_tokenized_samples(f) -> Generator[List[str],None,None]:
    lines = map(lambda l: l.strip(), f)
    lines = map(lambda l: " ".join(l.split()[1:]), lines)
    lines = map(clean, lines)
    return lines


class ReviewDataset(torch.utils.data.IterableDataset):
    def __init__(self, review_files : List[str], vocab : List[str], repeat=False):
        self.unk             = "<UNK>"
        self.i2w             = dict(enumerate(vocab))
        self.unk_i           = len(self.i2w)
        self.i2w[self.unk_i] = self.unk
        self.w2i             = {w:i for (i,w) in self.i2w.items()}
        self._review_files   = review_files
        self.repeat          = repeat
    def get_index(self, w : str):
        return self.w2i.get(w, self.unk_i)
    def __iter__(self):
        def gen() -> Generator[torch.Tensor, None, None]:
            for path in self._review_files:
                with open(path, "r", encoding="utf-8") as r:
                    reviews = get_tokenized_samples(r)
                    reviews = map(lambda rev: rev.split(), reviews)
                    for rev in reviews:
                        for i in range(2, len(rev)):
                            yield torch.tensor(self.get_index(rev[i-2])), torch.tensor(self.get_index(rev[i-1])), torch.tensor(self.get_index(rev[i]))
        if self.repeat:
            def rep_gen():
                while True:
                    yield from gen()
            return rep_gen()
        else:
            return gen()
