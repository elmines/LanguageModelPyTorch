"""
Usage: train.ft.txt out.txt N 
"""

# Python STL
import sys
if len(sys.argv) != 4:
    sys.stderr.write(__doc__)
    sys.exit(1)
from collections import Counter
import pdb
# Local
from lm import get_tokenized_samples

N = int(sys.argv[3])
counts = Counter()
with open(sys.argv[1], "r", encoding="utf-8") as r:
    samples = get_tokenized_samples(r)
    for (i,s) in enumerate(samples, start=1):
        counts.update(s.split())
        if i % 10000 == 0:
            print(f"Parsed {i} samples")

with open(sys.argv[2], "w", encoding="utf-8") as w:
    words = map(lambda p: p[0], counts.most_common(N))
    w.write("\n".join(words))
