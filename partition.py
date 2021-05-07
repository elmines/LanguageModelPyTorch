"""
Usage: python partition.py in.txt out_dir/ N_1 N_2 ...
"""
# Python STL
import sys
if len(sys.argv) < 5:
    sys.stderr.write(__doc__)
    sys.exit(1)
import os
from functools import reduce
import pdb

out_dir = sys.argv[2]
os.makedirs(out_dir, exist_ok=True)


ratios = [int(N) for N in sys.argv[3:]]
indices = reduce(lambda a,b: a + b, [[i]*N for (i,N) in enumerate(ratios)])
handles = [ open(os.path.join(out_dir, f"{i}.txt"), "w", encoding="utf-8") for i in range(len(ratios)) ]

with open(sys.argv[1], "r", encoding="utf-8") as r:
    for (j, line) in enumerate(r):
        handles[ indices[j % len(indices)] ].write(line)
