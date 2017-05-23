import sys
import json
import collections

cntr = collections.defaultdict(lambda: 0)
rels = 0
for line in open(sys.argv[1]):
    rel = json.loads(line)
    if rel['Type'] not in ['Implicit', 'EntRel']:
        continue
    cntr[rel['Sense'][0]] += 1
    rels += 1

total = float(sum(cntr.values()))
mclass = None
msize = 0
for klass in sorted(cntr):
    print(klass, cntr[klass], round(cntr[klass]/total,4))
    if cntr[klass] > msize:
        msize = cntr[klass]
        mclass = klass

print("---\nMajority class:", mclass, msize, round(msize/total,4))
print("Number of relations:", rels)
print("Number of classes:", len(cntr))
