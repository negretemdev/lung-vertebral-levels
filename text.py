import random
from totalsegmentator.map_to_binary import class_map

random.seed(0)
lines = ['0 0 0 0 0 0 0 "Clear Label"']
for i, n in sorted(class_map["total"].items()):
    r, g, b = (random.randint(40, 255) for _ in range(3))
    lines.append(f'{i} {r} {g} {b} 1 1 1 "{n}"')

with open("labels_itksnap.txt", "w") as f:
    f.write("\n".join(lines))
