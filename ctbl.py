import random
from totalsegmentator.map_to_binary import class_map

random.seed(0)
with open("totalseg.ctbl", "w") as f:
    f.write("0 background 0 0 0 0\n")
    for i, n in sorted(class_map["total"].items()):
        r, g, b = (random.randint(40, 255) for _ in range(3))
        f.write(f"{i} {n} {r} {g} {b} 255\n")
