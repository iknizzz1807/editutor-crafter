import itertools
from wordfreq import zipf_frequency

letters = "ciolnabhnl"
remaining_pool = list(letters)
if "l" in remaining_pool:
    remaining_pool.remove("l")

print(f"Pool: {remaining_pool}")

found = []

# Try lengths from 3 to 10
for length in range(2, 10):
    print(f"Checking length {length + 1}...")
    for perm in set(itertools.permutations(remaining_pool, length)):
        word = "l" + "".join(perm)
        freq = zipf_frequency(word, "en")
        if freq > 3:
            found.append((word, freq))

# Also try length 10
print("Checking length 10...")
for perm in set(itertools.permutations(remaining_pool, 9)):
    word = "l" + "".join(perm)
    freq = zipf_frequency(word, "en")
    if freq > 0:
        found.append((word, freq))

found.sort(key=lambda x: x[1], reverse=True)
print("Found words:")
for w, f in found[:20]:
    print(f"{w}: {f}")
