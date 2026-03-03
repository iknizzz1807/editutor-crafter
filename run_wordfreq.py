import itertools
from wordfreq import zipf_frequency

letters = "ciolnabhnl"

# fix chữ đầu là L.
# letters contains 'l' (lowercase). User used replace("L", ...).
# I will use lowercase 'l' to ensure we are using the letters from the pool.
remaining = list(letters)
if "l" in remaining:
    remaining.remove("l")
elif "L" in remaining:
    remaining.remove("L")

found = set()

# remaining is now 9 letters. 9! = 362,880.
print(f"Checking {362880} permutations...")

for perm in itertools.permutations(remaining):
    word = "l" + "".join(perm)
    # check if it exists in wordfreq at all
    if zipf_frequency(word, "en") > 0:
        found.add(word)

print("Found words:")
for w in sorted(found):
    print(w)
