"""The prompt of tests/test_long.c as text: n notes, the passphrase after n/10.
usage: haystack.py <n> text|json"""
import json, sys
C = ["red", "green", "blue", "grey", "yellow", "white", "black", "brown"]
T = ["crate", "barrel", "letter", "lamp", "rope", "ledger", "kettle", "anchor"]
P = ["north hall", "east dock", "old mill", "harbor office", "south gate", "upper loft", "river shed", "chapel yard"]
V = ["counted", "moved", "painted", "checked", "repaired", "weighed", "sealed", "cleaned"]
n = int(sys.argv[1])
out = ["Below is a long log of notes. Read it, then answer the question at the end.\n\n"]
for i in range(n):
    if i == n // 10:
        out.append("Note about the passphrase: the secret passphrase is amber-falcon-4127. Remember it, you will be asked for it. ")
    x = (i * 2654435761) & 0xffffffff
    out.append(f"Note {i}: the {C[x % 8]} {T[(x >> 3) % 8]} in the {P[(x >> 6) % 8]} was {V[(x >> 9) % 8]} on day {1 + (x >> 12) % 365}. ")
out.append("\n\nQuestion: what is the secret passphrase given in the note about the passphrase? Answer with the passphrase only.")
text = "".join(out)
if sys.argv[2] == "text":
    sys.stdout.write(text.replace("\n", " ") + "\n")
else:
    sys.stdout.write(json.dumps({"max_tokens": 24, "temperature": 0, "messages": [{"role": "user", "content": text}]}))
