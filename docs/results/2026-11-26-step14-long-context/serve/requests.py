"""The two requests of serve/: the haystack of haystack.py (5500 notes) as
the system message, then a question. sys1.json asks for the passphrase,
sys2.json about note 7 (the brown letter in the chapel yard, moved on day 81)."""
import json, subprocess
text = subprocess.run(["python3", "haystack.py", "5500", "text"], capture_output=True, text=True).stdout
doc, q = text.split("  Question: ")
q1 = "Question: " + q.strip()
q2 = "Question: in note 7, what was done to the item, and on which day? Answer in one short sentence."
for name, qq in (("sys1.json", q1), ("sys2.json", q2)):
    json.dump({"max_tokens": 32, "temperature": 0, "messages": [{"role": "system", "content": doc.strip()},
               {"role": "user", "content": qq}]}, open(name, "w"))
