#!/bin/sh
# answer.txt names app/net/client.py and the value 7.
test -f answer.txt || { echo "no answer.txt"; exit 1; }
tr -d ' \r' < answer.txt | grep -Eq '^(\./)?app/net/client\.py:7$' || { echo "answer.txt: $(head -c 200 answer.txt)"; exit 1; }
