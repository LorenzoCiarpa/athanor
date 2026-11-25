#!/bin/sh
# fizzbuzz(n) is right for n up to 100, and the script prints fizzbuzz(15).
python3 - <<'PY' || exit 1
import importlib.util, subprocess, sys
spec = importlib.util.spec_from_file_location("fizzbuzz", "fizzbuzz.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
want = lambda i: "FizzBuzz" if i % 15 == 0 else "Fizz" if i % 3 == 0 else "Buzz" if i % 5 == 0 else str(i)
for n in (1, 15, 100):
    if list(m.fizzbuzz(n)) != [want(i) for i in range(1, n + 1)]:
        sys.exit(f"fizzbuzz({n}) is wrong")
out = subprocess.run([sys.executable, "fizzbuzz.py"], capture_output=True, text=True).stdout.split()
if out != [want(i) for i in range(1, 16)]:
    sys.exit("the script does not print fizzbuzz(15)")
PY
