#!/bin/sh
# The tests pass and test_calc.py is unchanged.
python3 test_calc.py >/dev/null 2>&1 || { echo "test_calc.py fails"; exit 1; }
grep -q 'assert add(2, 3) == 5' test_calc.py || { echo "test_calc.py was changed"; exit 1; }
