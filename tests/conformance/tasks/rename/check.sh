#!/bin/sh
# No trace of the old name, and the tests pass.
if grep -rn --include='*.py' 'get_usr_name' . >/dev/null; then echo "get_usr_name is still used"; exit 1; fi
python3 test_shop.py >/dev/null 2>&1 || { echo "test_shop.py fails"; exit 1; }
