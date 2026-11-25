from shop.orders import order_lines
from shop.report import greeting
from shop.users import get_user_name

assert get_user_name(1) == "ada"
assert get_user_name(9) == "unknown"
assert order_lines() == ["ada: book", "grace: lamp", "linus: pen"]
assert greeting(2) == "Hello, linus!"
print("OK")
