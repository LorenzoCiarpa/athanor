from shop.users import get_usr_name

ORDERS = [(1, "book"), (3, "lamp"), (2, "pen")]


def order_lines():
    return [f"{get_usr_name(uid)}: {item}" for uid, item in ORDERS]
