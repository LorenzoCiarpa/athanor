USERS = {1: "ada", 2: "linus", 3: "grace"}


def get_usr_name(user_id):
    """The name of a user, or "unknown"."""
    return USERS.get(user_id, "unknown")
