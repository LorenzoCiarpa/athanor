from shop import users


def greeting(user_id):
    return "Hello, " + users.get_usr_name(user_id) + "!"
