import os


def get_var(name, args):
    v = os.environ.get(name.upper(), None) or getattr(args, name)
    return v