# Bobby Carrot Python implementation

# expose `main` lazily so importing the package doesn't require pygame

def main(*args, **kwargs):
    from .game import main as _main
    return _main(*args, **kwargs)


from .rl_env import BobbyCarrotEnv

__all__ = ["main", "BobbyCarrotEnv"]
