from .mujoco import MJArticulationCfg


def __getattr__(name: str):
    if name == "SimpleEnv":
        from .locomotion import SimpleEnv

        return SimpleEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["MJArticulationCfg", "SimpleEnv"]
