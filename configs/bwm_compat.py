"""Canonical BWM switch with a live alias for unchanged legacy tools."""

from easydict import EasyDict


class BWMConfig(EasyDict):
    """Store only use_bwm; old use_hbwr attribute/key writes update it too."""

    def __init__(self, d=None, **kwargs):
        # EasyDict's default constructor copies subclass methods into entries.
        # Populate normally without that class-attribute copying step.
        values = dict(d or {})
        values.update(kwargs)
        for name, value in values.items():
            setattr(self, name, value)

    def __getattr__(self, name):
        if name == "use_hbwr":
            return object.__getattribute__(self, "use_bwm")
        raise AttributeError(name)

    def __setattr__(self, name, value):
        super().__setattr__("use_bwm" if name == "use_hbwr" else name, value)

    __setitem__ = __setattr__

    def __getitem__(self, name):
        return super().__getitem__("use_bwm" if name == "use_hbwr" else name)

    def __contains__(self, name):
        return super().__contains__("use_bwm" if name == "use_hbwr" else name)

    def get(self, name, default=None):
        return super().get("use_bwm" if name == "use_hbwr" else name, default)

    def pop(self, name, *args):
        return super().pop("use_bwm" if name == "use_hbwr" else name, *args)

    def __delattr__(self, name):
        super().__delattr__("use_bwm" if name == "use_hbwr" else name)
