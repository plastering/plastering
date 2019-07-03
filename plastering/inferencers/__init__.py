from .inferencer import *
from .zodiac import ZodiacInterface


def load_inferencer(summary):
    cls = globals()[summary['type']]
    kwargs = {k: v for k, v in summary.items() if k not in ['type']}
    obj = cls(**kwargs)
