from functools import wraps

class cleanup(object):
    def __init__(self, *args):
        self.args = args
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            out = func(*args, **kwargs)
            for arg in self.args:
                setattr(args[0], arg, None)
            return out
        return wrapper


def multi_stage_wrapper(func):
    # designed to wrap backward_p1
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.multi_stage:
            out = func(self, *args, **kwargs)
            self.backward_p2_non_recursive()
            if hasattr(self, "dL_dout"):
                self.dL_dout = None
            return out
        else:
            return func(self, *args, **kwargs)
    return wrapper
