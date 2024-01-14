import time
import contextlib

@contextlib.contextmanager
def benchmark(name=None):
    if name is None:
        name = "Elapsed Time"
    start_time = time.time()
    yield
    end_time = time.time()
    print(f"{name}: {end_time - start_time :.2f}s")
    