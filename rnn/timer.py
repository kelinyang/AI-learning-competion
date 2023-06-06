import time
import numpy as np
class Timer:
    """Record multipl running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()
    
    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """return the average time"""
        return sum(self.times) / len(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()