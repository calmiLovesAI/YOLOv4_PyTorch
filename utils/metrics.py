

class MeanMetric:
    def __init__(self):
        self.accumulated = 0
        self.count = 0

    def update(self, value):
        self.accumulated += value
        self.count += 1
        self.accumulated = self.accumulated / self.count

    def result(self):
        return self.accumulated

    def reset(self):
        self.accumulated = 0
        self.count = 0
