

class MeanMetric:
    def __init__(self):
        self.accumulated = 0
        self.count = 0

    def update(self, value):
        self.accumulated += value
        self.count += 1

    def result(self):
        return self.accumulated / self.count

    def reset(self):
        self.__init__()
