import torch


class Rescale:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample, *args, **kwargs):

        pass


class ToTensor:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass