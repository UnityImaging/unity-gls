class VirtualDataLoader:

    def __init__(self, data_loader, steps_per_epoch: int = 1000):
        self.data_loader = data_loader
        self.iterator = iter(self.data_loader)
        self.steps_per_epoch = steps_per_epoch
        self.current_step = 0

    def __len__(self):
        return self.steps_per_epoch
    def __iter__(self):
        return self

    def __next__(self):
        if self.current_step < self.steps_per_epoch:
            self.current_step += 1
            try:
                return next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.data_loader)
                return next(self.iterator)
        else:
            self.current_step = 0
            raise StopIteration

