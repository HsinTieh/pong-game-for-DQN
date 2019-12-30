class LinearSchedule():
    def __init__(self):
        self.start = 1.0
        self.final = 0.01
        self.decay = 1000000
    def value(self,t):
        s = min(float(t) /self.decay, 1.0)
        return self.start + s*(self.final - self.start)

