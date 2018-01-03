

class FrameworkInterface(object):
    """docstring for FrameworkInterface"""
    def __init__(self, conf):
        super(FrameworkInterface, self).__init__()
        self.conf = conf

    def learn(self, input_data):
        pass

    def active_learn(self, labeled_data):
        pass

    def infer(self):
        pass

    def result_summary(self):
        pass
