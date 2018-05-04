class BaseOracleException(Exception):
    def __init__(self, msg):
        self.msg = msg

class EmptyTrainingSamples(BaseOracleException):
    def __init__(self):
        super(EmptyTrainingSamples, self).__init__('Empty training samples')
