
class PlasterError(Exception):
    def __init__(self, *args, **kwargs):
        super(PlasterError, self).__init__(*args, **kwargs)


class UnlabeledError(PlasterError):
    def __init__(self, srcid, label_type, *args, **kwargs):
        self.srcid = srcid
        self.label_type = label_type
        super(UnlabeledError, self).__init__(*args, **kwargs)
    def __str__(self):
        return repr('{0} of {1} has not been labeled.'.format(self.label_type, self.srcid))

class UnlabeledFullparsingError(UnlabeledError):
    def __init__(self, srcid):
        super(UnlabeledFullparsingError, self).__init__(srcid, FULL_PARSING)

class UnlabeledPointTagsetError(UnlabeledError):
    def __init__(self, srcid):
        super(UnlabeledPointTagsetError, self).__init__(srcid, POINT_TAGSET)

class UnlabeledTagsetsError(UnlabeledError):
    def __init__(self, srcid):
        super(UnlabeledTagsetsError, self).__init__(srcid, ALL_TAGSETS)

class NotEnoughExamplesError(PlasterError):
    def __init__(self, num_examples, min_num_examples, *args, **kwargs):
        self.num_examples = num_examples
        self.min_num_examples = min_num_examples
        super(NotEnoughExamplesError, self).__init__(*args, **kwargs)

    def __str__(self):
        return repr('The number of training examples ({0}) are less than the minimum requirements ({1})'.format(
            self.num_examples, self.min_num_examples))

