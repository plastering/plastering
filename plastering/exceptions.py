
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
