
# Constants
POINT_TAGSET = 'point_tagset'
ALL_TAGSETS = 'tagsets'
FULL_PARSING = 'fullparsing'


POINT_POSTFIXES = ['sensor', 'setpoint', 'alarm', 'command', 'meter', 'status']
# May use point_tagset_list later.


def is_point_tagset(tagset):
    if tagset in ['unknown', 'none']:
        return True
    category_tag = tagset.split('_')[-1]
    if category_tag in POINT_POSTFIXES:
        return True
    else:
        return False


def select_point_tagset(tagsets, srcid=''):
    for tagset in tagsets:
        if is_point_tagset(tagset):
            return tagset
    print('no point found at {0} in {1}'.format(srcid, tagsets))
    return 'none'


def adder(x, y):
    return x + y
