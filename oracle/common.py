
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

def sel_point_tagset(tagsets):
    for tagset in tagsets:
        if is_point_tagset(tagset):
            return tagset
    return None

