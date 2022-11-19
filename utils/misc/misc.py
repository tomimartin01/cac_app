import logging

def check_multiple_detection(count, total):

    if count < 0.3*total:
        return True
        
    return False

def mp_validate_detection(x, y):

    if not x or not y:
        return False

    return True

def hgh_validate_detection(x, y, multiple_detection):

    if not x or not y or multiple_detection:
        return False

    return True