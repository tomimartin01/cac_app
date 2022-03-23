import logging

def check_multiple_detection(count, total):

    if count > 0.1*total:
        return True
        
    return False

def validate_video_format(total_frames, height, width):

    if total_frames > 1000 or height > 800 or width > 1300:
      logging.error(f'Error in video format. Frames: {total_frames}, Height: {height}, Width: {width}')
      return False

    return True

def mp_validate_detection(x, y):

    if not x or not y:
        return False

    return True

def hgh_validate_detection(x, y, multiple_detection):

    if not x or not y or not multiple_detection:
        return False

    return True