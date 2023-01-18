import os
def path2src(path, num=0):
    ext = path.split(".")[-1]
    if ext == "mp4" or ext == "avi":
        return "video"
    elif ext == "jpg" or ext == "png":
        return "image"
    elif path == "live":
        if num==1:
            return 0
        return "live"
    else:
        raise Exception("Invalid source")

def absolute_path(path):
    return os.path.abspath(path)