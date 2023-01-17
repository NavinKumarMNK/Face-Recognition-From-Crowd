def path2src(path):
    ext = path.split(".")[-1]
    if ext == "mp4" or ext == "avi":
        return "video"
    elif ext == "jpg" or ext == "png":
        return "image"
    elif path == "live":
        return "live"
    else:
        raise Exception("Invalid source")