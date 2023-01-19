import os
import configparser

ROOT_PATH = '/home/mnk/MegNav/Projects/Face-Recognition-from-crowd'


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

# Congifuration file
def config_parse(txt):
    config = configparser.ConfigParser()
    path = ROOT_PATH + '/project.cfg'
    config.read(path)
    params={}
    try:
        for key, value in config[txt].items():
            if 'path' in key:
                params[key] = absolute_path(value)
            else:
                params[key] = value
    except KeyError as e:
        print("Invalid key: ", e)
        print(path)    
    
    return params
