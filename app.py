import cv2
from flask import Flask, render_template, Response, send_from_directory
import webbrowser
import os
import psutil
from process import Process
import multiprocessing as mp
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
from threading import Thread

app = Flask(__name__, template_folder='./templates')

image_queue = mp.Queue()



@app.route('/test/<path:path>')
def send_image(path):
    return send_from_directory('images', path)

class MyHandler(FileSystemEventHandler):
    def __init__(self, queue):
        self.queue = queue
        self.previous_files = set()

    def on_modified(self, event):
        if event.is_directory:
            print(f"directory modified:{event.src_path}")
            current_files = set(os.listdir(event.src_path))
            new_files = current_files - self.previous_files
            for file in new_files:
                if os.path.isfile(os.path.join(event.src_path, file)):
                    file_path = os.path.join(event.src_path, file)
                    self.queue.put(file_path)
                    print(file_path)
            self.previous_files = current_files
                
def image_capture(queue):
    observer = Observer()
    event_handler = MyHandler(queue)
    observer.schedule(event_handler, path='./temp', recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()



@app.route('/video-feed')
def video_feed():
    video = Process(os.path.abspath('./temp'), 
                    './weights/yolov7-tinyface.pt', 'live')
    video_feed_process = mp.Process(target=video.start_capture
                    )
    video_feed_process.start()
    
    return Response(video.start_capture(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def image_stream():
    while True:
        image_path = image_queue.get()
        if image_path:
            with open(image_path, 'rb') as f:
                image = f.read()
                print(image)
                yield 'data: {}\n\n'.format(image_path)

@app.route('/image-feed')
def image_feed():
    images_process = mp.Process(target=image_capture, args=(image_queue, ))
    images_process.start()
    return Response(image_stream(),  mimetype='text/event-stream')


@app.route('/')
def index():
    return render_template('index.html'
                            , video_feed_url='/video-feed'
                            , image_feed_url='/image-feed')
    
                    
def get_interface_ip(ifname):
    for interface, addrs in psutil.net_if_addrs().items():
        if interface == ifname:
            for addr in addrs:
                if addr.family == 2:
                    return addr.address
    return None


eth0_ip = get_interface_ip('eth0')
wlo1_ip = get_interface_ip('wlo1')

if __name__ == '__main__':
    if eth0_ip:    
        ip = eth0_ip
    elif wlo1_ip:
        ip = wlo1_ip
    port=5005
    '''
    browser = webbrowser.get()
    url=f'http://{ip}:{port}/'
    if browser is None:
        webbrowser.open(url)
    else:
        browser.open(url, new=0)
    '''
    app.run(host='0.0.0.0', debug=True, port=port )