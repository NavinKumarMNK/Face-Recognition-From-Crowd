import cv2
from flask import Flask, render_template, Response
import webbrowser
import os
import psutil
from scripts.process import Process

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

'''
def main():

    cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 6)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            else:
                # write on the frame and send it to the client
                cv2.putText(
                    frame,
                    "Press 'q' to quit",
                    (20, 70),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (0, 255, 0),
                    2,
                )
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n'
                    b'Content-Length: ' + str(len(frame)).encode() + b'\r\n\r\n' + frame + b'\r\n')

'''

@app.route('/video-feed')
def video_feed():
    return Response(Process(os.path.abspath('./temp'), os.path.abspath('./weights/yolo-tinyface.pt')).set_capture('live').start_capture(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


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
    port=5002
    browser = webbrowser.get()
    url=f'http://{ip}:{port}/video-feed'
    if browser is None:
        webbrowser.open(url)
    else:
        browser.open(url, new=1)
    app.run(host='0.0.0.0', debug=True, port=port )