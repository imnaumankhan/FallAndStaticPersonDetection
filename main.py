import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
import torch
import cv2
import os
import math
import time
import numpy as np
import json
import pika
from multiprocessing import Pool
import threading
from torchvision import transforms
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from datetime import datetime

app = Flask(__name__)

@app.route('/replace_json', methods=['POST'])
def replace_json():
    try:
        new_config = request.get_json()
        with open('config.json', 'w') as f:
            f.write(json.dumps(new_config))
        response = jsonify({'code': 200, 'msg': 'JSON data has been replaced successfully'})
        return response
    except Exception as e:
        response = jsonify({'code': 500, 'msg': f'Error: {str(e)}'})
        return response

def start_flask():
    app.run(host='0.0.0.0', port=5000)

class get_rabbitmq:
    def __init__(self):
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host='127.0.0.1', port=5672, credentials=pika.PlainCredentials('hsfroot', 'Hsf@root123')))
        self.channel = self.connection.channel()

    def producter(self,exchange,queue,message,routing_key=None):
        self.channel.queue_declare(queue,durable=True)
        self.channel.basic_publish(exchange=exchange,routing_key=routing_key,body=message)
        self.channel.close()

with open('C:/fall_and_static_person_detection/config.json') as f:
    data = json.load(f)

devices = data['Devices']
devices_size = data['DevicesSize']

device = devices[0]
choose_algo = device['chooseAlgo']
ip_address = device['ipAddress']
rtsp_url = device['rtspUrl']

algo1 = choose_algo[0]
algo_name1 = algo1['algoName']
roi1 = algo1['roi']
single_alarm_interval1 = algo1['singleAlarmInterval']
single_sensitivity1 = algo1['singleSensitivity']

algo2 = choose_algo[1]
algo_name2 = algo2['algoName']
roi2 = algo2['roi']
single_alarm_interval2 = algo2['singleAlarmInterval']
single_sensitivity2 = algo2['singleSensitivity']

torch.multiprocessing.set_start_method('spawn', force=True)
device = torch.device("cuda:0")
weights = torch.load('pose_model.pt')
model = weights['model']
model = model.half().to(device)
STREAM_FILES = ['streams.txt']
_ = model.eval()
fps = 4

roi_str = data['Devices'][0]['chooseAlgo'][0]['roi']
roi_coords = [float(val) for val in roi_str.split(',')]


def detect(url):
    stream = cv2.VideoCapture(url)

    if stream.isOpened() == False:
        print(f'[!] error opening {url}')

    stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    stream.set(cv2.CAP_PROP_FPS, fps)
    a = 0
    b = 0

    try:
        t_end = time.time() + 30
        while stream.isOpened():
            now = time.time()
            ret, frame = stream.read()
            if ret == True:
                frame_width = int(stream.get(3))
                frame_height = int(stream.get(4))
                vid_write_image = letterbox(stream.read()[1], (frame_width), stride=64, auto=True)[0]
                resize_height, resize_width = vid_write_image.shape[:2]
                orig_image = frame
                image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
                image = letterbox(image, (frame_width), stride=64, auto=True)[0]
                image = transforms.ToTensor()(image)
                image = torch.tensor(np.array([image.numpy()]))
                image = image.to(device)
                image = image.half()

                ts = int(round((now * 1000)))
                date_time_str = datetime.now().strftime("%Y%m%d%H%M%S")
                filename = '%s.jpg' % date_time_str
                path = 'C:/fall_and_static_person_detection/Alarm_images'

                with torch.no_grad():
                    output, _ = model(image)

                output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'],
                                                 kpt_label=True)
                output = output_to_keypoint(output)
                im0 = image[0].permute(1, 2, 0) * 255
                im0 = im0.cpu().numpy().astype(np.uint8)
                im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
                cv2.rectangle(im0,(int(roi_coords[0]),int(roi_coords[1])), (int(roi_coords[2]),int(roi_coords[3])), color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
                for idx in range(output.shape[0]):
                    xmin, ymin = (output[idx, 2] - output[idx, 4] / 2), (output[idx, 3] - output[idx, 5] / 2)
                    xmax, ymax = (output[idx, 2] + output[idx, 4] / 2), (output[idx, 3] + output[idx, 5] / 2)
                    if (xmin >= roi_coords[0] and  
                    ymin >= roi_coords[1] and  
                    xmax <= roi_coords[2] and 
                    ymax <= roi_coords[3]): 
                        if int(xmin) == a and int(ymin) == b:
                            cv2.rectangle(im0, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(0, 0, 255),
                                      thickness=3, lineType=cv2.LINE_AA)
                            int_now = int(time.time())
                            if int_now % single_alarm_interval2 == 0:
                                cv2.imwrite(os.path.join(path, filename), im0)
                                message = json.dumps({"alarmType": algo_name2,"alarmTime": ts,"alarmPicUrl":filename,"cameraUrl": rtsp_url,"cameraIp": ip_address})
                                get_rabbitmq().producter(exchange="processing_event",routing_key="event.behaviorAlarm.#",queue="hw_behaviorAlarm_queue",message=message)
                        int_now = int(time.time())
                        if int_now % single_sensitivity2 == 0:
                            a = int(xmin)
                            b = int(ymin)
                        cv2.rectangle(im0, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

                        left_shoulder_y = output[idx][23]
                        left_shoulder_x = output[idx][22]
                        left_body_y = output[idx][41]
                        left_body_x = output[idx][40]
                        len_factor = math.sqrt(((left_shoulder_y - left_body_y) ** 2 + (left_shoulder_x - left_body_x) ** 2))
                        left_foot_y = output[idx][53]

                        int_now = int(time.time())
                        if left_shoulder_y > left_foot_y - len_factor and left_body_y > left_foot_y - (len_factor / 2) and left_shoulder_y > left_body_y - (len_factor / 2) and int_now % single_sensitivity1 == 0:
                            cv2.rectangle(im0, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(0, 0, 255), thickness=3, lineType=cv2.LINE_AA)
                            int_now = int(time.time())
                            if int_now % single_alarm_interval1 == 0:
                                cv2.imwrite(os.path.join(path, filename), im0)
                                message = json.dumps({"alarmType": algo_name1,"alarmTime": ts,"alarmPicUrl":filename,"cameraUrl": rtsp_url,"cameraIp": ip_address})
                                get_rabbitmq().producter(exchange="processing_event",routing_key="event.behaviorAlarm.#",queue="hw_behaviorAlarm_queue",message=message)

                cv2.namedWindow(url, cv2.WINDOW_NORMAL)
                im1 = cv2.resize(im0, (1280, 720))
                cv2.imshow(url, im1)
                cv2.waitKey(1)
                if time.time() > t_end:
                    stream.release()

            else:
                break
    except KeyboardInterrupt:
        pass
    stream.release()


if __name__ == '__main__':
    flask_thread = threading.Thread(target=start_flask)
    flask_thread.start()
    while True:
        for file_name in STREAM_FILES:
            with open(file_name) as file:
                streams_urls = [line.rstrip() for line in file]
                stream_count = len(streams_urls)

                with Pool(stream_count) as process_pool:
                    process_pool.map(detect, streams_urls)