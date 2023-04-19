import cv2
import argparse
import supervision as sv
from ultralytics import YOLO
import numpy as np
import os
import datetime


def main():

    if os.path.exists('output.mp4'):
        os.remove('output.mp4')
        print("slisndi")

    cap = cv2.VideoCapture("example-video.mp4")

    model = YOLO("yolov8n.pt")

    box_annotator = sv.BoxAnnotator(
        thickness = 2,
        text_thickness = 2,
        text_scale = 1
    )
    num_people_previous = 0

    fourcc = cv2.VideoWriter_fourcc('h', '2', '6', '4')
    out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (1200,700))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)
    text_position = (10, 670)

    while(cap.isOpened()):
        ret,frame = cap.read()
        frame = cv2.resize(frame,(1200,700))
        result=model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)
        num_people = 0
        for id in detections.class_id.tolist():
            if id == 0:
                num_people += 1
        if(num_people!=num_people_previous):
            current_time = datetime.datetime.now()
            timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, text_position, font, font_scale, font_color, 2)
            out.write(frame)
            print("bas")
        else:
            print("basma")
        num_people_previous=num_people
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]
        frame = box_annotator.annotate(scene=frame, detections = detections, labels= labels)
        cv2.imshow("video", frame)
        if(cv2.waitKey(30)==27):
                break
        
    

if __name__ == '__main__':
    main()
