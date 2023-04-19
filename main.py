from ultralytics import YOLO
from supervision.video.source import get_video_frames_generator
from supervision.tools.detections import Detections, BoxAnnotator
import cv2
import numpy as np
from supervision.draw.color import ColorPalette
from supervision.video.sink import VideoSink
from supervision.video.dataclasses import VideoInfo
from tqdm import tqdm
import os
from typing import List

from ByteTrack.yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass

@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

byte_tracker = BYTETracker(BYTETrackerArgs())
# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))


# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections: Detections, 
    tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)
    
    tracker_ids = [None] * len(detections)
    
    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids


def main():
    if os.path.exists('output.mp4'):
        os.remove('output.mp4')
        print("silindi")
    
    model = YOLO("yolov8x.pt")
    model.fuse()
    CLASS_NAMES_DICT = model.model.names
    ##sadece 0 (person) idli objeler için tracking implemente edilir 
    CLASS_ID = [0]

    video_info = VideoInfo.from_video_path("Arıkapı Video.mp4")    
    fourcc = cv2.VideoWriter_fourcc('h', '2', '6', '4')
    out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (1200,700))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)
    text_position = (10, 670)
    cap = cv2.VideoCapture("Arıkapı Video.mp4")
    track_list = []
    while(cap.isOpened()):
        
        ret,frame = cap.read()
        if ret:
            frame = cv2.resize(frame,(1200,700))
            result=model(frame, agnostic_nms=True)[0]
            
            box_annotator = BoxAnnotator(color=ColorPalette(), thickness=4,text_thickness=4, text_scale=2)

            detections = Detections(
                    xyxy= result.boxes.xyxy.cpu().numpy(),
                    confidence = result.boxes.conf.cpu().numpy(),
                    class_id = result.boxes.cls.cpu().numpy().astype(int)
            )
            mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
            detections.filter(mask=mask, inplace=True)
            # tracking detections
            tracks = byte_tracker.update(
                output_results=detections2boxes(detections=detections),
                img_info=frame.shape,
                img_size=frame.shape
            )
            tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
            
            detections.tracker_id = np.array(tracker_id)
            # filtering out detections without trackers
            mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
            detections.filter(mask=mask, inplace=True)
            print(detections)
            labels = [
                f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f} {tracker_id}"
                for _, confidence, class_id, tracker_id
                in detections
        ]  

            if(set(tracker_id)!=track_list):
                out.write(frame)
                out.write(frame)
                out.write(frame)
                out.write(frame)
                out.write(frame)
                out.write(frame)
                out.write(frame)
                out.write(frame)
                out.write(frame)
                out.write(frame)
                print(track_list)
                print(tracker_id)
                print("//////////")
                
            track_list = set([x for x in tracker_id if x is not None])
            cv2.imshow("video", frame)
            if(cv2.waitKey(30)==27):
                    break
        else:
            break
    

if __name__ == '__main__':
    main()