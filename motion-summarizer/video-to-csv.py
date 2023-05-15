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
import datetime
import pandas as pd
import math

# TRACKING DEFINTIONS
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
    if os.path.exists('byproduct.mp4'):
        os.remove('byproduct.mp4')
        print("previous video removed")

    model = YOLO("yolov8x.pt")
    model.fuse()
    CLASS_NAMES_DICT = model.model.names
    # sadece 0 (person) idli objeler için tracking implemente edilir 
    CLASS_ID = [0]
    

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)
    text_position = (10, 670)

    process_per_sec = 5
    cap = cv2.VideoCapture("lab-record.ts")
    #only process last 30 minutes of video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    start_frame = total_frames - fps * 780
    end_frame = total_frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    # get video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # create video writer
    fourcc =cv2.VideoWriter_fourcc('h', '2', '6', '4')
    out = cv2.VideoWriter('byproduct.mp4', fourcc, process_per_sec, (width, height))
    frame_index=start_frame-1
    rows = []
    # Initialize previous frame
    prev_frame = None
    while True:   
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not ret:
            break
        if(frame_index>end_frame):
            break
        if ret:
            frame_index=frame_index+1
            # Convert frame to grayscale
            if frame_index % math.floor(fps/process_per_sec) == 0:
                # If previous frame is not None, compare with current frame
                if prev_frame is not None:
                    # Calculate absolute difference between frames
                    diff = cv2.absdiff(gray, prev_frame)
                    # Apply threshold to difference image
                    threshold = 200
                    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
                    # Count number of non-zero pixels in difference image
                    nz_count = cv2.countNonZero(thresh)
                    # If difference is detected, process the frame
                    if nz_count > 0:
                        # PROCESS
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
                        labels = [
                            f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f} {tracker_id}"
                            for _, confidence, class_id, tracker_id
                            in detections
                        ]
                        frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
                        out.write(frame)
                        ids = detections.tracker_id
                        boxes = detections.xyxy
                        for i in range(len(ids)):
                            row = {'frame': frame_index, 'object': ids[i], 'coordinate': boxes[i]}
                            print(row)
                            rows.append(row)
                        cv2.imshow("video", frame)
                    pass
            # Store current frame as previous frame
            prev_frame = gray
        # Wait for a key press, but only for a short time
        key = cv2.waitKey(1) & 0xFF

        # Check if the user has pressed the escape key
        if key == 27:
            break
   
    
    if os.path.exists('output.csv'):
        os.remove('output.csv')
        print("previous csv removed")
    df = pd.DataFrame(rows)
    with open('output.csv', mode='a', newline='') as file:
        df.to_csv(file, index=False, header=file.tell()==0)
    print(datetime.datetime.now())
    

if __name__ == '__main__':
    main()