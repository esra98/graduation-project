import pandas as pd
import cv2
import os
from math import floor
from shapely.geometry import Polygon
import ast
from datetime import datetime, timedelta
import re

def find_object_with_highest_time_span(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Convert time_span column to timedelta format
    df['time_span'] = pd.to_timedelta(df['time_span'])

    # Find the object with the highest time_span
    max_time_span_row = df.loc[df['time_span'].idxmax()]
    max_time_span_obj_id = max_time_span_row['object_id']

    return max_time_span_obj_id

def object_time_csv_former(original_csv, time):      
    # Read the original CSV file into a Pandas dataframe
    df = pd.read_csv(original_csv)
    # Set initial time
    time = datetime.strptime(time, "%H:%M")
    # Group the dataframe by object ID
    grouped = df.groupby('object')

    # Create a new dataframe to store the first and last frames for each object
    new_df = pd.DataFrame(columns=['object_id', 'first_frame', 'last_frame', 'frame_difference'])

    # Loop through each object
    for object_id, group in grouped:
        # Get the first and last frames for the current object
        first_frame = group['frame'].min()
        last_frame = group['frame'].max()
        frame_difference = last_frame - first_frame
        # Increment time by seconds
        video = cv2.VideoCapture("lab-record.ts")
        fps = video.get(cv2.CAP_PROP_FPS)
        seconds_to_add_first_frame = first_frame/fps
        seconds_to_add_last_frame= last_frame/fps
        first_frame_time_raw = time + timedelta(seconds=seconds_to_add_first_frame)
        last_frame_time_raw = time + timedelta(seconds=seconds_to_add_last_frame)
        first_frame_time = first_frame_time_raw.strftime("%H:%M")
        last_frame_time = last_frame_time_raw.strftime("%H:%M")
        time_difference = re.sub(r"\.\d+", "", str(last_frame_time_raw - first_frame_time_raw))
        # Check if the duration falls within the range
        if timedelta(minutes=10) <= (last_frame_time_raw - first_frame_time_raw) <= timedelta(minutes=30):
            color = "1"
        elif timedelta(minutes=5) <= (last_frame_time_raw - first_frame_time_raw) <= timedelta(minutes=10):
            color = "2"
        elif timedelta(minutes=1) <= (last_frame_time_raw - first_frame_time_raw) <= timedelta(minutes=5):
            color = "3"
        elif timedelta(seconds=2) <= (last_frame_time_raw - first_frame_time_raw) <= timedelta(minutes=1):
            color = "4"
        else:
            color = "5"
        # Add the first and last frames to the new dataframe
        new_df = pd.concat([new_df, pd.DataFrame({
            'object_id': [object_id],
            'first_frame': [first_frame],
            'first_frame_time': [first_frame_time],
            'last_frame': [last_frame],
            'last_frame_time': [last_frame_time],
            'frame_difference': [frame_difference],
            'time_span': [time_difference],
            'color': [color]
        })])

    # Save the new dataframe to a CSV file
    new_df.to_csv('object_first_last_frames.csv', index=False)

def unique_objects(original_csv):
     # load the csv file into a dataframe
    df = pd.read_csv('output.csv')
    # get all unique values in the "object" column
    unique_objects = df['object'].unique()
    return unique_objects

def significant_frames_total_list(unique_objects):
    important_frames_list = []
    # print the unique values
    for el in unique_objects:
        frames = single_object_significant_frames(el)
        important_frames_list.append(frames)
    return important_frames_list

def color_list_with_frequency():
    df = pd.read_csv('object_first_last_frames.csv')
    frequency_list = df['frame_difference'].tolist()

def single_object_significant_frames(object_id):
    # Read CSV file into a pandas dataframe
    df = pd.read_csv('output.csv')
    df = df[df['object'] == object_id]
    # Convert the 'coordinate' column to a list of lists
    df['coordinate'] = df['coordinate'].apply(lambda x: [float(val.strip()) for val in x.strip('[]').split()])

    # Initialize the reference box using the first row of the dataframe
    reference_box = df.iloc[0]['coordinate']
    reference_poly = Polygon([[reference_box[0],reference_box[1]],[reference_box[2],reference_box[1]],[reference_box[2],reference_box[1]],[reference_box[0],reference_box[3]]])
    # Create an empty list to store the frames where the current box does not overlap with the reference box
    nonoverlapping_frames = [df.iloc[0]['frame']]
    # Iterate over the rows of the dataframe starting from the second row
    for index, row in df.iloc[1:].iterrows():
        current_box = row['coordinate']
        current_frame = row['frame']
        current_poly = Polygon([[current_box[0],current_box[1]],[current_box[2],current_box[1]],[current_box[2],current_box[1]],[current_box[0],current_box[3]]])
        iou = reference_poly.intersection(current_poly).area / reference_poly.union(current_poly).area
        if (iou < 0.2):
            reference_poly=current_poly
            nonoverlapping_frames.append(current_frame)    
    return nonoverlapping_frames

def get_frame_difference(object_id):
    data = pd.read_csv('object_first_last_frames.csv')
    data.set_index('object_id', inplace=True)
    return data.loc[object_id, 'first_frame_time'],data.loc[object_id, 'last_frame_time'],data.loc[object_id, 'time_span'],data.loc[object_id, 'color']

def frame_merger(frame_1, frame_2, obj_1, obj_2, most_recurrent):
    frame_2 = frame_2
    # create a VideoCapture object and open the video file
    cap = cv2.VideoCapture('lab-record.ts')
    # check if the video file was opened successfully
    if not cap.isOpened():
        print('Error opening video file')
    # set the frame index to the desired frame number (e.g., 100)
    frame_index = frame_1-25
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    # read the frame at the specified index
    ret, frame = cap.read()
    # check if the frame was successfully read
    if not ret:
        print(f'Error reading first frame {frame_index}')
    # save the frame as a JPEG image
    cv2.imwrite('frame1.jpg', frame)
     # set the frame index to the desired frame number (e.g., 100)
    frame_index = frame_2-25
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    # read the frame at the specified index
    ret, frame = cap.read()
    # check if the frame was successfully read
    if not ret:
        print(f'Error reading first frame {frame_index}')
    # save the frame as a JPEG image
    cv2.imwrite('frame2.jpg', frame)

    # Load the image you want to crop from
    img1 = cv2.imread('frame1.jpg')
    img2 = cv2.imread('frame2.jpg')
    # Set the opacity for each image
    alpha = 0.5
    beta = 1 - alpha

    # Blend the images using addWeighted
    canvas = cv2.addWeighted(img1, alpha, img2, beta, 0)
    # load the csv file into a dataframe
    df = pd.read_csv('output.csv')
    obj_cordinates_frame_1_raw =  df[df['frame'].isin([frame_1])]['coordinate'].tolist()
    obj_id_frame_1_raw =  df[df['frame'].isin([frame_1])]['object'].tolist()
    if(len(obj_cordinates_frame_1_raw)==len(obj_id_frame_1_raw)):
        for i in range(0, len(obj_cordinates_frame_1_raw)):
            if(obj_1==obj_id_frame_1_raw[i]):
                first_frame_time, last_frame_time, time_span, csv_color = get_frame_difference(obj_1)
                string_values = obj_cordinates_frame_1_raw[i].strip('[]').split()
                float_values = [float(val) for val in string_values]
                # Define the region of interest (ROI) you want to cut out from this float values
                x = int(float_values[0])-10
                y = int(float_values[1])-10
                w = int(float_values[2])+20
                h = int(float_values[3])+20
                if str(csv_color) == "1":
                    color = (0, 0, 255)
                elif str(csv_color) == "2":
                    color = (100, 100, 255)
                elif str(csv_color) == "3":
                    color = (255, 0, 255)
                elif str(csv_color) == "4":
                    color = (255, 0, 0)
                else:
                    color = (0, 0, 0)
                thickness = 2    
                roi = img1[y:y+h, x:x+w]
                canvas[y:y+h, x:x+w] = roi
                cv2.rectangle(canvas, (x, y), (w,  h), color, thickness)
                if(obj_1 == most_recurrent):
                    cv2.rectangle(canvas, (x, y), (w,  h), (0, 0, 255), 5)
                    cv2.putText(canvas, "MOST RECURRENT" + str(time_span) + str(first_frame_time), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                else:
                    cv2.putText(canvas, str(time_span) + str(first_frame_time), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    obj_cordinates_frame_2_raw =  df[df['frame'].isin([frame_2])]['coordinate'].tolist()
    obj_id_frame_2_raw =  df[df['frame'].isin([frame_2])]['object'].tolist()
    if(len(obj_cordinates_frame_2_raw)==len(obj_id_frame_2_raw)):
        for i in range(0, len(obj_cordinates_frame_2_raw)):
            if(obj_2==obj_id_frame_2_raw[i]):
                first_frame_time, last_frame_time, time_span, csv_color = get_frame_difference(obj_2)
                string_values = obj_cordinates_frame_2_raw[i].strip('[]').split()
                float_values = [float(val) for val in string_values]
                # Define the region of interest (ROI) you want to cut out from this float values
                x = int(float_values[0])-10
                y = int(float_values[1])-10
                w = int(float_values[2])+20
                h = int(float_values[3])+20
                if str(csv_color) == "1":
                    color = (0, 0, 255)
                elif str(csv_color) == "2":
                    color = (100, 100, 255)
                elif str(csv_color) == "3":
                    color = (255, 0, 255)
                elif str(csv_color) == "4":
                    color = (255, 0, 0)
                else:
                    color = (0, 0, 0)
                thickness = 2    
                roi = img2[y:y+h, x:x+w]
                canvas[y:y+h, x:x+w] = roi
                cv2.rectangle(canvas, (x, y), (w,  h), color, thickness)
                if(obj_2 == most_recurrent):
                    cv2.rectangle(canvas, (x, y), (w,  h), (0, 0, 255), 5)
                    cv2.putText(canvas, "MOST RECURRENT" + str(time_span) + str(first_frame_time), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                else:
                    cv2.putText(canvas, str(time_span) + str(first_frame_time), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Define the filename to save the image as
    filename = 'merged.jpg'
    # Save the image
    cv2.imwrite(filename, canvas)
        
            

def pair_list_id_frame(baselist, obj_list):  
    baselist = baselist
    obj_list = obj_list
    if len(baselist)>=2:
        first_list=[]
        second_list=[]
        for j in range(0, len(baselist)):
            if(len(first_list)<=len(second_list)):
                for i in range(0,len(baselist[j])):
                    # create a dictionary for holding obj i & respective frame
                    json_obj1 = {
                        "obj_id": obj_list[j],
                        "frame": baselist[j][i] 
                    }
                    # add the first JSON object to the list
                    first_list.append(json_obj1)
            elif(len(first_list)>len(second_list)):
                for i in range(0,len(baselist[j])):
                    # create a dictionary for holding obj i & respective frame
                    obj = {
                        "obj_id":obj_list[j],
                        "frame": baselist[j][i] 
                    }
                    second_list.append(obj)
    if len(first_list) > len(second_list):
        while len(second_list) < len(first_list):
            second_list.append({"obj_id":0,"frame": ""})
    else:
        while len(first_list) < len(second_list):
            first_list.append({"obj_id":0,"frame": ""})
    return first_list,second_list
    

def main():
    unique_objects_list = unique_objects("output.csv")
    object_time_csv_former("output.csv", "13:00")
    most_recurrent = find_object_with_highest_time_span("object_first_last_frames.csv")
    important_frames_list= significant_frames_total_list(unique_objects_list)
    list1,list2 = pair_list_id_frame(important_frames_list,unique_objects_list)
 
    # create video writer
    cap = cv2.VideoCapture("lab-record.ts")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc =cv2.VideoWriter_fourcc('h', '2', '6', '4')
    out = cv2.VideoWriter('output.mp4', fourcc, 2, (width, height))
    
    if(len(list1)==len(list2)):
        for i in range(0,len(list1)):
            frame_merger(list1[i]["frame"],list2[i]["frame"],list1[i]["obj_id"],list2[i]["obj_id"],most_recurrent)
            image = cv2.imread("merged.jpg")
            out.write(image)           
    

if __name__ == '__main__':
    main()