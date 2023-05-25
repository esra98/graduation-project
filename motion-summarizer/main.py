import pandas as pd
import cv2
import os
from math import floor
from shapely.geometry import Polygon
import ast
from datetime import datetime, timedelta
import re
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from tabulate import tabulate
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image, Spacer
import numpy as np

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

def convert_yolo_to_opencv(yolo_box, image_width, image_height):
    center_x, center_y, box_width, box_height = yolo_box

    # Convert YOLO coordinates to absolute values
    abs_center_x = int(center_x * image_width)
    abs_center_y = int(center_y * image_height)
    abs_box_width = int(box_width * image_width)
    abs_box_height = int(box_height * image_height)

    # Convert YOLO coordinates to OpenCV format
    x = int(abs_center_x - abs_box_width / 2)
    y = int(abs_center_y - abs_box_height / 2)
    width = abs_box_width
    height = abs_box_height

    return x, y, width, height

def frame_merger(frame_1, frame_2, obj_1, obj_2, most_recurrent):
    frame_2 = frame_2
    # create a VideoCapture object and open the video file
    cap = cv2.VideoCapture('lab-record.ts')
    # check if the video file was opened successfully
    if not cap.isOpened():
        print('Error opening video file')
    # set the frame index to the desired frame number (e.g., 100)
    frame_index = frame_1
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    # read the frame at the specified index
    ret, frame = cap.read()
    # check if the frame was successfully read
    if not ret:
        print(f'Error reading first frame {frame_index}')
    # save the frame as a JPEG image
    cv2.imwrite('frame1.jpg', frame)
     # set the frame index to the desired frame number (e.g., 100)
    frame_index = frame_2
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
    # Set the color and position of the rectangle
    rectangle_color = (128, 128, 128)  # Gray color in BGR format
    rectangle_x1 = 0
    rectangle_y1 = 0
    rectangle_x2 = 1060  # Adjust the width of the rectangle as needed
    rectangle_y2 = 50   # Adjust the height of the rectangle as needed

    # Draw the rectangle on the canvas
    cv2.rectangle(
        img=canvas,
        pt1=(rectangle_x1, rectangle_y1),
        pt2=(rectangle_x2, rectangle_y2),
        color=rectangle_color,
        thickness=cv2.FILLED
    )

    # Set the text to display
    text = " +10min|  5min.-10min.|  1min.-5min.|  2sec.-1min.|  < 2 sec."

    # Set the position of the text
    text_x = rectangle_x1 + 10  # Adjust the x-coordinate to provide padding
    text_y = rectangle_y1 + 30  # Adjust the y-coordinate to center the text

    # Draw the text on the canvas
    cv2.putText(
        img=canvas,
        text=text,
        org=(text_x, text_y),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(255, 255, 255),  # White color in BGR format
        thickness=1,
        lineType=cv2.LINE_AA
    )
    cv2.circle(canvas, (10, 20), radius=10, color = (0, 0, 255), thickness=-1)
    cv2.circle(canvas, (170, 20), radius=10, color = (0, 0, 100), thickness=-1)
    cv2.circle(canvas, (430, 20), radius=10, color = (0, 100, 100), thickness=-1)
    cv2.circle(canvas, (670, 20), radius=10, color = (0, 100, 0), thickness=-1)
    cv2.circle(canvas, (910, 20), radius=10, color = (0, 255, 0), thickness=-1)
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
                w = int(float_values[2])
                x = int(float_values[0])
                y = int(float_values[1])
                h = int(float_values[3])
                if str(csv_color) == "1":
                    color = (0, 0, 255)
                elif str(csv_color) == "2":
                    color = (0, 0, 100)
                elif str(csv_color) == "3":
                    color = (0, 100, 100)
                elif str(csv_color) == "4":
                    color = (0, 100, 0)
                else:
                    color = (0, 255, 0)
                roi = img1[y:y+h, x:x+w]
                canvas[y:y+h, x:x+w] = roi
                # labelling
                text = (first_frame_time + " / " + time_span)
                text_width, text_height = cv2.getTextSize(
                    text=text,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    thickness=1,
                )[0]
                text_x = x + 10
                text_y = y - 10
                text_background_x1 = x
                text_background_y1 = y - 2 * 10 - text_height
                text_background_x2 = x + 2 * 10 + text_width
                text_background_y2 = y
                cv2.rectangle(
                    img=canvas,
                    pt1=(text_background_x1, text_background_y1),
                    pt2=(text_background_x2, text_background_y2),
                    color=color,
                    thickness=cv2.FILLED,
                )
                cv2.putText(
                    img=canvas,
                    text=text,
                    org=(text_x, text_y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1   ,
                    color = (0, 0, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

                

    obj_cordinates_frame_2_raw =  df[df['frame'].isin([frame_2])]['coordinate'].tolist()
    obj_id_frame_2_raw =  df[df['frame'].isin([frame_2])]['object'].tolist()
    if(len(obj_cordinates_frame_2_raw)==len(obj_id_frame_2_raw)):
        for i in range(0, len(obj_cordinates_frame_2_raw)):
            first_frame_time, last_frame_time, time_span, csv_color = get_frame_difference(obj_2)
            string_values = obj_cordinates_frame_2_raw[i].strip('[]').split()
            float_values = [float(val) for val in string_values]
            if(obj_2==obj_id_frame_2_raw[i]):
                # Define the region of interest (ROI) you want to cut out from this float values
                w = int(float_values[2])
                x = int(float_values[0])
                y = int(float_values[1])
                h = int(float_values[3])
                if str(csv_color) == "1":
                    color = (0, 0, 255)
                elif str(csv_color) == "2":
                    color = (0, 0, 100)
                elif str(csv_color) == "3":
                    color = (0, 100, 100)
                elif str(csv_color) == "4":
                    color = (0, 100, 0)
                else:
                    color = (0, 255, 0)
                roi = img2[y:y+h, x:x+w]
                canvas[y:y+h, x:x+w] = roi
                # labelling
                text = (first_frame_time + " / " + time_span)
                text_width, text_height = cv2.getTextSize(
                    text=text,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    thickness=1,
                )[0]
                text_x = x + 10
                text_y = y - 10
                text_background_x1 = x
                text_background_y1 = y - 2 * 10 - text_height
                text_background_x2 = x + 2 * 10 + text_width
                text_background_y2 = y
                cv2.rectangle(
                    img=canvas,
                    pt1=(text_background_x1, text_background_y1),
                    pt2=(text_background_x2, text_background_y2),
                    color=color,
                    thickness=cv2.FILLED,
                )
                cv2.putText(
                    img=canvas,
                    text=text,
                    org=(text_x, text_y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1   ,
                    color = (0, 0, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

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
    
def get_middle_row_data(object_id):
    df = pd.read_csv('output.csv')
    object_df = df[df['object'] == object_id]
    if not object_df.empty:
        middle_index = object_df.shape[0] // 2
        row = object_df.iloc[middle_index]
        frame = row['frame']
        coordinate = row['coordinate']
        coordinate = [float(value) for value in coordinate.strip('[]').split()]
        return frame, coordinate
    return None, None  # Return None if no row matches the object ID

def get_first_row_coordinate(object_id):
    df = pd.read_csv('output.csv')
    object_df = df[df['object'] == object_id]
    if not object_df.empty:
        coordinate = object_df.iloc[0]['coordinate']
        coordinate = [float(value) for value in coordinate.strip('[]').split()]
        return coordinate
    return None  # Return None if no row matches the object ID

def get_last_row_coordinate(object_id):
    df = pd.read_csv('output.csv')
    object_df = df[df['object'] == object_id]
    if not object_df.empty:
        last_index = object_df.shape[0] - 1
        coordinate = object_df.iloc[last_index]['coordinate']
        coordinate = [float(value) for value in coordinate.strip('[]').split()]
        return coordinate
    return None  # Return None if no row matches the object ID
#to improve color in the report
def update(dot_color):
    r, g, b = dot_color

    # Update the color by reducing the green and blue channels
    r += 0
    g -= 1
    b += 1

    # Make sure the color values stay within the valid range (0-255)
    r = max(0, min(r, 255))
    g = max(0, min(g, 255))
    b = max(0, min(b, 255))

    return (r, g, b)
def mapper(id):
    # Load the CSV file
    csv_file = 'output.csv'

    # Read the image with 20% opacity
    image_path = 'blank.jpg'
    image = cv2.imread(image_path)
    overlay = np.zeros_like(image)
    alpha = 0.9  # Opacity of 20%
    background = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

    # Define the dot color with 10% opacity
    dot_color = (0, 255, 0)  # Green color with 10% opacity

    # Read the CSV file into a DataFrame using pandas
    df = pd.read_csv(csv_file)

    # Filter rows where object == 1
    df_filtered = df[df['object'] == id]

    # Iterate through each row in the filtered DataFrame
    for index, row in df_filtered.iterrows():
        coordinates = row['coordinate']
        # Parse the coordinates
        x1, y1, x2, y2 = map(float, coordinates.strip('[]').split())

        # Calculate the middle point of the bounding box
        x_middle = int((x1 + x2) / 2)
        y_middle = int((y1 + y2) / 2)
        # Draw a dot at the middle point on the image with 10% opacity
        cv2.circle(background, (int(x_middle), int(y_middle)), radius=2, color=dot_color, thickness=-1)
        #update color
        dot_color = update(dot_color)
        

    # Save the image with dots and 20% opacity
    output_image_path = 'mapped_'+str(id)+'.jpg'
    cv2.imwrite(output_image_path, background)

def reporter():
    cap = cv2.VideoCapture('lab-record.ts')
    df = pd.read_csv('object_first_last_frames.csv')
    table_data = df[['object_id','first_frame', 'last_frame', 'first_frame_time', 'last_frame_time', 'time_span']]
    table_data_list = table_data.values.tolist()
    pdf_file = 'report.pdf'

    # Create a SimpleDocTemplate object
    doc = SimpleDocTemplate(pdf_file, pagesize=letter)
    story = []

    # Define table styles
    table_style = TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('BACKGROUND', (0, 0), (-1, 0), colors.gray),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
        ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
    ])

    for i, row in enumerate(table_data_list):
        object_id = row[0]
        first_frame = row[1]
        last_frame = row[2]
        first_frame_time = row[3]
        last_frame_time = row[4]
        time_span = row[5]

        # Save the images with unique filenames
        person_enter_filename = f'person_enter_{object_id}.jpg'
        person_middle_filename = f'person_middle_{object_id}.jpg'
        person_exit_filename = f'person_exit_{object_id}.jpg'

        #ADD ENTRANCE OF OBJECT IMAGE 
        coordinate = get_first_row_coordinate(object_id)
        # Define the region of interest (ROI) you want to cut out from this float values
        person_w = int(coordinate[2])  
        person_x = int(coordinate[0]) 
        person_y = int(coordinate[1])
        person_h = int(coordinate[3])
        # Save the images
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
        ret, frame = cap.read()
        cropped_image = frame[person_y:person_h, person_x:person_w]
        cv2.imwrite(person_enter_filename, cropped_image)

        middle_frame, coordinate = get_middle_row_data(object_id)
        # Define the region of interest (ROI) you want to cut out from this float values
        person_w = int(coordinate[2])  
        person_x = int(coordinate[0]) 
        person_y = int(coordinate[1])
        person_h = int(coordinate[3])
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        ret, frame = cap.read()
        cropped_image = frame[person_y:person_h, person_x:person_w]
        cv2.imwrite(person_middle_filename, cropped_image)

        coordinate = get_last_row_coordinate(object_id)
        # Define the region of interest (ROI) you want to cut out from this float values
        person_w = int(coordinate[2])  
        person_x = int(coordinate[0]) 
        person_y = int(coordinate[1])
        person_h = int(coordinate[3])
        cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame)
        ret, frame = cap.read()
        cropped_image = frame[person_y:person_h, person_x:person_w]
        cv2.imwrite(person_exit_filename, cropped_image)
        # rewrite mapped.jpg image file for object
        mapper(object_id)
        
        # Create the nested table for the row
        nested_table_data = [
            ['Enter', 'Middle', 'Exit', 'Path'],
            [
                Image(person_enter_filename, width=100, height=100),
                Image(person_middle_filename, width=100, height=100),
                Image(person_exit_filename, width=100, height=100),
                Image('mapped_'+str(object_id)+'.jpg', width=100, height=100),
                ''
            ],
            ['Enter Time','', 'Exit Time', 'Duration'],
            [first_frame_time, '',last_frame_time, time_span]
        ]
        nested_table = Table(nested_table_data)
        nested_table.setStyle(table_style)

        # Add the nested table to the main table
        story.append(nested_table)

        # Check if there is enough space for the next row
        if i < len(table_data_list) - 1:
            story.append(Spacer(1, 20))  # Add some spacing between rows

    # Build the PDF document
    doc.build(story)

def main():
    unique_objects_list = unique_objects("output.csv")
    object_time_csv_former("output.csv", "13:00")
    reporter()
    most_recurrent = find_object_with_highest_time_span("object_first_last_frames.csv")
    important_frames_list= significant_frames_total_list(unique_objects_list)
    list1,list2 = pair_list_id_frame(important_frames_list,unique_objects_list)
    print(list1,list2)
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
