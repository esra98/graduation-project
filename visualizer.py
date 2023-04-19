import re
import json
import ast
import matplotlib.pyplot as plt
import numpy as np
import datetime

def visualize():
    with open('file.txt', 'r') as input_file:

        input_content = input_file.read()
        pattern = re.compile(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]')
        sections = pattern.split(input_content)
        sections.pop(0)

        cleaned_list = []

        class_id_pattern = r'"class_id":\s*\[([\d\s,]+)\]'

        if(len(sections)%2==0):
            # Print the resulting sections
            for i in range(0, len(sections), 2):
                class_id_str = re.search(r"class_id=array\((.*?)\)", sections[i+1]).group(1)
                class_id_list = [int(x) for x in class_id_str.strip('[]').replace(',', '').split()]
                class_id_dict = {x: class_id_list.count(x) for x in class_id_list}
                my_dict = {"date": sections[i], "detections": class_id_dict}
                cleaned_list.append(json.loads(json.dumps(my_dict)))


    data = cleaned_list
    unique_detections = set()
    for item in data:
        unique_detections.update(item['detections'].keys())

    # Create a dictionary to store the occurrences of each detection over time
    detection_occurrences = {}
    for detection in unique_detections:
        detection_occurrences[detection] = []

    # Store the occurrences of each detection over time
    for item in data:
        for detection in unique_detections:
            if detection in item['detections']:
                detection_occurrences[detection].append(item['detections'][detection])
            else:
                detection_occurrences[detection].append(0)

    # Convert the timestamps to datetime objects
    timestamps = [item['date'] for item in data]
    timestamps = [datetime.datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') for ts in timestamps]

    # Plot the occurrences of each detection over time
    for detection in unique_detections:
        plt.plot(timestamps, detection_occurrences[detection], label=detection)

    # Add a legend and axis labels
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Occurrences')

    # Rotate and align the x-axis labels
    plt.gcf().autofmt_xdate()

    # Show the plot
    plt.show()

visualize()