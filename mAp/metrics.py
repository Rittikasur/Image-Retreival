import numpy as np
import os

PROCESSED_FOLDER = "../processed_data"
#https://github.com/tensorflow/models/blob/master/research/delf/delf/python/datasets/google_landmarks_dataset/metrics.py
def _CountPositives(image_name):
    image_path = os.path.join(PROCESSED_FOLDER,image_name[:len(image_name)-6])
    if(os.path.exists(image_path) and len(os.listdir(image_path)>0)):
        return len(os.listdir(image_path))
    else:
        print("Positives NOt Found")

def GlobalAveragePrecision(predictions,ground_truth):
    num_positives = 3
    # num_positives = _CountPositives(ground_truth)
    gap = 0.0
    total_predictions = 0
    correct_predictions = 0
    #Apparenty we have to sort the predictions. IDK whys
    for index,prediction in enumerate(predictions):
        total_predictions += 1
        if ground_truth.__contains__(prediction):
            correct_predictions += 1
            gap += correct_predictions / total_predictions
    gap /= num_positives
    print(gap)
    return gap