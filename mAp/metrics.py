import numpy as np
import os

PROCESSED_FOLDER = "D:/ORG India/data/processed_data"
#https://github.com/tensorflow/models/blob/master/research/delf/delf/python/datasets/google_landmarks_dataset/metrics.py
def _CountPositives(image_name):
    image_path = os.path.join(PROCESSED_FOLDER,image_name[:len(image_name)-6])
    if(os.path.exists(image_path) and len(os.listdir(image_path))>0):
        return len(os.listdir(image_path))
    else:
        print("Positives NOt Found")
def _SortPredictions(predictions,ground_truth):
    truth_list = []
    false_list = []
    ground_truth = ground_truth[14:len(ground_truth)-6]
    for prediction in predictions:
        if prediction.__contains__(ground_truth):
            truth_list.append(prediction)
        else:
            false_list.append(prediction)
    return truth_list + false_list
    
def GlobalAveragePrecision(predictions,ground_truth):
    # num_positives = 3
    # Switch it up - if ew decide to use the processed 2 folder
    num_positives = _CountPositives(ground_truth)
    predictions = _SortPredictions(predictions,ground_truth)
    ground_truth = ground_truth[14:len(ground_truth)-6]
    gap = 0.0
    total_predictions = 0
    correct_predictions = 0
    #Apparenty we have to sort the predictions. IDK whys
    for index,prediction in enumerate(predictions):
        total_predictions += 1
        if prediction.__contains__(ground_truth):
            correct_predictions += 1
            gap += correct_predictions / total_predictions
    gap /= num_positives
    print(gap)
    return gap