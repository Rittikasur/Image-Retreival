import sys
sys.path.append('D:/ORG India/Image-Retreival')
import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from milvus.src.Db import DBClass
#https://tfhub.dev/google/collections/image/1
#https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5

model = hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5",trainable=False)
dbmodel = DBClass()
dbmodel.open_connection()
def populate_db(list_of_files):
    dbmodel.loadOrCreate_collection("ORGIndia_IDKWTF",dimension=2048)
    for data_path in list_of_files:
        image = cv2.imread(os.path.join("D:/ORG India/data/indexed_data",data_path))
        image = cv2.resize(image,(299,299))
        image = np.expand_dims(image,axis=0)
        image_tensor = tf.convert_to_tensor(image, dtype=tf.int8)
        normalised_image = tf.image.convert_image_dtype(image_tensor, dtype=tf.float16, saturate=False)
        output = model(normalised_image)
        mr = dbmodel.insertData([data_path],[output[0].numpy().tolist()])
        print(mr)
    
def search_image(image_path):
    dbmodel.loadOrCreate_collection("ORGIndia_IDKWTF",dimension=2048)
    image = cv2.imread(image_path)
    image = cv2.resize(image,(299,299))
    image = np.expand_dims(image,axis=0)
    image_tensor = tf.convert_to_tensor(image, dtype=tf.int8)
    normalised_image = tf.image.convert_image_dtype(image_tensor, dtype=tf.float16, saturate=False)
    output = model(normalised_image)
    rs = dbmodel.searchEmbedding(output[0].numpy().tolist())
    print(rs)
if __name__=="__main__":
    data_path = "D:/ORG India/data/indexed_data/1657513964760-chaminda products-chaminda#3.png"
    search_image(data_path)
    #images_list = os.listdir("D:/ORG India/data/indexed_data")
    #populate_db(images_list[:3])