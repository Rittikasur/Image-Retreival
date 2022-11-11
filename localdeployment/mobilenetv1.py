
import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from milvus.src.Db import DBClass
from Config import Config
#https://tfhub.dev/google/collections/image/1
#https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/feature_vector/5

class Mobilenetv1:
    def __init__(self):
        self.config = Config("inception_v3")
        self.model = hub.KerasLayer(self.config.MODEL_WEIGHTS,trainable=False)
        self.dbmodel = DBClass()
        self.dbmodel.open_connection(host="milvus-standalone")
        self.dbmodel.loadOrCreate_collection(self.config.MILVUS_DBNAME,dimension=self.config.FEATURE_VECTOR_SIZE)
        
    def search_image(self,image_path):
        # self.dbmodel.loadOrCreate_collection(self.config.MILVUS_DBNAME,dimension=self.config.FEATURE_VECTOR_SIZE)
        image = cv2.imread(image_path)
        image = cv2.resize(image,self.config.INPUT_DIMS)
        image = np.expand_dims(image,axis=0)
        image_tensor = tf.convert_to_tensor(image, dtype=tf.int8)
        normalised_image = tf.image.convert_image_dtype(image_tensor, dtype=tf.float16, saturate=False)
        output = self.model(normalised_image)
        rs = self.dbmodel.searchEmbedding(output[0].numpy().tolist())
        resultlist = list(rs[0].ids)
        queryresult = self.dbmodel.querywithId(resultlist)
        sku_name_query = [i['sku_name'] for i in queryresult]
        return(sku_name_query) #This queryresult is a list
        # print(type(queryresult))
    def populate_db(self,root_folder,list_of_files):
        for data_path in list_of_files:
            image = cv2.imread(os.path.join(root_folder,data_path))
            image = cv2.resize(image,self.config.INPUT_DIMS)
            image = np.expand_dims(image,axis=0)
            image_tensor = tf.convert_to_tensor(image, dtype=tf.int8)
            normalised_image = tf.image.convert_image_dtype(image_tensor, dtype=tf.float16, saturate=False)
            output = self.model(normalised_image)
            mr = self.dbmodel.insertData([data_path],[output[0].numpy().tolist()])
            print(mr)
    def releasedbfrommemory(self):
        self.dbmodel.releaseCollectionFromMemory()
