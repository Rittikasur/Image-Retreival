import sys
sys.path.append('D:/ORG India/Image-Retreival')
import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from milvus.src.Db import DBClass
# from Benchmark.SaliencyMap import generate_saliency_matrix,plot_saliency_map
from scipy import spatial
from mAp.metrics import GlobalAveragePrecision as GAP
#https://tfhub.dev/google/collections/image/1
#https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5

model = hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5",trainable=False)
dbmodel = DBClass()
dbmodel.open_connection()
def populate_db(root_folder,list_of_files):
    dbmodel.loadOrCreate_collection("ORGIndia_TFM1",dimension=2048)
    for data_path in list_of_files:
        image = cv2.imread(os.path.join(root_folder,data_path))
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
    resultlist = list(rs[0].ids)
    queryresult = dbmodel.querywithId(resultlist)
    return(queryresult) #This queryresult is a list
    # print(type(queryresult))

def extractfeature(np_image):
    image = np.expand_dims(np_image,axis=0)
    image_tensor = tf.convert_to_tensor(image, dtype=tf.int8)
    normalised_image = tf.image.convert_image_dtype(image_tensor, dtype=tf.float16, saturate=False)
    output = model(normalised_image)
    return(output[0].numpy().tolist())

def create_source_embedding_from_cvimage(list_of_image_array):
    list_of_embeddings = []
    for img in list_of_image_array:
        embedding = extractfeature(img)
        list_of_embeddings.append(embedding)
    return(list_of_embeddings)

def scoring_function(source,target):
    result = spatial.distance.cosine(source, target)
    return(result)

if __name__=="__main__":
    #Searching the DB
    data_path = "D:/ORG India/data/indexed_data/1657513964760-chaminda products-chaminda#3.png"
    # query_result = search_image(data_path)
    GAP(["a","b","a"],"a")
    # Populating the db
    # images_list = os.listdir("D:/ORG India/data/all_data")
    # populate_db("D:/ORG India/data/all_data",images_list)

    # Generating Saliency Graph in the output folders
    # query = cv2.imread(data_path)
    # query = cv2.resize(query,(299,299))
    # saliency_matrix = generate_saliency_matrix(query,extractfeature,create_source_embedding_from_cvimage,scoring_function,50,25,use_pil=False)
    # plot_saliency_map(saliency_matrix,data_path)