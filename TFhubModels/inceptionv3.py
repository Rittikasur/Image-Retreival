import sys
sys.path.append('D:/ORG India/Image-Retreival')
import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from milvus.src.Db import DBClass
from Benchmark.SaliencyMap import generate_saliency_matrix,plot_saliency_map
from scipy import spatial
from mAp.metrics import GlobalAveragePrecision as GAP
from Config.Config import Config
#https://tfhub.dev/google/collections/image/1
#https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5
config = Config("inception_v3")
model = hub.KerasLayer(config.MODEL_WEIGHTS,trainable=False)
dbmodel = DBClass()
dbmodel.open_connection()
dbmodel.loadOrCreate_collection(config.MILVUS_DBNAME,dimension=config.FEATURE_VECTOR_SIZE)
def populate_db(root_folder,list_of_files):
    for data_path in list_of_files:
        image = cv2.imread(os.path.join(root_folder,data_path))
        image = cv2.resize(image,config.INPUT_DIMS)
        image = np.expand_dims(image,axis=0)
        image_tensor = tf.convert_to_tensor(image, dtype=tf.int8)
        normalised_image = tf.image.convert_image_dtype(image_tensor, dtype=tf.float16, saturate=False)
        output = model(normalised_image)
        mr = dbmodel.insertData([data_path],[output[0].numpy().tolist()])
        print(mr)
    
def search_image(image_path):
    # dbmodel.loadOrCreate_collection(config.MILVUS_DBNAME,dimension=config.FEATURE_VECTOR_SIZE)
    image = cv2.imread(image_path)
    image = cv2.resize(image,config.INPUT_DIMS)
    image = np.expand_dims(image,axis=0)
    image_tensor = tf.convert_to_tensor(image, dtype=tf.int8)
    normalised_image = tf.image.convert_image_dtype(image_tensor, dtype=tf.float16, saturate=False)
    output = model(normalised_image)
    rs = dbmodel.searchEmbedding(output[0].numpy().tolist())
    resultlist = list(rs[0].ids)
    queryresult = dbmodel.querywithId(resultlist)
    sku_name_query = [i['sku_name'] for i in queryresult]
    return(sku_name_query) #This queryresult is a list
    # print(type(queryresult))

def extractfeature(np_image):
    image = cv2.resize(np_image,config.INPUT_DIMS)
    image = np.expand_dims(image,axis=0)
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
    result = 1 - spatial.distance.cosine(source, target)
    return(result)

def mean_AveragePrecision(data_root,N):
    totalgap = 0
    for image_nsme in os.listdir(data_root)[:N]:
        data_path = os.path.join(data_root,image_nsme)
        query_result = search_image(data_path)
        try:
            gap = GAP(query_result,image_nsme)
            totalgap += gap
            print(image_nsme)
            print(query_result)
            print(gap)
            print("----------------------------------------------------------------")
        except:
            print("Some Error found")
            print(image_nsme)
            print("----------------------------------------------------------------")

    print("The total gap is")
    print(totalgap)
    print("The total MAP is")
    print(totalgap/len(os.listdir(data_root)[:N]))
    print("Work done")
if __name__=="__main__":
    #Generating MEan Average Precision
    # data_path = "D:/ORG India/data/all_data"
    # mean_AveragePrecision(data_path,N=10)




    #Searching the DB
    # data_path = "D:/ORG India/data/indexed_data/1657513964760-chaminda products-chaminda#3.png"
    # query_result = search_image(data_path)
    # print(query_result)
    # a = GAP(query_result,"1657513964760-chaminda products-chaminda#3.png")

    # Populating the db
    # images_list = os.listdir("D:/ORG India/data/all_data")
    # populate_db("D:/ORG India/data/all_data",images_list)

    # Generating Saliency Graph in the output folders
    # data_path = "D:/ORG India/data/indexed_data/1657513964760-chaminda products-chaminda#3.png"
    # query = cv2.imread(data_path)
    # saliency_matrix = generate_saliency_matrix(query,extractfeature,create_source_embedding_from_cvimage,scoring_function,50,25,use_pil=False)
    # plot_saliency_map(saliency_matrix,data_path)

    dbmodel.releaseCollectionFromMemory()