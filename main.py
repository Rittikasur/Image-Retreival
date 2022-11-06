import os
import cv2
from Mac.mac import MAC
from delf.delf import run_delf,match_images,download_and_resize
from milvus.src.Db import DBClass




def getDelfEmbedding(data_path):
    target = download_and_resize(data_path)
    targetEmbedding = run_delf(target)
    match = match_images(targetEmbedding,targetEmbedding)
    return targetEmbedding


def populate_db(indexed_path,getEmbeddingFunction):
    # dbmodel = DBClass()
    # dbmodel.open_connection()
    # dbmodel.loadOrCreate_collection("ORGIndiaDelf")
    indexed_data = os.listdir(indexed_path)[:10]
    for file in indexed_data:
        file_path = os.path.join(indexed_path,file)
        embedding = getEmbeddingFunction(file_path)
        # print(embedding.keys())
        # for key in embedding.keys():
        #     print(embedding[key].shape,end =" ")



if __name__=="__main__":
    data_path = "D:/ORG India/data/indexed_data"
    populate_db(data_path,getDelfEmbedding)
    
    #Single Delf
    # target = download_and_resize(data_path)
    # targetEmbedding = run_delf(target)
    # match = match_images(targetEmbedding,targetEmbedding)
    # print(match)

    # #Single MAC
    # mac = MAC(4)
    # mac.featuremodel()
    # query = cv2.imread(data_path) #D:\Rohit\ORG India\images\image_effect_HUc1.png
    # query = cv2.resize(query, (256, 256))
    # queryImageFeature = mac.extractfeature(query)
    # exfea = mac.create_source_embedding_from_cvimage([query])
    # for i in range(len(exfea)):
    #     embeddings = exfea[i]
    #     score = mac.scoring_function(queryImageFeature,embeddings)
    #     print("for index  the score is ",score)