
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt
import PIL
import cv2
import time
import os

def get_image(path,show=False):
    im = Image.open(path)
    if show == True:
        im.show()
    image = np.array(im)
    return image


def generate_occlusion(image,center, filter_size=3):
    x,y=center
    startX = x - math.floor(filter_size/2)
    endX = x + math.floor(filter_size/2) 
    startY = y - math.floor(filter_size/2)
    endY = y + math.floor(filter_size/2) 
    new_img = image.copy()
    black_patch = np.zeros((filter_size,filter_size))
    new_img[startX:endX,startY:endY,:] = 0
    occluded_image = Image.fromarray(new_img)
    return occluded_image



def generated_occluded_images(image,filter_size=3,stride=1):
    occluded_images = []
    h,w,c = image.shape
#     print(image.shape)
    row_start = (math.floor(filter_size/2) + 1)%h
    row_end = (h - math.floor(filter_size/2) - 1)%h
    col_start = (math.floor(filter_size/2) + 1)%w
    col_end = (w - math.floor(filter_size/2) - 1)%w
#     print(row_start,row_end,col_start,col_end)
    print((row_end-row_start) * (col_end-col_start)/stride)
    for row_center in range(row_start,row_end,stride):
        for col_center in range(col_start,col_end,stride):
            print((row_center,col_center))
            occluded_image = generate_occlusion(image,(row_center,col_center),filter_size)
            meta_info = {
                "startX":(row_center - math.floor(filter_size/2)),
                "endX" : row_center + math.floor(filter_size/2) ,
                "startY" : col_center - math.floor(filter_size/2),
                "endY" : col_center + math.floor(filter_size/2) 
            }
            occluded_images.append({"meta":meta_info,"image":occluded_image})
    print(len(occluded_images))
    return occluded_images

def plot_saliency_map(saliency_matrix,image_path,image_shape=(256,256)):
    if(os.path.exists("./tmp/") != True):
        os.makedirs("./tmp")
    if(os.path.exists("./output/") != True):
        os.makedirs("./output")
    heatmap_file_path = "./tmp/"+time.strftime("%d-%H-%M-%S") + "-heatmap_image.png"
    imgf = plt.imshow(saliency_matrix, cmap='hot', interpolation='nearest')
    imgf.set_cmap('hot')
    plt.axis('off')
    plt.savefig(heatmap_file_path, bbox_inches='tight')
    # If the upper block does not work then this code is fine
    # im = Image.fromarray(np.uint8(saliency_matrix))
    # im.save("pilout.png")
    plotimg = cv2.imread(image_path)
    plotimg = cv2.resize(plotimg,image_shape)
    heatmapplot = cv2.imread(heatmap_file_path)
    heatmapplot = cv2.resize(heatmapplot,image_shape)
    fig = plt.figure(figsize=(20,20))
    ax2 = plt.subplot(1, 4, 3, aspect='equal')
    ax2.imshow(plotimg, alpha = 1)
    ax2.imshow(heatmapplot, alpha = 0.5)
    saliency_map_file_path = "./output/"+time.strftime("%d-%H-%M-%S") + "-salency_map.png"
    plt.savefig(saliency_map_file_path,bbox_inches='tight')

def generate_saliency_matrix(image,model,create_embedding,scoring_function,filter_size=100,stride=50):
    assert(type(image) == PIL.Image.Image)
    h,w = image.size
    image = np.array(image)
    occluded_images_with_meta = generated_occluded_images(image,filter_size,stride)
    # print(occluded_images_with_meta[0])
    occluded_images = list(map(lambda x:x["image"],occluded_images_with_meta))
    occluded_images_meta_list = list(map(lambda x : x["meta"],occluded_images_with_meta))
    targetEmbedding = model(image)
    SourceEmbeddingList = create_embedding(occluded_images)
    saliency_matrix = np.zeros((h,w))
    print(len(SourceEmbeddingList))
    print(len(occluded_images_meta_list))
    print(occluded_images_meta_list[0])
    assert(len(SourceEmbeddingList) == len(occluded_images_meta_list) )
    for i in range(len(SourceEmbeddingList)):
        embeddings = SourceEmbeddingList[i]
        startX = occluded_images_meta_list[i]["startX"]
        endX = occluded_images_meta_list[i]["endX"]
        startY = occluded_images_meta_list[i]["startY"]
        endY = occluded_images_meta_list[i]["endY"]
        try:
            inliers = scoring_function(targetEmbedding,embeddings)
            if(inliers > 255):
                inliers = 255
            saliency_matrix[startX:endX,startY:endY] = inliers
        except:
            print("Exception has occured")
    return(saliency_matrix)
