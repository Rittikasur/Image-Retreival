import cv2
import numpy as np
import re
import tensorflow as tf
import matplotlib.pyplot as plt
import tqdm
import os
from PIL import Image
class MAC():
    def __init__(self,L):
        self.L = L
        self.lastconvfeature = None
        self.max_pool_2d = None

    def calculate_centers(self,w,h):
        wl = []
        hl = []
        s  = []
        centers = dict()
        for l in range(2,self.L+1):
            wi = (2 * w)/(l+1)
            hi = (2 * h)/(l+1)
            s_val = max(wi,hi)
            s.append(s_val)
            wl.append(wi)
            hl.append(hi)
            it_centers = []
            for i in range(1,l+1):
                for j in range(1,l+1):
                    bw =  (w-wi)/(l-1)
                    bh =  (h-hi)/(l-1)
                    cx =  (wi/2) + (i-1)*bw
                    cy =  (hi/2) + (j-1)*bh
                    it_centers.append([int(cx),int(cy)])
            centers[int(s_val)] = it_centers
        return centers

    def patches(self,side,center,image):
        h,w,c = image.shape
        half_side = side / 2
        left_corner_x = int(center[0] - half_side) 
        left_corner_y = int(center[1] - half_side)
        if(left_corner_x < 0):
            left_corner_x = 0
        if(left_corner_y < 0):
            left_corner_y = 0
        stridex = int(left_corner_x+side)
        stridey = int(left_corner_y+side)
        if(stridex>h):
            stridex = int(left_corner_x + side-(stridex-h))
        if(stridey>w):
            stridey = int(left_corner_y + side-(stridey-w))
        #print(left_corner_x,stridex,left_corner_y,stridey)
        image_patch = image[left_corner_x:stridex,left_corner_y:stridey,:]
        return(image_patch)

    def all_patches(self,centers,img):
        all_patches = []
        for sides,allcenters in centers.items():
            for single_centers in allcenters:
                all_patches.append(self.patches(sides,single_centers,img))
        assert len(all_patches)+1 == sum([i**2 for i in range(1,self.L+1)])
        return all_patches

    def show_patches(self,all_patches):
        for sap in all_patches:
            plt.imshow(sap)
            plt.show()

    def featuremodel(self):
        self.max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(8, 8),strides=(1, 1), padding='valid')
        model = tf.keras.applications.inception_v3.InceptionV3(
            include_top=True,
            weights='imagenet',
            pooling=None,
            classes=1000,
            classifier_activation='softmax'
        )
        convname = re.compile(r'conv2d_*')
        convlayerlist = []
        for layers in model.layers:
            mo1 = bool(re.match(convname,layers.name))
            if(mo1):
                convlayerlist.append(layers.name)
        self.lastconvfeature = tf.keras.Model(
            inputs = model.inputs,
            outputs= model.get_layer(convlayerlist[-1]).output)
        print("Feature Maps Generated")

    def extractfeature(self,ap):
        resized_ap = np.resize(ap,(1,299,299,3))
        lastconvoutput = self.lastconvfeature(resized_ap)
        finalfeature = self.max_pool_2d(lastconvoutput)
        finalfeature = np.squeeze(finalfeature)
        return finalfeature


    def create_source_embedding_from_cvimage(self,list_of_files):
        print('%d Cv Image Images found'%len(list_of_files))
        embeddingList=[]
        embedding_with_meta = []
        for eachFile in list_of_files:
            distlist = []
            referenceimage = cv2.imread(os.path.join("D:/ORG India/data/all_data/",eachFile))
            centers = self.calculate_centers(referenceimage.shape[0],referenceimage.shape[1])
            all_patches = self.all_patches(centers,referenceimage)
            extractedfeature = []
            for ap in all_patches:
                returnedfeature = self.extractfeature(ap)
                extractedfeature.append(returnedfeature)
            embedding_with_meta.append({"name":eachFile,"embedding":extractedfeature})
        print("Embedding Created from CV Image")
        return embedding_with_meta

    def scoring_function(self,source_embedding_with_meta,target_embedding):
        distlist = []
        minindex = 1000
        for ef in source_embedding_with_meta:
            dist = np.linalg.norm(ef["embedding"] - target_embedding)
            distlist.append(dist)
            if(min(distlist)<minindex):
                minindex = min(distlist)
                minindexpath = ef["name"]
        print(ef['name'])
        return minindex

if __name__=="__main__":
    query_data = "D:/Rohit/ORG India/images/husky.jpg"
    mac = MAC(4)
    mac.featuremodel()

    # The model for feature extraction
    query = cv2.imread(query_data) #D:\Rohit\ORG India\images\image_effect_HUc1.png
    queryImageFeature = mac.extractfeature(query)

    path = "D:/ORG India/data/all_data/"
    referenceimages = os.listdir(path)[:3]
    exfea = mac.create_source_embedding_from_cvimage(referenceimages)
    mac.scoring_function(exfea,queryImageFeature)

    