import os
# from absl import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from scipy.spatial import cKDTree
from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform
from six import BytesIO
import PIL
import tensorflow as tf

import tensorflow_hub as hub

# from Benchmark.SaliencyMap import generate_saliency_matrix,plot_saliency_map
# from six.moves.urllib.request import urlopen
# embeddingList = []
# tf.logging.set_verbosity(tf.logging.ERROR)
delf = hub.load('https://tfhub.dev/google/delf/1').signatures['default']

def download_and_resize(path, new_width=256, new_height=256):
  # path = tf.keras.utils.get_file(url.split('/')[-1], url)
  image = Image.open(path)
  image = ImageOps.fit(image, (new_width, new_height), Image.ANTIALIAS)
  return image

def run_delf(image):
  np_image = np.array(image)
  float_image = tf.image.convert_image_dtype(np_image, tf.float32)

  return delf(
      image=float_image,
      score_threshold=tf.constant(100.0),
      image_scales=tf.constant([0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0]),
      max_feature_num=tf.constant(1000))

#@title TensorFlow is not needed for this post-processing and visualization
def match_images(result1, result2):
  distance_threshold = 0.8

  # Read features.
  num_features_1 = result1['locations'].shape[0]
  print("Loaded image 1's %d features" % num_features_1)
  
  num_features_2 = result2['locations'].shape[0]
  print("Loaded image 2's %d features" % num_features_2)

  # Find nearest-neighbor matches using a KD tree.
  d1_tree = cKDTree(result1['descriptors'])
  _, indices = d1_tree.query(
      result2['descriptors'],
      distance_upper_bound=distance_threshold)

  # Select feature locations for putative matches.
  locations_2_to_use = np.array([
      result2['locations'][i,]
      for i in range(num_features_2)
      if indices[i] != num_features_1
  ])
  locations_1_to_use = np.array([
      result1['locations'][indices[i],]
      for i in range(num_features_2)
      if indices[i] != num_features_1
  ])

  print("location for use")
  print(locations_1_to_use.shape,locations_2_to_use.shape)
  # Perform geometric verification using RANSAC.
  _, inliers = ransac(
      (locations_1_to_use, locations_2_to_use),
      AffineTransform,
      min_samples=3,
      residual_threshold=20,
      max_trials=1000)
  if(inliers is not None):
    print('Found %d inliers' % sum(inliers))
    return(sum(inliers))

def create_source_embedding_from_pil(list_of_files):
  print('%d PIL Images found'%len(list_of_files))
  embeddingList=[]
  for eachFile in tqdm(list_of_files,total=len(list_of_files)):
    imageEmbedding = run_delf(eachFile)
    embeddingdata = imageEmbedding
    embeddingList.append(embeddingdata)
  print("Embedding Created from PIL")
  return embeddingList



def create_source_embedding_from_files(list_of_files):
  print('%d files found'%len(list_of_files))
  embeddingList=[]
  for eachFile in tqdm(list_of_files,total=len(list_of_files)):
    filepath = "D:/ORG India/data/all_data/" + eachFile#os.path.abspath(eachFile)
    imagefile = download_and_resize(filepath)
    imageEmbedding = run_delf(imagefile)
    embeddingdata = {"name":eachFile,"embedding":imageEmbedding}
    embeddingList.append(embeddingdata)
  print("Embedding Created")
  return embeddingList

def find_matching_image(embeddingList,targetembedding):
  inlinerList = []
  for embeddingData in embeddingList:
    print("For Target",embeddingData["name"])
    try:
      inliers = match_images(targetembedding,embeddingData["embedding"])
      if(inliers is not None):
        inlinerList.append(inliers)
      else:
        inlinerList.append(0)
    except:
      print("Exception has occured")
      inlinerList.append(0)
  indexOfMaxInlier = inlinerList.index(max(inlinerList))
  print("Matching Image is ",embeddingList[indexOfMaxInlier]["name"])



# if __name__ == "__main__":
#   print("Creating Embeddings")
#   all_data = "../../data/all_data"
#   data_path = "D:/ORG India/data/all_data/30311-ceylon biscuits limited-munchee chocolate cream#1.png"
#   target = download_and_resize(data_path)
#   targetEmbedding = run_delf(target)
#   sourceEmbeddding = create_source_embedding_from_files(os.listdir(all_data)[:1])
#   #find_matching_image(sourceEmbeddding,targetEmbedding)
#   print(type(target) == PIL.Image.Image)
#   saliency_matrix = generate_saliency_matrix(target,run_delf,create_source_embedding_from_pil,match_images,50,25)
#   plot_saliency_map(saliency_matrix,data_path)

