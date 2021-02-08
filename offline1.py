import glob
import os
import pickle
from PIL import Image
from feature_extractor import FeatureExtractor
import numpy as np

fe = FeatureExtractor()
features=[]
images=[]

import os
dir = r'/home/tusshar/furniture/static/img'
for files in os.listdir(dir):
    
    cur_dir = dir+"/"+files
    count = 0
    for file in os.listdir(cur_dir):
        
        img = Image.open(cur_dir+"/"+file)  # PIL(Python Imaging Library) image
        feature = fe.extract(img)
        #feature_path = 'static/feature/' + os.path.splitext(os.path.basename(img_path))[0] + '.pkl'
        features.append(feature)
        abs=cur_dir+"/"+file
        images.append(abs.split(r"furniture/")[1])
        print(abs.split(r"furniture/")[1])

np.save("features.npy",np.array(features))
np.save("images.npy",np.array(images))
