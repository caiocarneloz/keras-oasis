import os
import numpy as np
import pandas as pd
from keras.preprocessing import image
from keras.models import  Model
from keras.applications.xception import Xception as xc
from keras.applications.xception import preprocess_input as xc_p
from keras.applications.vgg16 import VGG16 as vgg
from keras.applications.vgg16 import preprocess_input as vgg_p
from keras.applications.densenet import DenseNet121 as nn
from keras.applications.densenet  import preprocess_input as nn_p
from keras.applications.resnet50 import ResNet50 as rn
from keras.applications.resnet50 import preprocess_input as rn_p

from keras.layers import Flatten, Input


def getLabel(path):
    f = open(path, "r")

    line = f.readline()
    while('CDR' not in line):
        line = f.readline()

    try:
        value = float(line.split(' ')[10].split('\n')[0])
    except:
        return 'Control'

    if(value > 0):
        return 'Alzheimer'
    else:
        return 'Control'

def main():

    path = './oasis_images/'
    
    for pt_model, pp_i, m_name in zip([xc,vgg,rn,nn],[xc_p,vgg_p,rn_p,nn_p],['Xception','VGG16','ResNet50','NasNetLarge']):

        model = pt_model(weights='imagenet', include_top = False)
        
        model.summary()
    
        preprocess  = pp_i
        
        images = []
        labels = []
        for folder, subdirs, files in os.walk(path):
            for name in files:
                if '.gif' in name:
                    labels.append(getLabel(path+name.split('_mpr')[0]+'.txt'))
                    img_path = folder+'/'+name
                    img = image.load_img(img_path, target_size=(176, 208))
                    img_data = image.img_to_array(img)
                    img_data = np.expand_dims(img_data, axis=0)
                    img_data = preprocess(img_data)
                    images.append(img_data)
    
        images = np.vstack(images)
    
        input = Input(shape=(176, 208, 3), name='input')
        output = model(input)
        x = Flatten(name='flatten')(output)
        extractor = Model(inputs=input, outputs=x)
        features = extractor.predict(images)
    
        print(features.shape)
    
        df = pd.DataFrame.from_records(features)
        df['label'] = labels
        df = df.loc[:, (df != 0).any(axis=0)]
        df.columns = np.arange(0,len(df.columns))
        sufix = folder.split('/')
        df.to_csv(path+'./'+m_name+'_features_'+sufix[len(sufix)-2]+'.csv',sep=';',index=False)

if __name__ == "__main__":
    main()