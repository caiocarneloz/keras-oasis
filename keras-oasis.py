import os
import numpy as np
import pandas as pd
from keras.models import  Model
from keras.preprocessing import image
from keras.layers import Flatten, Input

from keras.applications.densenet import DenseNet121 as dense, preprocess_input as pp_dense
from keras.applications.vgg16 import VGG16 as vgg16, preprocess_input as pp_vgg16
from keras.applications.vgg19 import VGG19 as vgg19, preprocess_input as pp_vgg19
from keras.applications.xception import Xception as xception, preprocess_input as pp_xception
from keras.applications.resnet50 import ResNet50 as resnet, preprocess_input as pp_resnet
from keras.applications.nasnet import NASNetLarge as nasnet, preprocess_input as pp_nasnet
from keras.applications.mobilenet import MobileNet as mobilenet, preprocess_input as pp_mobilenet

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

    for pt_model, pp_i, m_name in zip([dense,vgg16,vgg19,xception,resnet,nasnet,mobilenet], \
                                      [pp_dense,pp_vgg16,pp_vgg19,pp_xception,pp_resnet,pp_nasnet,pp_mobilenet], \
                                      ['DenseNet','VGG16','VGG19','Xception','ResNet50','NASNetLarge','MobileNet']):

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