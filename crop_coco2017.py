import json
from PIL import Image
from io import BytesIO
from io import BytesIO as Bytes2Data
import cv2
import numpy as np
import base64

from PIL import Image
import cv2
import time
import numpy as np
import imageio
import matplotlib.pyplot as plt

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


with open("/home/rszj-ai/hdd/chenrui/task/instances_train2017.json", 'r') as load_f:
    load_dict = json.load(load_f)
    print(len(load_dict['annotations']))
    for i in range(len(load_dict['annotations'])):
        iz = load_dict['annotations'][i]
        ic = iz['image_id']
        iq = iz['category_id']
        il = iz['id']
        # print(ic)
        ie = len(str(ic))
        # print(ie)
        img_path = '/home/rszj-ai/hdd/chason/datasets/coco2017/train2017/' + '0' * (12 - ie) + str(ic) + '.jpg'
        #print(img_path)

        ori_image = cv2.imread(img_path)
        ib = ori_image.shape
        ia1 = ib[0]
        ia2 = ib[1]
        # print(ia1, ia2)

        box = iz['bbox']
        # print(box)
        x1 = box[0]
        y1 = box[1]
        delta_x = box[2]
        delta_y = box[3]
        a = int(x1)
        b = int(delta_x)
        c = int(y1)
        d = int(delta_y)
        print(a, b, c, d)
        res = ori_image[c:(c + d), a:(a + b)]
        cv2.imwrite('/home/rszj-ai/hdd/chason/datasets/coco2017/train2017_crop/' + str(iq) + '/' +  str(il) + '.jpg', res)
        print(str(i) + ' is cropped!\n')
