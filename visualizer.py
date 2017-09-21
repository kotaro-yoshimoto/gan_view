import pickle
from PIL import Image
import os
from io import BytesIO
import math
import base64

import chainer
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer import Variable
from chainer.utils import type_check
from chainer import function

import chainer.functions as F
import chainer.links as L

import numpy as np

inputimagesize = 200 #入力画像（正方想定）の１軸当りの画素数

class Generator(chainer.Chain):
    def __init__(self):
        super(Generator, self).__init__(
            l0s=L.Linear(10, int(inputimagesize/4) * int(inputimagesize/4) * 64),
            dc1s=L.Deconvolution2D(64, 32, 4, stride=2, pad=1),
            dc2s=L.Deconvolution2D(32, 3, 4, stride=2, pad=1),
            bn0s=L.BatchNormalization(int(inputimagesize/4) * int(inputimagesize/4) * 64),
            bn1s=L.BatchNormalization(32),
        )

#    def __call__(self, z, test=False):
#        h = F.reshape(F.relu(self.bn0s(self.l0s(z), test=test)), (z.data.shape[0], 64, int(inputimagesize/4), int(inputimagesize/4)))
#        h = F.relu(self.bn1s(self.dc1s(h), test=test))
#        x = (self.dc2s(h))
#        return x
    def __call__(self, z, test=False):
        h = self.l0s(z)
        h = self.bn0s(h, test=test)
        h = F.relu(h)
        h = F.reshape(h, (z.data.shape[0], 64, int(inputimagesize/4), int(inputimagesize/4)))
        h = self.dc1s(h)
        h = self.bn1s(h, test=test)
        h = F.relu(h)
        x = self.dc2s(h)
        return x


def clip_img(x):
	return np.float32(-1 if x<-1 else (1 if x>1 else x))

gen = Generator()
gen.to_cpu()

def gen_image_b64(noise_list):
    model_file = 'model.h5'
    serializers.load_hdf5(model_file, gen)
    z = np.array(noise_list)
    noise = (z - 50) / float(50)
    z = Variable(noise.reshape((1, 10)).astype(np.float32))
    x = gen(z, test=True)
    x = x.data
    img_array = ((np.vectorize(clip_img)(x[0,:,:,:])+1)/2 * 255).transpose(1,2,0)
    image = Image.fromarray(np.uint8(img_array))
    output = BytesIO()
    image.save(output, format='PNG')
    return base64.b64encode(output.getvalue()).decode()
