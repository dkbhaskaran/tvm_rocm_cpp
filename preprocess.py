import os
import os.path
import argparse

import numpy as np
from PIL import Image

import tvm
from tvm import relay
from tvm.contrib.download import download
from tvm.contrib import graph_executor


import onnx
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("preprocess")

batchsize = 128

def save_model(opts):
    datatype = 'float32'
    logging.info(f'Loading model {opts.model}')
    onnx_model = onnx.load(opts.model)
    input_shape = {batchsize, 3, 224, 224}
    shape_dict = {'data': (batchsize, 3, 224, 224)}
    
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, datatype)
    with tvm.transform.PassContext(opt_level=3):
        factory = relay.build(mod, target='rocm -libs=miopen,rocblas', params=params)

    factory.get_lib().export_library('lib.so')
    log.info(f'Saved lib to lib.so')
    
    with open('graph.json', 'w') as f_graph_json:
        f_graph_json.write(factory.get_graph_json())    
        log.info(f'Saved graph to graph.json')

    with open('mod.params', 'wb') as f_params:
        f_params.write(relay.save_param_dict(factory.get_params()))
        log.info(f'Saved params to mod.params')

def preprocess_image(img_path, output_path):
    resized_image = Image.open(img_path).resize((224, 224))
    img_data = np.asarray(resized_image).astype("float32")

    # ONNX expects NCHW input, so convert the array
    img_data = np.transpose(img_data, (2, 0, 1))
    
    # Normalize according to ImageNet
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_stddev = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype("float32")
    for i in range(img_data.shape[0]):
          norm_img_data[i, :, :] = (img_data[i, :, :] / 255 - imagenet_mean[i]) / imagenet_stddev[i]
    
    # Add batch dimension
    img_data = np.expand_dims(norm_img_data, axis=0)
    
    with open(output_path, 'wb') as f:
        f.write(img_data)

def verify_saved_model():
    log.info('verifying saved model')
    with open('snake.bin', 'rb') as fp:
        img_data = np.fromfile(fp, dtype='float32').reshape(1, 3, 224, 224)

    input_name = 'data'
    target='rocm -libs=miopen,rocblas'

    mlib = tvm.runtime.load_module("lib.so")
    with open('graph.json') as f:
        graph = f.read()
    with open('mod.params', mode='rb') as fp:
        params = fp.read()

    dev = tvm.device(str(target))
    module = graph_executor.create(graph, mlib, dev)
    module.load_params(params)

    img = np.vstack([img_data]*batchsize)
    module.set_input(input_name, img)

    module.run()
    tvm_output = module.get_output(0).numpy()
    results0 = np.argmax(tvm_output[0], axis = 0)
    results1 = np.argmax(tvm_output[127], axis = 0)
    if results0 == 65 and results1 == 65:
        log.info('saved model verification : success')
    else:
        log.info('saved model verification : failed')

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="./resnet50-v2-7.onnx")
    opts = parser.parse_args()

    if not os.path.isfile('./resnet50-v2-7.onnx'):
        log.info(f'Downloading model')
        model_url = 'https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v2-7.onnx'
        download(model_url, "resnet50-v2-7.onnx")
        log.info(f'Downloading model complete')

    img1_path = './imagenet_cat.png'
    if not os.path.isfile(img1_path):
        img_url = 'https://s3.amazonaws.com/model-server/inputs/kitten.jpg'
        download(img_url, './imagenet_cat.png')
    preprocess_image(img1_path, 'cat.bin')

    img2_path = './imagenet_snake.jpeg'
    if not os.path.isfile(img2_path):
        img_url = 'https://user-images.githubusercontent.com/19551872/163961172-87bc3b70-84ea-40e0-962d-be1a9d58d83d.JPEG'
        download(img_url, './imagenet_snake.jpeg')
    preprocess_image(img2_path, 'snake.bin')
    
    save_model(opts)
    verify_saved_model()
