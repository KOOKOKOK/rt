import time

import cv2

from rt.utils import common
from rt.utils.common import load_and_resize,shuffle_and_normalize
import torch
import tensorrt as trt

def inference(model,img_path):
    model.eval()
    img = cv2.imread(img_path)


    height, width = img.shape[:2]

    image_raw, image = load_and_resize(img_path, (320, 320))
    image = torch.tensor(shuffle_and_normalize(image))

    with torch.no_grad():
        start = time.time()
        results = model(image)
        end = time.time()
        print('Torch Infer time: %s Seconds' % (end - start))
    return results

def inference_trt(engine_file_path ,input_image_path,h=320,w=320):
    # engine_file_path = 'checkpoints/nanodet_m.trt'
    # input_image_path = 'testdata/bear1.jpg'
    TRT_LOGGER = trt.Logger()

    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine_trt = runtime.deserialize_cuda_engine(f.read())
        image_raw, image = load_and_resize(input_image_path, (h, w))
        image = shuffle_and_normalize(image)
        with engine_trt as engine, engine.create_execution_context() as context:
            inputs, outputs, bindings, stream = common.allocate_buffers(engine)
            # Do inference
            print('Running inference on image {}...'.format(input_image_path))
            inputs[0].host = image

            start = time.time()
            trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs,
                                                 stream=stream)
            end = time.time()
            print('RT Infer time: %s Seconds' % (end - start))
            return trt_outputs