from rt.export_tools.export import *
from rt.runtime_tools.inference import *
import torch.nn as nn
from rt.exchange_layer_tools.ex_resnet import *
from torchvision.models.inception import inception_v3
from torchvision.models.vgg import vgg16
from rt.exchange_layer_tools.ex_shufflenetv2 import shufflenet_v2_x0_5
from rt.exchange_layer_tools.SSD import SSD300
if __name__ == '__main__':
    h,w = 640,640
    torch_model = SSD300()


    onnx_file_path = 'onnx_folder/SSD300.onnx'
    trt_file_path = 'trt_folder/SSD300.trt'
    img_path = 'resource/bear1.jpg'
    export_to_onnx(torch_model,onnx_file_path,h,w)
    export_to_trt(onnx_file_path,trt_file_path,h,w)

    inference(model=torch_model,img_path=img_path)
    inference_trt(engine_file_path=trt_file_path,input_image_path=img_path)


