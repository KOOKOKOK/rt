import onnx
import torch
import tensorrt as trt
from rt.utils import common
def export_to_onnx(model,onnx_file_path,h,w):

    model.eval()
    dummy_input = torch.empty((1, 3, h, w))
    # onnx_file_path = 'checkpoints/nanodet_m.onnx'
    torch.onnx.export(model, dummy_input, onnx_file_path, verbose=False, input_names=None,
                      output_names=None, opset_version=11)

    model = onnx.load(onnx_file_path)

    # Check that the IR is well formed
    onnx.checker.check_model(model)

    # Print a human readable representation of the graph
    print(onnx.helper.printable_graph(model.graph))  # print()


def export_to_trt(onnx_file_path,trt_file_path,h=320,w=320):
    # trt_file_path = 'checkpoints/nanodet_m.trt'
    TRT_LOGGER = trt.Logger()
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            common.EXPLICIT_BATCH) as network, trt.OnnxParser(
        network, TRT_LOGGER) as parser:
        # builder.max_workspace_size = 1 << 32  # 4096MB but you also can ingnore this step
        builder.max_batch_size = 1
        with open(onnx_file_path, 'rb') as model:  # 'deeplabV3Sim.onnx'  onnx_file_path
            print('Beginning ONNX file parsing')
            is_parser = parser.parse(model.read())
            print('parse onnx_model' + '\t:' + str(is_parser))
            for error in range(parser.num_errors):
                print(parser.get_error(error))

            network.get_input(0).shape = [1, 3, h, w]
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(trt_file_path, "wb") as f:
                f.write(engine.serialize())