from torchvision.models.shufflenetv2 import ShuffleNetV2
import torch
class ExShuffleNetV2(ShuffleNetV2):
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        return x
def _shufflenetv2(arch, pretrained, progress, *args, **kwargs):
    model = ExShuffleNetV2(*args, **kwargs)

    if pretrained:
        pass

    return model

def shufflenet_v2_x0_5(pretrained=False, progress=True, **kwargs):
    """
    Constructs a ShuffleNetV2 with 0.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x0.5', pretrained, progress,
                         [4, 8, 4], [24, 48, 96, 192, 1024], **kwargs)

if __name__ == '__main__':
    model = shufflenet_v2_x0_5()
    input  = torch.empty((1,3,640,640))
    print(model(input))