from torchvision.models.resnet import Bottleneck
from torchvision.models.resnet import ResNet

class ExRes(ResNet):

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ExRes(block, layers, **kwargs)
    if pretrained:
        pass
    return model
def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

if __name__ == '__main__':

    model = resnet50()
    model.eval()

