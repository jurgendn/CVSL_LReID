from collections import OrderedDict
from typing import Tuple

import torch
from dynaconf.base import DynaBox
from torch import Tensor, nn


def make_layers(block, no_relu_layers):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        else:
            conv2d = nn.Conv2d(in_channels=v[0],
                               out_channels=v[1],
                               kernel_size=v[2],
                               stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if layer_name not in no_relu_layers:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))

    return nn.Sequential(OrderedDict(layers))


class BodyPose(nn.Module):

    def __init__(self, cfg: DynaBox):
        super(BodyPose, self).__init__()
        self.CFG = cfg

        # Stage 1
        self.model0 = make_layers(block=self.CFG.STAGE_1.BLOCK_0,
                                  no_relu_layers=self.CFG.NO_RELU_LAYERS)
        self.model1_1 = make_layers(block=self.CFG.STAGE_1.BLOCK_1,
                                    no_relu_layers=self.CFG.NO_RELU_LAYERS)
        self.model1_2 = make_layers(block=self.CFG.STAGE_1.BLOCK_2,
                                    no_relu_layers=self.CFG.NO_RELU_LAYERS)
        # Stages 2 - 6
        for block_idx in range(2, 7):
            module = make_layers(block=self.CFG.STAGE_2.BLOCK_1,
                                 no_relu_layers=self.CFG.NO_RELU_LAYERS)
            setattr(self, f"model{block_idx}_1", module)
            module = make_layers(block=self.CFG.STAGE_2.BLOCK_2,
                                 no_relu_layers=self.CFG.NO_RELU_LAYERS)
            setattr(self, f"model{block_idx}_2", module)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:

        out1 = self.model0(x)

        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)
        out2 = torch.cat([out1_1, out1_2, out1], 1)

        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3 = torch.cat([out2_1, out2_2, out1], 1)

        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        out4 = torch.cat([out3_1, out3_2, out1], 1)

        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)
        out5 = torch.cat([out4_1, out4_2, out1], 1)

        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)
        out6 = torch.cat([out5_1, out5_2, out1], 1)

        out6_1 = self.model6_1(out6)
        out6_2 = self.model6_2(out6)

        return out6_1, out6_2
