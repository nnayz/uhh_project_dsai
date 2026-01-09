from torch import nn
from torch.nn import functional as F
import torch
import re
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ProtoNet(nn.Module):
    """
    ProtoNet model for few-shot classfication.
    """

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (num_samples, sequence_length, mel_bins) = x.shape
        x = x.view(-1, 1, sequence_length, mel_bins)
        x = self.encoder(x)
        x = nn.MaxPool2d(kernel_size=2)(x)

        return x.view(x.size(0), -1)


class BasicBlock(nn.Module):
    expansion = 1

    @staticmethod
    def conv3x3(
        in_planes: int, out_planes: int, stride: int = 1, with_bias: bool = False
    ) -> nn.Conv2d:
        """
        3x3 convolution with padding
        """

        return nn.Conv2d(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=with_bias,
        )

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        with_bias: bool = False,
        non_linearity: str = "leaky_relu",
        downsample: nn.Module | None = None,
        drop_rate: float = 0.0,
        drop_block: bool = False,
        block_size: int = 1,
    ):
        super().__init__()

        self.conv1 = self.conv3x3(
            in_planes=in_planes, out_planes=planes, stride=stride, with_bias=with_bias
        )

        self.bn1 = nn.BatchNorm2d(planes)

        if non_linearity == "leaky_relu":
            self.relu = nn.LeakyReLU(negative_slope=0.01)
        else:
            self.relu = nn.ReLU()

        self.conv2 = self.conv3x3(
            in_planes=planes, out_planes=planes, stride=1, with_bias=with_bias
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = self.conv3x3(
            in_planes=planes, out_planes=planes, stride=stride, with_bias=with_bias
        )
        self.bn3 = nn.BatchNorm2d(planes)

        self.maxpool = nn.MaxPool2d(stride=stride)
        self.downsample = downsample

        self.stride = stride
        self.drop_rate = drop_rate

        self.num_batches_tracked = 0
        self.drop_block = drop_block

        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        out = F.dropout(out, p=self.drop_rate, training=self.training, in_place=True)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        features: dict,
        linear_drop_rate: float = 0.5,
        block: nn.Module = BasicBlock,
        dropblock_size: int = 5,
        keep_prob: float = 1.0,
        drop_rate: float = 0.1,
        avg_pool: bool = True,
    ):
        super().__init__()

        drop_rate = features.drop_rate
        with_bias = features.with_bias

        self.inplanes = 1

        self.linear_drop_rate = linear_drop_rate
        self.layer1 = self._make_layer(
            block=block, planes=64, stride=2, features=features
        )

        self.layer2 = self._make_layer(
            block=block, planes=128, stride=2, features=features
        )
        self.layer3 = self._make_layer(
            block=block,
            planes=64,
            stride=2,
            drop_block=True,
            block_size=dropblock_size,
            features=features,
        )

        self.layer4 = self._make_layer(
            block=block,
            planes=64,
            stride=2,
            drop_block=True,
            block_size=dropblock_size,
            features=features,
        )

        self.keep_prob = keep_prob
        self.features = features

        self.keep_avg_pool = avg_pool

        self.drop_rate = drop_rate

        self.pool_avg = nn.AdaptiveAvgPool2d(
            (
                features.time_max_pool_dim,
                int(features.embedding_dim / (features.time_max_pool_dim * 64)),
            )
        )  # TODO: Try max pooling too

        self.pool_max = nn.AdaptiveMaxPool2d(
            (
                features.time_max_pool_dim,
                int(features.embedding_dim / (features.time_max_pool_dim * 64)),
            )
        )  # TODO: try max pooling

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity=features.non_linearity
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def load_weight(self, weight_file, device: torch.device):
        """
        utility to load a weight file to a device
        """

        state_dict = torch.load(weight_file, map_location=device)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Remove not needed prefixes from the keys of parameters
        weights = {}

        for k in state_dict:
            m = re.search(
                r"(^layer1\.|\.layer1\.|^layer2\.|\.layer2\.|^layer3\.|\.layer3\.|^layer4\.|\.layer4\.)",
                k,
            )

            if m is None:
                continue

            new_k = k[m.start() :]
            new_k = new_k[1:] if new_k[0] == "." else new_k
            weights[new_k] = state_dict[k]

            self.load_state_dict(weights)
            self.eval()

            logger.info(
                f"using audio embedding network pretrained weight: {Path(weight_file).name}"
            )
            return self

    def _make_layer(
        self,
        block: nn.Module,
        planes: int,
        stride: int = 1,
        drop_block: bool = False,
        block_size: int = 1,
        features: dict = None,
    ) -> nn.Sequential:

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.inplanes,
                    out_channels=planes * block.expansion,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                in_planes=self.inplanes,
                planes=planes,
                stride=stride,
                downsample=downsample,
                drop_rate=features.drop_rate,
                drop_block=drop_block,
                with_bias=features.with_bias,
                non_linearity=features.non_linearity,
                block_size=block_size,
            )
        )
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (num_samples, sequence_length, mel_bins) = x.shape
        x = x.view(-1, 1, sequence_length, mel_bins)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.features.layer_4:
            x = self.pool_avg(x)

        x = x.view(x.size(0), -1)
        return x


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from rich.progress import Progress

    def get_n_params(model: nn.Module) -> int:
        pp = 0
        with Progress() as progress:
            task = progress.add_task(
                "Calculating number of parameters", total=len(list(model.parameters()))
            )
            for p in list(model.parameters()):
                nn = 1
                for s in list(p.size()):
                    nn = nn * s
                pp += nn
                progress.advance(task)

        return pp

    cfg = OmegaConf.load("configs/train.yaml")

    model = ResNet(features=cfg.features)
    print(model)
    print("Calculating number of parameters...")
    print(get_n_params(model))

    input = torch.randn((3, 17, 128))
    print(model(input).size())
