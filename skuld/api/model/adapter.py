import io
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import minio
import numpy as np
import torch
from PIL import Image
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None, residual: bool = False) -> None:
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=mid_channels),
            nn.GELU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
        )
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual:
            return nn.functional.gelu(x + self.double_conv(x))
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels=in_channels, out_channels=in_channels, residual=True),
            DoubleConv(in_channels=in_channels, out_channels=out_channels),
        )
        self.embedding = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, out_channels))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.max_pool_conv(x)
        emb = self.embedding(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True), DoubleConv(in_channels, out_channels, in_channels // 2)
        )
        self.embedding = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, out_channels))

    def forward(self, x: torch.Tensor, skip_x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.embedding(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


def positional_encoding(t: torch.Tensor, embed_dim: int) -> torch.Tensor:
    t = t.reshape(-1, 1)
    inverse_frequency = 1.0 / 10_000 ** torch.linspace(0, 1, embed_dim // 2, device=t.device)
    pos_enc_a = torch.sin(t.repeat(1, embed_dim // 2) * inverse_frequency)
    pos_enc_b = torch.cos(t.repeat(1, embed_dim // 2) * inverse_frequency)
    return torch.cat([pos_enc_a, pos_enc_b], dim=-1)


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.input_convolution = DoubleConv(in_channels=in_channels, out_channels=64)

        self.down1 = Down(in_channels=64, out_channels=128, embed_dim=embed_dim)
        self.down2 = Down(in_channels=128, out_channels=256, embed_dim=embed_dim)
        self.down3 = Down(in_channels=256, out_channels=256, embed_dim=embed_dim)

        self.bottleneck1 = DoubleConv(256, 512)
        self.bottleneck2 = DoubleConv(512, 512)
        self.bottleneck3 = DoubleConv(512, 256)

        self.up1 = Up(in_channels=512, out_channels=128, embed_dim=embed_dim)
        self.up2 = Up(in_channels=256, out_channels=64, embed_dim=embed_dim)
        self.up3 = Up(in_channels=128, out_channels=64, embed_dim=embed_dim)

        self.output_convolution = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)

    def forward(self, images: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        conditions = positional_encoding(conditions, self.embed_dim)

        x1 = self.input_convolution(images)
        x2 = self.down1(x1, conditions)
        x3 = self.down2(x2, conditions)
        x = self.down3(x3, conditions)
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.up1(x, x3, conditions)
        x = self.up2(x, x2, conditions)
        x = self.up3(x, x1, conditions)
        return self.output_convolution(x)


class ModelAdapter:
    def __init__(self) -> None:
        self._minio_client = minio.Minio(
            endpoint=os.environ["SKULD_MINIO_ENDPOINT"],
            access_key=os.environ["SKULD_MINIO_ACCESS_KEY"],
            secret_key=os.environ["SKULD_MINIO_SECRET_KEY"],
            secure=False,
        )
        self._model = self._load_model()

    def _load_model(self) -> UNet:
        with tempfile.TemporaryDirectory() as temp_path:
            state_dict_path = Path(temp_path, "model.pth")
            self._minio_client.fget_object(
                bucket_name=os.environ["SKULD_API_MINIO_BUCKET_NAME"],
                object_name=os.environ["SKULD_API_MINIO_MODEL_NAME"],
                file_path=state_dict_path,
            )
            state_dict = torch.load(state_dict_path, map_location=torch.device("cpu"))

        model = UNet(in_channels=1, out_channels=1, embed_dim=256)
        model.load_state_dict(state_dict["model"])
        return model

    @staticmethod
    def _prepare_sample(image: bytes, angle_of_attack: float) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(io.BytesIO(image)).resize((256, 256))
        image = np.array(image, dtype=float).reshape((1, 1, 256, 256))
        image = torch.Tensor(image / 255)
        angle_of_attack = torch.Tensor([angle_of_attack % 360 / 360])
        return image, angle_of_attack

    @staticmethod
    def _tensor_to_png_bytes(tensor: torch.Tensor) -> bytes:
        tensor = (tensor / tensor.max() * 255)[0, 0, :, :].numpy().round().astype(np.uint8)
        image = Image.fromarray(tensor)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        return image_bytes.getvalue()

    def __call__(self, image: bytes, angle_of_attack: float) -> bytes:
        image, angle_of_attack = self._prepare_sample(image=image, angle_of_attack=angle_of_attack)

        with torch.no_grad():
            predict = self._model(image, angle_of_attack)

        return self._tensor_to_png_bytes(predict)
