import io
import os
import tempfile
from pathlib import Path
from typing import List

import minio
from PIL import Image


class ImageService:
    def __init__(self) -> None:
        self._minio_client = minio.Minio(
            endpoint=os.environ["SKULD_MINIO_ENDPOINT"],
            access_key=os.environ["SKULD_MINIO_ACCESS_KEY"],
            secret_key=os.environ["SKULD_MINIO_SECRET_KEY"],
            secure=False,
        )
        self._masks_bucket_name = os.environ["SKULD_UI_MINIO_MASKS_BUCKET_NAME"]
        self._results_bucket_name = os.environ["SKULD_UI_MINIO_RESULTS_BUCKET_NAME"]

    def list_mask_names(self) -> List[str]:
        return [x.object_name for x in self._minio_client.list_objects(bucket_name=self._masks_bucket_name)]

    def _get_image_bytes(self, bucket_name: str, object_name: str) -> bytes:
        with tempfile.TemporaryDirectory() as temp_path:
            file_path = Path(temp_path, "image")

            self._minio_client.fget_object(bucket_name=bucket_name, object_name=object_name, file_path=file_path)
            mask = Image.open(file_path)

        mask_bytes = io.BytesIO()
        mask.save(mask_bytes, format="PNG")
        return mask_bytes.getvalue()

    def get_mask_by_name(self, name: str) -> bytes:
        return self._get_image_bytes(bucket_name=self._masks_bucket_name, object_name=name)

    def get_result(self, mask_name: str, angle_of_attack: float) -> bytes:
        angle_of_attack = round(angle_of_attack % 360)
        return self._get_image_bytes(bucket_name=self._results_bucket_name, object_name=f"{mask_name}-{angle_of_attack:03d}")
