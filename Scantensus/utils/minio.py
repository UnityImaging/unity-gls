from dataclasses import dataclass
from pathlib import Path
import tempfile

from minio import Minio
import io 
import os

from typing import Union

import tomlkit
import logging

import pydicom

@dataclass
class MinioConfig:
    """
    Represents standard MinIO configuration informatino.
    """
    
    endpoint: str
    access_key: str
    secret_key: str

    @staticmethod
    def from_secrets(path: Path = Path('infra_secrets/minio.toml')) -> 'MinioConfig':
        """
        Builds a MinIO configuration from the secrets folder.

        :param path: The path to the MinIO config.
        """

        doc = tomlkit.parse(path.read_text(encoding='utf-8'))

        return MinioConfig(
            endpoint=doc['MINIO_ENDPOINT'],
            access_key=doc['MINIO_ACCESS_KEY'],
            secret_key=doc['MINIO_SECRET']
        )


    def build_client(self) -> Minio:
        """
        Builds a MinIO client from the given configuration.
        """
        return Minio(endpoint=self.endpoint, access_key=self.access_key, secret_key=self.secret_key)
    

    def write(self, data: Union[str, bytes, io.BytesIO], bucket: str, file: str, content_type: str = "application/octet-stream"):
        """
        Writes the given data to the given file in MinIO.

        :param data: The data to write. Can be a string, bytes, or io.BytesIO.
        :param file: The file to write to.
        :param bucket: The bucket to write to.
        :param content_type: The content type of the data.
        """

        client = self.build_client()
        
        data_io = io.BytesIO()

        if type(data) is str:
            data_io.write(data.encode())
        elif type(data) is bytes:
            data_io.write(data)
        elif type(data) is io.BytesIO:
            data_io = data
            data_io.seek(0, 2)
        else:
            raise Exception("Data is not str / bytes / io.BytesIO")

        data_io_len = data_io.tell()
        data_io.seek(0)

        client.put_object(
            bucket_name=bucket,
            object_name=file,
            data=data_io,
            length=data_io_len,
            content_type=content_type
        )

    def read(self, bucket: str, file: str) -> bytes:
        """
        Reads the given file from MinIO.

        :param file: The file to read.
        :param bucket: The bucket to read from.

        :returns: The bytes data read from MinIO.
        """

        client = self.build_client()

        resp = client.get_object(bucket_name=bucket, object_name=file)

        if resp.status != 200:
            raise Exception(f"MinIO returned status {resp.status}")

        return resp.data


    def read_dicom(self, bucket: str, file: str) -> pydicom.FileDataset:
        """
        Reads the given DICOM file from MinIO.

        :param file: The file to read.
        :param bucket: The bucket to read from.

        :returns: The DICOM data.
        """

        data = self.read(bucket=bucket, file=file)
        
        return pydicom.read_file(io.BytesIO(data))


    def clone_directory(self, bucket: str, directory_path: str, output_path: Path | None = None) -> Path:
        """
        Clones the given directory from MinIO to a local directory.

        :param bucket: The bucket to read from.
        :param directory_path: The directory to read from.
        :param output_path: The output path to write to. If None, a temporary directory will be created.

        :returns: The local directory path.
        """

        output_path = output_path or Path(tempfile.mkdtemp(prefix='echocardium-minio-'))

        client = self.build_client()
        objects = client.list_objects(bucket_name=bucket, prefix=directory_path, recursive=True)

        for obj in objects:
            local_path: Path = output_path / obj.object_name
    
            if obj.is_dir:
                continue

            local_path.parent.mkdir(parents=True, exist_ok=True)
            client.fget_object(bucket_name=bucket, object_name=obj.object_name, file_path=str(local_path))

            logging.info(f"Downloaded {obj.object_name} to {local_path}")

        return output_path
