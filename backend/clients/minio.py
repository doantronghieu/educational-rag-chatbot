"""MinIO object storage client."""

from typing import BinaryIO


class StorageClient:
    """Storage client for file storage."""

    def __init__(self, endpoint: str, access_key: str, secret_key: str, bucket: str):
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket = bucket
        # TODO: Initialize minio client when implementing file upload

    async def upload_file(
        self, file_path: str, file_data: BinaryIO, content_type: str
    ) -> str:
        """Upload a file to storage."""
        # TODO: Implement file upload to MinIO
        return file_path

    async def download_file(self, file_path: str) -> bytes:
        """Download a file from storage."""
        # TODO: Implement file download from MinIO
        return b""
