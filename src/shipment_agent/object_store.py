"""Object storage for shipment documents — the ``ObjectStore`` port.

Production document handling is object storage, not local files: a
shipment's documents are objects in an S3-compatible bucket (AWS
S3, or MinIO in the compose stack), and the shipment payload
references them by **object key** (``DocumentInput.object_key``).

Two directions, both in the service layer:

- **Inbound** — an intake adapter (email ingestion, a TMS webhook)
  drops the raw document into the bucket first and submits only the
  key; the service fetches the bytes through this port and decodes
  them as the document's text before the pipeline runs.
- **Archive** — when a shipment arrives with inline text and a
  store is configured, the service writes each document to the
  bucket under ``shipments/<id>/documents/<doc>.txt`` and records
  the key on the stored shipment, so the record of the decision
  points at durable objects, not at a request body that is gone.

Implementations: :class:`S3ObjectStore` (boto3, any S3-compatible
endpoint via ``S3_ENDPOINT_URL``) and :class:`InMemoryObjectStore`,
the test double. Configuration is env-only: ``S3_BUCKET`` selects
the whole feature (unset → no object store, documents travel inline
exactly as before), ``S3_ENDPOINT_URL`` / ``S3_REGION`` shape the
client, and credentials come from the standard AWS chain
(``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` or the runtime's
role) — MinIO's compose credentials ride the same variables.
"""

from __future__ import annotations

from .config import env_str, load_dotenv


class InMemoryObjectStore:
    """Dict-backed object store (tests). Keys → bytes, nothing else."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put(self, key: str, data: bytes, content_type: str = "text/plain") -> str:
        self.objects[key] = bytes(data)
        return key

    def get(self, key: str) -> bytes:
        return self.objects[key]

    def exists(self, key: str) -> bool:
        return key in self.objects


class S3ObjectStore:
    """boto3-backed store over one bucket.

    The boto3 client is built lazily on first use; ``client`` may be
    injected (tests use a recording stub with the same three calls).
    """

    def __init__(
        self,
        bucket: str,
        *,
        endpoint_url: str | None = None,
        region: str | None = None,
        client=None,
    ) -> None:
        self._bucket = bucket
        self._endpoint_url = endpoint_url
        self._region = region or "us-east-1"
        self._client = client

    def _s3(self):
        if self._client is None:
            import boto3

            self._client = boto3.client(
                "s3",
                endpoint_url=self._endpoint_url,
                region_name=self._region,
            )
        return self._client

    def put(self, key: str, data: bytes, content_type: str = "text/plain") -> str:
        self._s3().put_object(
            Bucket=self._bucket, Key=key, Body=data, ContentType=content_type
        )
        return key

    def get(self, key: str) -> bytes:
        response = self._s3().get_object(Bucket=self._bucket, Key=key)
        return response["Body"].read()

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self._s3().head_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
                return False
            raise


def get_object_store() -> S3ObjectStore | None:
    """The configured object store, or None when ``S3_BUCKET`` is unset
    (documents then travel inline only — the pre-existing behaviour)."""
    load_dotenv()
    bucket = env_str("S3_BUCKET")
    if not bucket:
        return None
    return S3ObjectStore(
        bucket,
        endpoint_url=env_str("S3_ENDPOINT_URL"),
        region=env_str("S3_REGION") or env_str("AWS_REGION"),
    )
