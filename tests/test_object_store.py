"""Object storage: the port, the S3 implementation over a stub
client, and the service's fetch/archive behaviour."""

from __future__ import annotations

import copy
import io

import pytest

from shipment_agent.model_backends import MockModelBackend
from shipment_agent.object_store import InMemoryObjectStore, S3ObjectStore
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.samples import load_sample_shipments
from shipment_agent.schemas import ShipmentInput
from shipment_agent.service import ShipmentService
from shipment_agent.store import InMemoryStore

SAMPLES = {s["shipment_id"]: s for s in load_sample_shipments()}


def _service(records, object_store) -> ShipmentService:
    return ShipmentService(
        backend=MockModelBackend(),
        retriever=KeywordRetriever(),
        store=records,
        checkpointer=False,
        object_store=object_store,
    )


def test_inmemory_object_store_round_trip():
    store = InMemoryObjectStore()
    assert store.exists("k") is False
    assert store.put("k", b"abc") == "k"
    assert store.exists("k") is True
    assert store.get("k") == b"abc"
    with pytest.raises(KeyError):
        store.get("missing")


class _StubS3Client:
    """Records the three boto3 calls the S3 store makes."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.put_calls: list[dict] = []

    def put_object(self, Bucket, Key, Body, ContentType):
        self.put_calls.append(
            {"Bucket": Bucket, "Key": Key, "Body": Body, "ContentType": ContentType}
        )
        self.objects[(Bucket, Key)] = Body

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def head_object(self, Bucket, Key):
        from botocore.exceptions import ClientError

        if (Bucket, Key) not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return {}


def test_s3_object_store_over_stub_client():
    client = _StubS3Client()
    store = S3ObjectStore("shipment-documents", client=client)
    assert store.exists("a.txt") is False
    store.put("a.txt", b"hello", "text/plain")
    assert client.put_calls[0]["Bucket"] == "shipment-documents"
    assert store.exists("a.txt") is True
    assert store.get("a.txt") == b"hello"


def test_service_fetches_key_only_documents():
    objects = InMemoryObjectStore()
    objects.put("inbox/syn-9001-bol.txt", b"Bill of lading text fetched from the bucket")
    payload = copy.deepcopy(SAMPLES["SYN-1001"])
    payload["shipment_id"] = "SYN-9001"
    payload["documents"] = [
        {
            "doc_type": "bol",
            "document_id": "BOL-9",
            "object_key": "inbox/syn-9001-bol.txt",
            "raw_text": "",
            "fields": {},
        }
    ]
    records = InMemoryStore()
    service = _service(records, objects)
    service.analyze(ShipmentInput.model_validate(payload))
    saved = records.get("SYN-9001")
    assert saved is not None
    assert (
        saved.shipment["documents"][0]["raw_text"]
        == "Bill of lading text fetched from the bucket"
    )


def test_service_archives_inline_documents_and_records_keys():
    objects = InMemoryObjectStore()
    records = InMemoryStore()
    service = _service(records, objects)
    service.analyze(copy.deepcopy(SAMPLES["SYN-1001"]))
    saved = records.get("SYN-1001")
    assert saved is not None
    archived = [
        doc for doc in saved.shipment["documents"] if doc.get("object_key")
    ]
    assert archived, "expected at least one document to be archived"
    for doc in archived:
        assert objects.exists(doc["object_key"])
        assert objects.get(doc["object_key"]).decode("utf-8") == doc["raw_text"]


def test_service_without_object_store_is_unchanged():
    records = InMemoryStore()
    service = _service(records, False)
    service.analyze(copy.deepcopy(SAMPLES["SYN-1001"]))
    saved = records.get("SYN-1001")
    assert saved is not None
    assert all(not doc.get("object_key") for doc in saved.shipment["documents"])
