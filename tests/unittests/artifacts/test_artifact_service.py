# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the artifact service."""

from datetime import datetime
import enum
from typing import Optional
from typing import Union
from unittest import mock
from unittest.mock import patch

from google.adk.artifacts.base_artifact_service import ArtifactVersion
from google.adk.artifacts.gcs_artifact_service import GcsArtifactService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.genai import types
import pytest

Enum = enum.Enum

# Define a fixed datetime object to be returned by datetime.now()
FIXED_DATETIME = datetime(2025, 1, 1, 12, 0, 0)


class ArtifactServiceType(Enum):
  IN_MEMORY = "IN_MEMORY"
  GCS = "GCS"


class MockBlob:
  """Mocks a GCS Blob object.

  This class provides mock implementations for a few common GCS Blob methods,
  allowing the user to test code that interacts with GCS without actually
  connecting to a real bucket.
  """

  def __init__(self, name: str) -> None:
    """Initializes a MockBlob.

    Args:
        name: The name of the blob.
    """
    self.name = name
    self.content: Optional[bytes] = None
    self.content_type: Optional[str] = None

  def upload_from_string(
      self, data: Union[str, bytes], content_type: Optional[str] = None
  ) -> None:
    """Mocks uploading data to the blob (from a string or bytes).

    Args:
        data: The data to upload (string or bytes).
        content_type:  The content type of the data (optional).
    """
    if isinstance(data, str):
      self.content = data.encode("utf-8")
    elif isinstance(data, bytes):
      self.content = data
    else:
      raise TypeError("data must be str or bytes")

    if content_type:
      self.content_type = content_type

  def download_as_bytes(self) -> bytes:
    """Mocks downloading the blob's content as bytes.

    Returns:
        bytes: The content of the blob as bytes.

    Raises:
        Exception: If the blob doesn't exist (hasn't been uploaded to).
    """
    if self.content is None:
      return b""
    return self.content

  def delete(self) -> None:
    """Mocks deleting a blob."""
    self.content = None
    self.content_type = None


class MockBucket:
  """Mocks a GCS Bucket object."""

  def __init__(self, name: str) -> None:
    """Initializes a MockBucket.

    Args:
        name: The name of the bucket.
    """
    self.name = name
    self.blobs: dict[str, MockBlob] = {}

  def blob(self, blob_name: str) -> MockBlob:
    """Mocks getting a Blob object (doesn't create it in storage).

    Args:
        blob_name: The name of the blob.

    Returns:
        A MockBlob instance.
    """
    if blob_name not in self.blobs:
      self.blobs[blob_name] = MockBlob(blob_name)
    return self.blobs[blob_name]


class MockClient:
  """Mocks the GCS Client."""

  def __init__(self) -> None:
    """Initializes MockClient."""
    self.buckets: dict[str, MockBucket] = {}

  def bucket(self, bucket_name: str) -> MockBucket:
    """Mocks getting a Bucket object."""
    if bucket_name not in self.buckets:
      self.buckets[bucket_name] = MockBucket(bucket_name)
    return self.buckets[bucket_name]

  def list_blobs(self, bucket: MockBucket, prefix: Optional[str] = None):
    """Mocks listing blobs in a bucket, optionally with a prefix."""
    if prefix:
      return [
          blob for name, blob in bucket.blobs.items() if name.startswith(prefix)
      ]
    return list(bucket.blobs.values())


def mock_gcs_artifact_service():
  with mock.patch("google.cloud.storage.Client", return_value=MockClient()):
    return GcsArtifactService(bucket_name="test_bucket")


def get_artifact_service(
    service_type: ArtifactServiceType = ArtifactServiceType.IN_MEMORY,
):
  """Creates an artifact service for testing."""
  if service_type == ArtifactServiceType.GCS:
    return mock_gcs_artifact_service()
  return InMemoryArtifactService()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "service_type", [ArtifactServiceType.IN_MEMORY, ArtifactServiceType.GCS]
)
async def test_load_empty(service_type):
  """Tests loading an artifact when none exists."""
  artifact_service = get_artifact_service(service_type)
  assert not await artifact_service.load_artifact(
      app_name="test_app",
      user_id="test_user",
      session_id="session_id",
      filename="filename",
  )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "service_type", [ArtifactServiceType.IN_MEMORY, ArtifactServiceType.GCS]
)
async def test_save_load_delete(service_type):
  """Tests saving, loading, and deleting an artifact."""
  artifact_service = get_artifact_service(service_type)
  artifact = types.Part.from_bytes(data=b"test_data", mime_type="text/plain")
  app_name = "app0"
  user_id = "user0"
  session_id = "123"
  filename = "file456"

  await artifact_service.save_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
      artifact=artifact,
  )
  assert (
      await artifact_service.load_artifact(
          app_name=app_name,
          user_id=user_id,
          session_id=session_id,
          filename=filename,
      )
      == artifact
  )

  # Attempt to load a version that doesn't exist
  assert not await artifact_service.load_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
      version=3,
  )

  await artifact_service.delete_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
  )
  assert not await artifact_service.load_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
  )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "service_type", [ArtifactServiceType.IN_MEMORY, ArtifactServiceType.GCS]
)
async def test_list_keys(service_type):
  """Tests listing keys in the artifact service."""
  artifact_service = get_artifact_service(service_type)
  artifact = types.Part.from_bytes(data=b"test_data", mime_type="text/plain")
  app_name = "app0"
  user_id = "user0"
  session_id = "123"
  filename = "filename"
  filenames = [filename + str(i) for i in range(5)]

  for f in filenames:
    await artifact_service.save_artifact(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=f,
        artifact=artifact,
    )

  assert (
      await artifact_service.list_artifact_keys(
          app_name=app_name, user_id=user_id, session_id=session_id
      )
      == filenames
  )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "service_type", [ArtifactServiceType.IN_MEMORY, ArtifactServiceType.GCS]
)
async def test_list_versions(service_type):
  """Tests listing versions of an artifact."""
  artifact_service = get_artifact_service(service_type)

  app_name = "app0"
  user_id = "user0"
  session_id = "123"
  filename = "with/slash/filename"
  versions = [
      types.Part.from_bytes(
          data=i.to_bytes(2, byteorder="big"), mime_type="text/plain"
      )
      for i in range(3)
  ]
  versions.append(types.Part.from_text(text="hello"))

  for i in range(4):
    await artifact_service.save_artifact(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=filename,
        artifact=versions[i],
    )

  response_versions = await artifact_service.list_versions(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
  )

  assert response_versions == list(range(4))


@pytest.mark.asyncio
async def test_list_keys_preserves_user_prefix():
  """Tests that list_artifact_keys preserves 'user:' prefix in returned names."""
  artifact_service = InMemoryArtifactService()
  artifact = types.Part.from_bytes(data=b"test_data", mime_type="text/plain")
  app_name = "app0"
  user_id = "user0"
  session_id = "123"

  # Save artifacts with "user:" prefix (cross-session artifacts)
  await artifact_service.save_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename="user:document.pdf",
      artifact=artifact,
  )

  await artifact_service.save_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename="user:image.png",
      artifact=artifact,
  )

  # Save session-scoped artifact without prefix
  await artifact_service.save_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename="session_file.txt",
      artifact=artifact,
  )

  # List artifacts should return names with "user:" prefix for user-scoped artifacts
  artifact_keys = await artifact_service.list_artifact_keys(
      app_name=app_name, user_id=user_id, session_id=session_id
  )

  # Should contain prefixed names and session file
  expected_keys = ["user:document.pdf", "user:image.png", "session_file.txt"]
  assert sorted(artifact_keys) == sorted(expected_keys)


@pytest.mark.asyncio
async def test_list_artifact_versions_and_get_artifact_version():
  """Tests listing artifact versions and getting a specific version."""
  artifact_service = InMemoryArtifactService()
  app_name = "app0"
  user_id = "user0"
  session_id = "123"
  filename = "filename"
  versions = [
      types.Part.from_bytes(
          data=i.to_bytes(2, byteorder="big"), mime_type="text/plain"
      )
      for i in range(4)
  ]

  with patch(
      "google.adk.artifacts.base_artifact_service.datetime"
  ) as mock_datetime:
    mock_datetime.now.return_value = FIXED_DATETIME

    for i in range(4):
      await artifact_service.save_artifact(
          app_name=app_name,
          user_id=user_id,
          session_id=session_id,
          filename=filename,
          artifact=versions[i],
          custom_metadata={"key": "value" + str(i)},
      )

    artifact_versions = await artifact_service.list_artifact_versions(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=filename,
    )

    expected_artifact_versions = [
        ArtifactVersion(
            version=i,
            canonical_uri=(
                f"memory://apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{filename}/versions/{i}"
            ),
            custom_metadata={"key": "value" + str(i)},
            mime_type="text/plain",
            create_time=FIXED_DATETIME.timestamp(),
        )
        for i in range(4)
    ]
    assert artifact_versions == expected_artifact_versions

    # Get latest artifact version when version is not specified
    assert (
        await artifact_service.get_artifact_version(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            filename=filename,
        )
        == expected_artifact_versions[-1]
    )

    # Get artifact version by version number
    assert (
        await artifact_service.get_artifact_version(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            filename=filename,
            version=2,
        )
        == expected_artifact_versions[2]
    )


@pytest.mark.asyncio
async def test_list_artifact_versions_with_user_prefix():
  """Tests listing artifact versions with user prefix."""
  artifact_service = InMemoryArtifactService()
  app_name = "app0"
  user_id = "user0"
  session_id = "123"
  user_scoped_filename = "user:document.pdf"
  versions = [
      types.Part.from_bytes(
          data=i.to_bytes(2, byteorder="big"), mime_type="text/plain"
      )
      for i in range(4)
  ]

  with patch(
      "google.adk.artifacts.base_artifact_service.datetime"
  ) as mock_datetime:
    mock_datetime.now.return_value = FIXED_DATETIME

    for i in range(4):
      # Save artifacts with "user:" prefix (cross-session artifacts)
      await artifact_service.save_artifact(
          app_name=app_name,
          user_id=user_id,
          session_id=session_id,
          filename=user_scoped_filename,
          artifact=versions[i],
          custom_metadata={"key": "value" + str(i)},
      )

    artifact_versions = await artifact_service.list_artifact_versions(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=user_scoped_filename,
    )

    expected_artifact_versions = [
        ArtifactVersion(
            version=i,
            canonical_uri=(
                f"memory://apps/{app_name}/users/{user_id}/artifacts/{user_scoped_filename}/versions/{i}"
            ),
            custom_metadata={"key": "value" + str(i)},
            mime_type="text/plain",
            create_time=FIXED_DATETIME.timestamp(),
        )
        for i in range(4)
    ]
    assert artifact_versions == expected_artifact_versions


@pytest.mark.asyncio
async def test_get_artifact_version_artifact_does_not_exist():
  """Tests getting an artifact version when artifact does not exist."""
  artifact_service = InMemoryArtifactService()
  assert not await artifact_service.get_artifact_version(
      app_name="test_app",
      user_id="test_user",
      session_id="session_id",
      filename="filename",
  )


@pytest.mark.asyncio
async def test_get_artifact_version_out_of_index():
  """Tests loading an artifact with an out-of-index version."""
  artifact_service = InMemoryArtifactService()
  app_name = "app0"
  user_id = "user0"
  session_id = "123"
  filename = "filename"
  artifact = types.Part.from_bytes(data=b"test_data", mime_type="text/plain")

  await artifact_service.save_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
      artifact=artifact,
  )

  # Attempt to get a version that doesn't exist
  assert not await artifact_service.get_artifact_version(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
      version=3,
  )
