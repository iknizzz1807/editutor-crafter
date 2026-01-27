#!/usr/bin/env python3
"""
Add Content & Media projects - file uploads, media processing, CDN.
"""

import yaml
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(script_dir, '..', 'data', 'projects.yaml')

content_media_projects = {
    "file-upload-service": {
        "name": "Resumable File Upload Service",
        "description": "Build a file upload service supporting chunked uploads, resumable transfers, virus scanning, and storage backends (S3, local).",
        "why_expert": "File uploads are tricky - large files, network failures, security. Understanding chunked protocols and storage abstractions enables reliable upload handling.",
        "difficulty": "advanced",
        "tags": ["files", "uploads", "storage", "s3", "resumable"],
        "estimated_hours": 35,
        "prerequisites": ["build-http-server"],
        "milestones": [
            {
                "name": "Chunked Upload Protocol",
                "description": "Implement tus.io-style resumable upload protocol",
                "skills": ["Chunked uploads", "Resume logic", "HTTP headers"],
                "hints": {
                    "level1": "Client sends chunks; server tracks offset; resume from last chunk",
                    "level2": "Use Upload-Offset header to track progress",
                    "level3": """
```python
from dataclasses import dataclass, field
from typing import Optional, BinaryIO
from enum import Enum
import os
import secrets
import time
import hashlib

class UploadStatus(Enum):
    CREATED = "created"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    EXPIRED = "expired"

@dataclass
class Upload:
    id: str
    size: int                    # Total file size
    offset: int                  # Current offset
    status: UploadStatus
    filename: Optional[str]
    content_type: Optional[str]
    metadata: dict
    created_at: float
    expires_at: float
    checksum: Optional[str] = None

class ChunkedUploadService:
    def __init__(self, storage_path: str, max_size: int = 5 * 1024**3,
                 chunk_size: int = 5 * 1024**2, expiry_hours: int = 24):
        self.storage_path = storage_path
        self.max_size = max_size
        self.chunk_size = chunk_size
        self.expiry_hours = expiry_hours
        self.uploads: dict[str, Upload] = {}

        os.makedirs(storage_path, exist_ok=True)
        os.makedirs(os.path.join(storage_path, 'chunks'), exist_ok=True)
        os.makedirs(os.path.join(storage_path, 'complete'), exist_ok=True)

    def create_upload(self, size: int, filename: str = None,
                      content_type: str = None, metadata: dict = None) -> Upload:
        '''Create new upload session (POST /uploads)'''
        if size > self.max_size:
            raise ValueError(f"File size exceeds maximum ({self.max_size})")

        upload_id = secrets.token_urlsafe(16)
        now = time.time()

        upload = Upload(
            id=upload_id,
            size=size,
            offset=0,
            status=UploadStatus.CREATED,
            filename=filename,
            content_type=content_type,
            metadata=metadata or {},
            created_at=now,
            expires_at=now + (self.expiry_hours * 3600)
        )

        self.uploads[upload_id] = upload

        # Create chunk file
        chunk_path = self._get_chunk_path(upload_id)
        with open(chunk_path, 'wb') as f:
            f.truncate(size)  # Pre-allocate

        return upload

    def upload_chunk(self, upload_id: str, data: bytes,
                     offset: int, checksum: str = None) -> Upload:
        '''Upload a chunk (PATCH /uploads/{id})'''
        upload = self.uploads.get(upload_id)
        if not upload:
            raise ValueError("Upload not found")

        if upload.status == UploadStatus.EXPIRED:
            raise ValueError("Upload expired")

        if upload.status == UploadStatus.COMPLETE:
            raise ValueError("Upload already complete")

        if offset != upload.offset:
            raise ValueError(f"Offset mismatch. Expected {upload.offset}, got {offset}")

        # Verify checksum if provided
        if checksum:
            computed = hashlib.sha256(data).hexdigest()
            if computed != checksum:
                raise ValueError("Checksum mismatch")

        # Write chunk
        chunk_path = self._get_chunk_path(upload_id)
        with open(chunk_path, 'r+b') as f:
            f.seek(offset)
            f.write(data)

        upload.offset += len(data)
        upload.status = UploadStatus.IN_PROGRESS

        # Check if complete
        if upload.offset >= upload.size:
            self._finalize_upload(upload)

        return upload

    def get_upload(self, upload_id: str) -> Optional[Upload]:
        '''Get upload status (HEAD /uploads/{id})'''
        upload = self.uploads.get(upload_id)
        if upload and time.time() > upload.expires_at:
            upload.status = UploadStatus.EXPIRED
        return upload

    def _finalize_upload(self, upload: Upload):
        '''Move completed upload to final location'''
        chunk_path = self._get_chunk_path(upload.id)
        final_path = self._get_final_path(upload.id, upload.filename)

        os.rename(chunk_path, final_path)
        upload.status = UploadStatus.COMPLETE

        # Calculate final checksum
        with open(final_path, 'rb') as f:
            upload.checksum = hashlib.sha256(f.read()).hexdigest()

    def _get_chunk_path(self, upload_id: str) -> str:
        return os.path.join(self.storage_path, 'chunks', upload_id)

    def _get_final_path(self, upload_id: str, filename: str = None) -> str:
        name = filename or upload_id
        return os.path.join(self.storage_path, 'complete', f"{upload_id}_{name}")

    def cleanup_expired(self):
        '''Remove expired uploads'''
        now = time.time()
        expired = [uid for uid, u in self.uploads.items()
                   if now > u.expires_at and u.status != UploadStatus.COMPLETE]

        for upload_id in expired:
            chunk_path = self._get_chunk_path(upload_id)
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
            del self.uploads[upload_id]
```
"""
                },
                "pitfalls": [
                    "Pre-allocate file to avoid fragmentation",
                    "Offset mismatch: client must track and resume from server's offset",
                    "Chunk checksum prevents corruption from network errors",
                    "Expiry cleanup must not delete in-progress uploads"
                ]
            },
            {
                "name": "Storage Abstraction",
                "description": "Implement storage backends (local, S3, GCS) with common interface",
                "skills": ["Storage abstraction", "S3 API", "Multipart uploads"],
                "hints": {
                    "level1": "Abstract storage: put, get, delete, list operations",
                    "level2": "S3 multipart upload for large files",
                    "level3": """
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import BinaryIO, Iterator, Optional
import os
import shutil

@dataclass
class StorageObject:
    key: str
    size: int
    content_type: Optional[str]
    last_modified: float
    metadata: dict
    etag: Optional[str] = None

class StorageBackend(ABC):
    @abstractmethod
    def put(self, key: str, data: BinaryIO, size: int,
            content_type: str = None, metadata: dict = None) -> StorageObject:
        pass

    @abstractmethod
    def get(self, key: str) -> tuple[BinaryIO, StorageObject]:
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        pass

    @abstractmethod
    def list(self, prefix: str = "", limit: int = 1000) -> Iterator[StorageObject]:
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        pass

class LocalStorageBackend(StorageBackend):
    def __init__(self, base_path: str):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def _get_path(self, key: str) -> str:
        # Sanitize key to prevent path traversal
        safe_key = key.replace('..', '').lstrip('/')
        return os.path.join(self.base_path, safe_key)

    def put(self, key: str, data: BinaryIO, size: int,
            content_type: str = None, metadata: dict = None) -> StorageObject:
        path = self._get_path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'wb') as f:
            shutil.copyfileobj(data, f)

        # Store metadata in sidecar file
        meta_path = path + '.meta'
        import json
        with open(meta_path, 'w') as f:
            json.dump({
                'content_type': content_type,
                'metadata': metadata or {}
            }, f)

        stat = os.stat(path)
        return StorageObject(
            key=key,
            size=stat.st_size,
            content_type=content_type,
            last_modified=stat.st_mtime,
            metadata=metadata or {}
        )

    def get(self, key: str) -> tuple[BinaryIO, StorageObject]:
        path = self._get_path(key)
        if not os.path.exists(path):
            raise FileNotFoundError(key)

        stat = os.stat(path)

        # Load metadata
        meta_path = path + '.meta'
        content_type = None
        metadata = {}
        if os.path.exists(meta_path):
            import json
            with open(meta_path) as f:
                meta = json.load(f)
                content_type = meta.get('content_type')
                metadata = meta.get('metadata', {})

        obj = StorageObject(
            key=key,
            size=stat.st_size,
            content_type=content_type,
            last_modified=stat.st_mtime,
            metadata=metadata
        )

        return open(path, 'rb'), obj

    def delete(self, key: str) -> bool:
        path = self._get_path(key)
        meta_path = path + '.meta'

        if os.path.exists(path):
            os.remove(path)
            if os.path.exists(meta_path):
                os.remove(meta_path)
            return True
        return False

    def list(self, prefix: str = "", limit: int = 1000) -> Iterator[StorageObject]:
        base = self._get_path(prefix)
        count = 0

        for root, dirs, files in os.walk(self.base_path):
            for name in files:
                if name.endswith('.meta'):
                    continue

                path = os.path.join(root, name)
                key = os.path.relpath(path, self.base_path)

                if prefix and not key.startswith(prefix):
                    continue

                stat = os.stat(path)
                yield StorageObject(
                    key=key,
                    size=stat.st_size,
                    content_type=None,
                    last_modified=stat.st_mtime,
                    metadata={}
                )

                count += 1
                if count >= limit:
                    return

    def exists(self, key: str) -> bool:
        return os.path.exists(self._get_path(key))

class S3StorageBackend(StorageBackend):
    '''S3-compatible storage backend'''

    def __init__(self, bucket: str, region: str = 'us-east-1',
                 endpoint_url: str = None):
        import boto3
        self.bucket = bucket
        self.client = boto3.client(
            's3',
            region_name=region,
            endpoint_url=endpoint_url  # For MinIO, LocalStack
        )

    def put(self, key: str, data: BinaryIO, size: int,
            content_type: str = None, metadata: dict = None) -> StorageObject:

        extra_args = {}
        if content_type:
            extra_args['ContentType'] = content_type
        if metadata:
            extra_args['Metadata'] = {k: str(v) for k, v in metadata.items()}

        # Use multipart for large files
        if size > 100 * 1024 * 1024:  # 100MB
            return self._multipart_upload(key, data, size, extra_args)

        self.client.upload_fileobj(data, self.bucket, key, ExtraArgs=extra_args)

        # Get object info
        response = self.client.head_object(Bucket=self.bucket, Key=key)

        return StorageObject(
            key=key,
            size=response['ContentLength'],
            content_type=response.get('ContentType'),
            last_modified=response['LastModified'].timestamp(),
            metadata=response.get('Metadata', {}),
            etag=response['ETag'].strip('"')
        )

    def _multipart_upload(self, key: str, data: BinaryIO, size: int,
                          extra_args: dict) -> StorageObject:
        '''Multipart upload for large files'''
        part_size = 100 * 1024 * 1024  # 100MB parts

        # Initiate multipart upload
        response = self.client.create_multipart_upload(
            Bucket=self.bucket,
            Key=key,
            **extra_args
        )
        upload_id = response['UploadId']

        parts = []
        part_number = 1

        try:
            while True:
                chunk = data.read(part_size)
                if not chunk:
                    break

                response = self.client.upload_part(
                    Bucket=self.bucket,
                    Key=key,
                    UploadId=upload_id,
                    PartNumber=part_number,
                    Body=chunk
                )

                parts.append({
                    'PartNumber': part_number,
                    'ETag': response['ETag']
                })
                part_number += 1

            # Complete upload
            self.client.complete_multipart_upload(
                Bucket=self.bucket,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={'Parts': parts}
            )

        except Exception:
            # Abort on failure
            self.client.abort_multipart_upload(
                Bucket=self.bucket,
                Key=key,
                UploadId=upload_id
            )
            raise

        return self.get(key)[1]

    def get(self, key: str) -> tuple[BinaryIO, StorageObject]:
        response = self.client.get_object(Bucket=self.bucket, Key=key)

        obj = StorageObject(
            key=key,
            size=response['ContentLength'],
            content_type=response.get('ContentType'),
            last_modified=response['LastModified'].timestamp(),
            metadata=response.get('Metadata', {}),
            etag=response['ETag'].strip('"')
        )

        return response['Body'], obj

    def delete(self, key: str) -> bool:
        try:
            self.client.delete_object(Bucket=self.bucket, Key=key)
            return True
        except:
            return False

    def list(self, prefix: str = "", limit: int = 1000) -> Iterator[StorageObject]:
        paginator = self.client.get_paginator('list_objects_v2')
        count = 0

        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                yield StorageObject(
                    key=obj['Key'],
                    size=obj['Size'],
                    content_type=None,
                    last_modified=obj['LastModified'].timestamp(),
                    metadata={},
                    etag=obj['ETag'].strip('"')
                )
                count += 1
                if count >= limit:
                    return

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except:
            return False
```
"""
                },
                "pitfalls": [
                    "S3 multipart: parts must be at least 5MB (except last)",
                    "Local storage: sanitize keys to prevent path traversal",
                    "S3 eventually consistent for overwrite - use versioning",
                    "Streaming large files - don't load entire file in memory"
                ]
            },
            {
                "name": "Virus Scanning & Validation",
                "description": "Implement file validation with type checking and virus scanning",
                "skills": ["File validation", "MIME detection", "Antivirus integration"],
                "hints": {
                    "level1": "Check magic bytes, not just extension",
                    "level2": "ClamAV for virus scanning via clamd socket",
                    "level3": """
```python
from dataclasses import dataclass
from typing import Optional, BinaryIO
from enum import Enum
import mimetypes

class ScanResult(Enum):
    CLEAN = "clean"
    INFECTED = "infected"
    ERROR = "error"
    SKIPPED = "skipped"

@dataclass
class ValidationResult:
    valid: bool
    mime_type: Optional[str]
    detected_extension: Optional[str]
    scan_result: ScanResult
    scan_details: Optional[str] = None
    errors: list[str] = None

class FileValidator:
    # Magic bytes for common file types
    MAGIC_BYTES = {
        b'\\x89PNG\\r\\n\\x1a\\n': ('image/png', '.png'),
        b'\\xff\\xd8\\xff': ('image/jpeg', '.jpg'),
        b'GIF87a': ('image/gif', '.gif'),
        b'GIF89a': ('image/gif', '.gif'),
        b'%PDF': ('application/pdf', '.pdf'),
        b'PK\\x03\\x04': ('application/zip', '.zip'),
        b'\\x1f\\x8b': ('application/gzip', '.gz'),
    }

    def __init__(self, allowed_types: list[str] = None,
                 max_size: int = 100 * 1024 * 1024,
                 scan_viruses: bool = True):
        self.allowed_types = set(allowed_types) if allowed_types else None
        self.max_size = max_size
        self.scan_viruses = scan_viruses
        self.scanner = ClamAVScanner() if scan_viruses else None

    def validate(self, data: BinaryIO, filename: str = None,
                 claimed_type: str = None) -> ValidationResult:
        errors = []

        # Read header for magic byte detection
        header = data.read(16)
        data.seek(0)

        # Detect MIME type from magic bytes
        detected_type, detected_ext = self._detect_type(header)

        # Check size
        data.seek(0, 2)  # Seek to end
        size = data.tell()
        data.seek(0)

        if size > self.max_size:
            errors.append(f"File too large: {size} > {self.max_size}")

        # Check allowed types
        if self.allowed_types and detected_type:
            if detected_type not in self.allowed_types:
                errors.append(f"File type not allowed: {detected_type}")

        # Check extension matches content
        if filename and detected_ext:
            actual_ext = '.' + filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
            if actual_ext and actual_ext != detected_ext:
                errors.append(f"Extension mismatch: {actual_ext} vs detected {detected_ext}")

        # Virus scan
        scan_result = ScanResult.SKIPPED
        scan_details = None

        if self.scan_viruses and self.scanner:
            scan_result, scan_details = self.scanner.scan(data)
            data.seek(0)

            if scan_result == ScanResult.INFECTED:
                errors.append(f"Virus detected: {scan_details}")

        return ValidationResult(
            valid=len(errors) == 0,
            mime_type=detected_type,
            detected_extension=detected_ext,
            scan_result=scan_result,
            scan_details=scan_details,
            errors=errors if errors else None
        )

    def _detect_type(self, header: bytes) -> tuple[Optional[str], Optional[str]]:
        for magic, (mime_type, ext) in self.MAGIC_BYTES.items():
            if header.startswith(magic):
                return mime_type, ext
        return None, None

class ClamAVScanner:
    '''Virus scanner using ClamAV daemon'''

    def __init__(self, socket_path: str = '/var/run/clamav/clamd.ctl',
                 host: str = None, port: int = 3310):
        self.socket_path = socket_path
        self.host = host
        self.port = port

    def scan(self, data: BinaryIO) -> tuple[ScanResult, Optional[str]]:
        import socket

        try:
            if self.host:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((self.host, self.port))
            else:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(self.socket_path)

            # Send INSTREAM command
            sock.send(b'nINSTREAM\\n')

            # Send file in chunks
            while True:
                chunk = data.read(8192)
                if not chunk:
                    break
                # Send chunk size (4 bytes big endian) + data
                sock.send(len(chunk).to_bytes(4, 'big') + chunk)

            # Send zero-length chunk to end
            sock.send(b'\\x00\\x00\\x00\\x00')

            # Read response
            response = sock.recv(4096).decode().strip()
            sock.close()

            if 'OK' in response:
                return ScanResult.CLEAN, None
            elif 'FOUND' in response:
                virus_name = response.split('FOUND')[0].strip()
                return ScanResult.INFECTED, virus_name
            else:
                return ScanResult.ERROR, response

        except Exception as e:
            return ScanResult.ERROR, str(e)
```
"""
                },
                "pitfalls": [
                    "Never trust file extension alone - always check magic bytes",
                    "Virus scan before storing, not after",
                    "ClamAV can timeout on large files - set appropriate limits",
                    "Quarantine infected files, don't delete immediately (for forensics)"
                ]
            }
        ]
    },

    "media-processing": {
        "name": "Media Processing Pipeline",
        "description": "Build a media processing service for image resizing, video transcoding, and thumbnail generation.",
        "why_expert": "Media processing is CPU-intensive and complex. Understanding codecs, formats, and async processing enables building efficient media services.",
        "difficulty": "advanced",
        "tags": ["media", "images", "video", "transcoding", "thumbnails"],
        "estimated_hours": 40,
        "prerequisites": ["build-message-queue"],
        "milestones": [
            {
                "name": "Image Processing",
                "description": "Implement image resizing, format conversion, and optimization",
                "skills": ["Image formats", "Resizing algorithms", "Optimization"],
                "hints": {
                    "level1": "Generate multiple sizes on upload for responsive images",
                    "level2": "Preserve aspect ratio; use Lanczos for quality downscaling",
                    "level3": """
```python
from dataclasses import dataclass
from typing import Optional, BinaryIO
from enum import Enum
from PIL import Image
import io

class ImageFormat(Enum):
    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"
    AVIF = "avif"

@dataclass
class ImageVariant:
    name: str
    width: int
    height: Optional[int]  # None = maintain aspect ratio
    format: ImageFormat
    quality: int = 85

@dataclass
class ProcessedImage:
    variant: str
    width: int
    height: int
    format: ImageFormat
    size: int
    data: bytes

class ImageProcessor:
    DEFAULT_VARIANTS = [
        ImageVariant("thumbnail", 150, 150, ImageFormat.WEBP, 80),
        ImageVariant("small", 320, None, ImageFormat.WEBP, 85),
        ImageVariant("medium", 640, None, ImageFormat.WEBP, 85),
        ImageVariant("large", 1280, None, ImageFormat.WEBP, 90),
        ImageVariant("original", 0, None, ImageFormat.WEBP, 95),  # 0 = keep size
    ]

    def __init__(self, variants: list[ImageVariant] = None):
        self.variants = variants or self.DEFAULT_VARIANTS

    def process(self, image_data: BinaryIO) -> list[ProcessedImage]:
        '''Process image into all variants'''
        original = Image.open(image_data)

        # Convert to RGB if necessary (for JPEG output)
        if original.mode in ('RGBA', 'P'):
            background = Image.new('RGB', original.size, (255, 255, 255))
            if original.mode == 'P':
                original = original.convert('RGBA')
            background.paste(original, mask=original.split()[3])
            original = background
        elif original.mode != 'RGB':
            original = original.convert('RGB')

        results = []
        for variant in self.variants:
            processed = self._process_variant(original, variant)
            results.append(processed)

        return results

    def _process_variant(self, image: Image.Image,
                         variant: ImageVariant) -> ProcessedImage:
        # Calculate target size
        if variant.width == 0:
            # Keep original size
            target_size = image.size
        elif variant.height:
            # Fixed dimensions (crop to fit)
            target_size = (variant.width, variant.height)
            image = self._crop_to_aspect(image, target_size)
        else:
            # Width only - maintain aspect ratio
            ratio = variant.width / image.width
            target_size = (variant.width, int(image.height * ratio))

        # Resize if needed
        if target_size != image.size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)

        # Convert to output format
        output = io.BytesIO()
        format_str = variant.format.value.upper()

        if variant.format == ImageFormat.WEBP:
            image.save(output, 'WEBP', quality=variant.quality, method=6)
        elif variant.format == ImageFormat.JPEG:
            image.save(output, 'JPEG', quality=variant.quality, optimize=True)
        elif variant.format == ImageFormat.PNG:
            image.save(output, 'PNG', optimize=True)
        elif variant.format == ImageFormat.AVIF:
            # Requires pillow-avif-plugin
            image.save(output, 'AVIF', quality=variant.quality)

        data = output.getvalue()

        return ProcessedImage(
            variant=variant.name,
            width=image.width,
            height=image.height,
            format=variant.format,
            size=len(data),
            data=data
        )

    def _crop_to_aspect(self, image: Image.Image,
                        target_size: tuple[int, int]) -> Image.Image:
        '''Crop image to target aspect ratio (center crop)'''
        target_ratio = target_size[0] / target_size[1]
        image_ratio = image.width / image.height

        if image_ratio > target_ratio:
            # Image is wider - crop sides
            new_width = int(image.height * target_ratio)
            left = (image.width - new_width) // 2
            image = image.crop((left, 0, left + new_width, image.height))
        elif image_ratio < target_ratio:
            # Image is taller - crop top/bottom
            new_height = int(image.width / target_ratio)
            top = (image.height - new_height) // 2
            image = image.crop((0, top, image.width, top + new_height))

        return image

    def extract_metadata(self, image_data: BinaryIO) -> dict:
        '''Extract EXIF and other metadata'''
        image = Image.open(image_data)
        metadata = {
            'width': image.width,
            'height': image.height,
            'format': image.format,
            'mode': image.mode
        }

        # Extract EXIF
        if hasattr(image, '_getexif') and image._getexif():
            exif = image._getexif()
            # Map common EXIF tags
            exif_tags = {
                271: 'camera_make',
                272: 'camera_model',
                306: 'datetime',
                274: 'orientation'
            }
            for tag_id, name in exif_tags.items():
                if tag_id in exif:
                    metadata[name] = exif[tag_id]

        return metadata
```
"""
                },
                "pitfalls": [
                    "EXIF orientation: rotate image according to tag before processing",
                    "WebP/AVIF save significant bandwidth but check browser support",
                    "Lanczos is best for downscaling; use different for upscaling",
                    "Strip EXIF from output for privacy (location data)"
                ]
            },
            {
                "name": "Video Transcoding",
                "description": "Implement video transcoding with FFmpeg for web playback",
                "skills": ["FFmpeg", "Video codecs", "HLS streaming"],
                "hints": {
                    "level1": "Transcode to H.264/AAC for broad compatibility",
                    "level2": "Generate HLS segments for adaptive streaming",
                    "level3": """
```python
from dataclasses import dataclass
from typing import Optional
import subprocess
import os
import json

@dataclass
class VideoProfile:
    name: str
    width: int
    height: int
    video_bitrate: str
    audio_bitrate: str
    preset: str = "medium"

@dataclass
class TranscodeResult:
    profile: str
    path: str
    width: int
    height: int
    duration: float
    size: int

class VideoTranscoder:
    PROFILES = [
        VideoProfile("360p", 640, 360, "800k", "96k"),
        VideoProfile("480p", 854, 480, "1400k", "128k"),
        VideoProfile("720p", 1280, 720, "2800k", "128k"),
        VideoProfile("1080p", 1920, 1080, "5000k", "192k"),
    ]

    def __init__(self, output_dir: str, ffmpeg_path: str = "ffmpeg"):
        self.output_dir = output_dir
        self.ffmpeg = ffmpeg_path
        os.makedirs(output_dir, exist_ok=True)

    def get_video_info(self, input_path: str) -> dict:
        '''Get video metadata using ffprobe'''
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_format', '-show_streams',
            input_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)

        video_stream = next(
            (s for s in data['streams'] if s['codec_type'] == 'video'),
            None
        )

        return {
            'duration': float(data['format'].get('duration', 0)),
            'width': video_stream['width'] if video_stream else 0,
            'height': video_stream['height'] if video_stream else 0,
            'codec': video_stream['codec_name'] if video_stream else None,
            'bitrate': int(data['format'].get('bit_rate', 0))
        }

    def transcode_mp4(self, input_path: str, job_id: str,
                      profile: VideoProfile) -> TranscodeResult:
        '''Transcode to MP4 with H.264'''
        output_path = os.path.join(
            self.output_dir, f"{job_id}_{profile.name}.mp4"
        )

        cmd = [
            self.ffmpeg, '-y',
            '-i', input_path,
            '-c:v', 'libx264',
            '-preset', profile.preset,
            '-b:v', profile.video_bitrate,
            '-maxrate', profile.video_bitrate,
            '-bufsize', str(int(profile.video_bitrate.rstrip('k')) * 2) + 'k',
            '-vf', f'scale={profile.width}:{profile.height}:force_original_aspect_ratio=decrease,pad={profile.width}:{profile.height}:(ow-iw)/2:(oh-ih)/2',
            '-c:a', 'aac',
            '-b:a', profile.audio_bitrate,
            '-movflags', '+faststart',  # Enable streaming
            output_path
        ]

        subprocess.run(cmd, check=True, capture_output=True)

        info = self.get_video_info(output_path)
        return TranscodeResult(
            profile=profile.name,
            path=output_path,
            width=info['width'],
            height=info['height'],
            duration=info['duration'],
            size=os.path.getsize(output_path)
        )

    def generate_hls(self, input_path: str, job_id: str) -> dict:
        '''Generate HLS playlist with multiple qualities'''
        hls_dir = os.path.join(self.output_dir, job_id, 'hls')
        os.makedirs(hls_dir, exist_ok=True)

        info = self.get_video_info(input_path)
        source_height = info['height']

        # Select appropriate profiles
        profiles = [p for p in self.PROFILES if p.height <= source_height]
        if not profiles:
            profiles = [self.PROFILES[0]]

        variants = []
        for profile in profiles:
            variant_dir = os.path.join(hls_dir, profile.name)
            os.makedirs(variant_dir, exist_ok=True)

            output_playlist = os.path.join(variant_dir, 'playlist.m3u8')

            cmd = [
                self.ffmpeg, '-y',
                '-i', input_path,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-b:v', profile.video_bitrate,
                '-vf', f'scale={profile.width}:-2',
                '-c:a', 'aac',
                '-b:a', profile.audio_bitrate,
                '-hls_time', '6',
                '-hls_playlist_type', 'vod',
                '-hls_segment_filename', os.path.join(variant_dir, 'segment_%03d.ts'),
                output_playlist
            ]

            subprocess.run(cmd, check=True, capture_output=True)

            variants.append({
                'name': profile.name,
                'bandwidth': int(profile.video_bitrate.rstrip('k')) * 1000,
                'resolution': f'{profile.width}x{profile.height}',
                'playlist': f'{profile.name}/playlist.m3u8'
            })

        # Generate master playlist
        master_path = os.path.join(hls_dir, 'master.m3u8')
        self._write_master_playlist(master_path, variants)

        return {
            'master_playlist': master_path,
            'variants': variants,
            'duration': info['duration']
        }

    def _write_master_playlist(self, path: str, variants: list[dict]):
        with open(path, 'w') as f:
            f.write('#EXTM3U\\n')
            for v in sorted(variants, key=lambda x: x['bandwidth']):
                f.write(f'#EXT-X-STREAM-INF:BANDWIDTH={v["bandwidth"]},RESOLUTION={v["resolution"]}\\n')
                f.write(f'{v["playlist"]}\\n')

    def generate_thumbnail(self, input_path: str, job_id: str,
                           time_offset: float = 1.0) -> str:
        '''Generate thumbnail at specified time'''
        output_path = os.path.join(self.output_dir, f"{job_id}_thumb.jpg")

        cmd = [
            self.ffmpeg, '-y',
            '-i', input_path,
            '-ss', str(time_offset),
            '-vframes', '1',
            '-vf', 'scale=640:-1',
            '-q:v', '3',
            output_path
        ]

        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
```
"""
                },
                "pitfalls": [
                    "H.264 baseline profile for maximum compatibility",
                    "-movflags +faststart enables progressive download",
                    "HLS segment size affects startup time vs seeking",
                    "Video processing is CPU-intensive - use job queue"
                ]
            },
            {
                "name": "Processing Queue & Progress",
                "description": "Implement async processing queue with progress tracking",
                "skills": ["Job queues", "Progress tracking", "Webhooks"],
                "hints": {
                    "level1": "Queue processing jobs; track progress with callbacks",
                    "level2": "Report progress percentage; notify on completion",
                    "level3": """
```python
from dataclasses import dataclass, field
from typing import Callable, Optional
from enum import Enum
import threading
import queue
import time

class JobStatus(Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ProcessingJob:
    id: str
    type: str  # image, video
    input_path: str
    status: JobStatus
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[dict] = None
    error: Optional[str] = None
    webhook_url: Optional[str] = None

class MediaProcessingQueue:
    def __init__(self, image_processor: ImageProcessor,
                 video_transcoder: VideoTranscoder,
                 num_workers: int = 4):
        self.image_processor = image_processor
        self.video_transcoder = video_transcoder
        self.queue = queue.PriorityQueue()
        self.jobs: dict[str, ProcessingJob] = {}
        self.callbacks: dict[str, list[Callable]] = {}
        self.workers = []

        for i in range(num_workers):
            worker = threading.Thread(target=self._worker, daemon=True)
            worker.start()
            self.workers.append(worker)

    def submit(self, job_id: str, job_type: str, input_path: str,
               priority: int = 5, webhook_url: str = None) -> ProcessingJob:
        job = ProcessingJob(
            id=job_id,
            type=job_type,
            input_path=input_path,
            status=JobStatus.QUEUED,
            webhook_url=webhook_url
        )

        self.jobs[job_id] = job
        self.queue.put((priority, time.time(), job_id))

        return job

    def get_status(self, job_id: str) -> Optional[ProcessingJob]:
        return self.jobs.get(job_id)

    def on_progress(self, job_id: str, callback: Callable[[float], None]):
        if job_id not in self.callbacks:
            self.callbacks[job_id] = []
        self.callbacks[job_id].append(callback)

    def _worker(self):
        while True:
            try:
                priority, timestamp, job_id = self.queue.get()
                job = self.jobs.get(job_id)

                if not job:
                    continue

                self._process_job(job)
                self.queue.task_done()

            except Exception as e:
                print(f"Worker error: {e}")

    def _process_job(self, job: ProcessingJob):
        job.status = JobStatus.PROCESSING
        job.started_at = time.time()

        try:
            if job.type == "image":
                result = self._process_image(job)
            elif job.type == "video":
                result = self._process_video(job)
            else:
                raise ValueError(f"Unknown job type: {job.type}")

            job.status = JobStatus.COMPLETED
            job.result = result
            job.progress = 100.0

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)

        finally:
            job.completed_at = time.time()
            self._notify_completion(job)

    def _process_image(self, job: ProcessingJob) -> dict:
        with open(job.input_path, 'rb') as f:
            results = self.image_processor.process(f)

        self._update_progress(job, 100)

        return {
            'variants': [
                {
                    'name': r.variant,
                    'width': r.width,
                    'height': r.height,
                    'format': r.format.value,
                    'size': r.size
                }
                for r in results
            ]
        }

    def _process_video(self, job: ProcessingJob) -> dict:
        # Generate HLS
        self._update_progress(job, 10)

        hls_result = self.video_transcoder.generate_hls(
            job.input_path, job.id
        )
        self._update_progress(job, 80)

        # Generate thumbnail
        thumbnail = self.video_transcoder.generate_thumbnail(
            job.input_path, job.id
        )
        self._update_progress(job, 100)

        return {
            'hls': hls_result,
            'thumbnail': thumbnail
        }

    def _update_progress(self, job: ProcessingJob, progress: float):
        job.progress = progress

        for callback in self.callbacks.get(job.id, []):
            try:
                callback(progress)
            except:
                pass

    def _notify_completion(self, job: ProcessingJob):
        if job.webhook_url:
            import httpx
            try:
                httpx.post(job.webhook_url, json={
                    'job_id': job.id,
                    'status': job.status.value,
                    'result': job.result,
                    'error': job.error
                })
            except:
                pass
```
"""
                },
                "pitfalls": [
                    "Video progress is hard to estimate - use stages not percentage",
                    "Webhook delivery can fail - implement retry",
                    "Clean up temp files after processing",
                    "Memory limits: process one large video at a time"
                ]
            }
        ]
    },

    "cdn-implementation": {
        "name": "Content Delivery Network (CDN)",
        "description": "Build a CDN with edge caching, cache invalidation, and origin shielding.",
        "why_expert": "CDNs are critical for performance. Understanding caching strategies, invalidation, and edge logic helps optimize content delivery.",
        "difficulty": "expert",
        "tags": ["cdn", "caching", "edge", "performance", "distributed"],
        "estimated_hours": 45,
        "prerequisites": ["build-http-server", "build-redis"],
        "milestones": [
            {
                "name": "Edge Cache Implementation",
                "description": "Implement edge caching with TTL and cache control headers",
                "skills": ["HTTP caching", "Cache-Control", "Vary headers"],
                "hints": {
                    "level1": "Cache responses based on Cache-Control headers from origin",
                    "level2": "Vary header determines cache key variations (Accept-Encoding, etc)",
                    "level3": """
```python
from dataclasses import dataclass, field
from typing import Optional
import hashlib
import time

@dataclass
class CacheEntry:
    key: str
    body: bytes
    headers: dict
    status_code: int
    created_at: float
    expires_at: float
    etag: Optional[str]
    last_modified: Optional[str]
    vary_headers: list[str]

class CacheControl:
    def __init__(self, header: str):
        self.max_age: Optional[int] = None
        self.s_maxage: Optional[int] = None
        self.no_cache = False
        self.no_store = False
        self.private = False
        self.public = False
        self.must_revalidate = False
        self.stale_while_revalidate: Optional[int] = None
        self.stale_if_error: Optional[int] = None

        self._parse(header)

    def _parse(self, header: str):
        if not header:
            return

        for directive in header.split(','):
            directive = directive.strip().lower()

            if directive == 'no-cache':
                self.no_cache = True
            elif directive == 'no-store':
                self.no_store = True
            elif directive == 'private':
                self.private = True
            elif directive == 'public':
                self.public = True
            elif directive == 'must-revalidate':
                self.must_revalidate = True
            elif directive.startswith('max-age='):
                self.max_age = int(directive.split('=')[1])
            elif directive.startswith('s-maxage='):
                self.s_maxage = int(directive.split('=')[1])
            elif directive.startswith('stale-while-revalidate='):
                self.stale_while_revalidate = int(directive.split('=')[1])
            elif directive.startswith('stale-if-error='):
                self.stale_if_error = int(directive.split('=')[1])

    def is_cacheable(self) -> bool:
        if self.no_store:
            return False
        if self.private:
            return False  # CDN can't cache private
        return True

    def get_ttl(self) -> int:
        # s-maxage takes precedence for shared caches (CDN)
        if self.s_maxage is not None:
            return self.s_maxage
        if self.max_age is not None:
            return self.max_age
        return 0

class EdgeCache:
    def __init__(self, max_size_bytes: int = 1024 * 1024 * 1024):  # 1GB
        self.cache: dict[str, CacheEntry] = {}
        self.max_size = max_size_bytes
        self.current_size = 0

    def generate_cache_key(self, url: str, vary_headers: dict) -> str:
        '''Generate cache key from URL and Vary headers'''
        key_parts = [url]

        for header_name in sorted(vary_headers.keys()):
            key_parts.append(f"{header_name}:{vary_headers[header_name]}")

        key_string = '|'.join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def get(self, key: str) -> Optional[CacheEntry]:
        entry = self.cache.get(key)
        if not entry:
            return None

        now = time.time()

        # Check if expired
        if now > entry.expires_at:
            # Check stale-while-revalidate
            # (would need to trigger background revalidation)
            return None

        return entry

    def put(self, key: str, response: 'Response') -> Optional[CacheEntry]:
        '''Cache response if cacheable'''
        cache_control = CacheControl(response.headers.get('Cache-Control', ''))

        if not cache_control.is_cacheable():
            return None

        ttl = cache_control.get_ttl()
        if ttl <= 0:
            return None

        # Parse Vary header
        vary_headers = []
        vary = response.headers.get('Vary', '')
        if vary and vary != '*':
            vary_headers = [h.strip() for h in vary.split(',')]

        now = time.time()
        entry = CacheEntry(
            key=key,
            body=response.body,
            headers=dict(response.headers),
            status_code=response.status_code,
            created_at=now,
            expires_at=now + ttl,
            etag=response.headers.get('ETag'),
            last_modified=response.headers.get('Last-Modified'),
            vary_headers=vary_headers
        )

        # Evict if needed
        entry_size = len(entry.body)
        while self.current_size + entry_size > self.max_size:
            self._evict_one()

        self.cache[key] = entry
        self.current_size += entry_size

        return entry

    def _evict_one(self):
        '''Evict oldest entry (simple LRU would be better)'''
        if not self.cache:
            return

        oldest_key = min(self.cache.keys(),
                         key=lambda k: self.cache[k].created_at)
        entry = self.cache.pop(oldest_key)
        self.current_size -= len(entry.body)

    def invalidate(self, pattern: str):
        '''Invalidate cache entries matching pattern'''
        import fnmatch
        keys_to_remove = [
            k for k in self.cache.keys()
            if fnmatch.fnmatch(k, pattern)
        ]
        for key in keys_to_remove:
            entry = self.cache.pop(key)
            self.current_size -= len(entry.body)

    def conditional_get(self, entry: CacheEntry, request_headers: dict) -> bool:
        '''Check if conditional GET can return 304'''
        # If-None-Match (ETag)
        if_none_match = request_headers.get('If-None-Match')
        if if_none_match and entry.etag:
            if if_none_match == entry.etag or if_none_match == '*':
                return True

        # If-Modified-Since
        if_modified = request_headers.get('If-Modified-Since')
        if if_modified and entry.last_modified:
            # Parse and compare dates
            return if_modified == entry.last_modified

        return False
```
"""
                },
                "pitfalls": [
                    "Vary: * means never cache - handle this case",
                    "s-maxage vs max-age: CDN should use s-maxage",
                    "ETag weak vs strong: weak allows semantic equivalence",
                    "Cache key must include all Vary dimensions"
                ]
            },
            {
                "name": "Cache Invalidation",
                "description": "Implement purge, ban, and tag-based invalidation",
                "skills": ["Invalidation strategies", "Purge propagation", "Surrogate keys"],
                "hints": {
                    "level1": "Purge: remove specific URL; Ban: remove by pattern",
                    "level2": "Surrogate keys: tag content for group invalidation",
                    "level3": """
```python
from dataclasses import dataclass
from typing import Optional
import fnmatch
import time
import threading

@dataclass
class SurrogateKey:
    key: str
    cache_keys: set[str]

class CacheInvalidator:
    def __init__(self, cache: EdgeCache):
        self.cache = cache
        self.surrogate_keys: dict[str, set[str]] = {}  # surrogate -> cache_keys
        self.bans: list[tuple[str, float]] = []  # (pattern, timestamp)
        self.ban_lock = threading.Lock()

    def purge(self, url: str) -> bool:
        '''Purge specific URL from cache'''
        # Generate all possible cache keys for this URL
        # (simplified - would need to check all Vary combinations)
        key = self.cache.generate_cache_key(url, {})

        if key in self.cache.cache:
            entry = self.cache.cache.pop(key)
            self.cache.current_size -= len(entry.body)
            return True
        return False

    def purge_surrogate(self, surrogate_key: str) -> int:
        '''Purge all objects tagged with surrogate key'''
        cache_keys = self.surrogate_keys.get(surrogate_key, set())
        count = 0

        for cache_key in list(cache_keys):
            if cache_key in self.cache.cache:
                entry = self.cache.cache.pop(cache_key)
                self.cache.current_size -= len(entry.body)
                count += 1

        # Clear surrogate mapping
        if surrogate_key in self.surrogate_keys:
            del self.surrogate_keys[surrogate_key]

        return count

    def ban(self, pattern: str, ttl: int = 3600):
        '''Ban pattern - matching requests bypass cache'''
        with self.ban_lock:
            expires = time.time() + ttl
            self.bans.append((pattern, expires))

            # Also immediately remove matching entries
            self.cache.invalidate(pattern)

    def is_banned(self, url: str) -> bool:
        '''Check if URL matches any active ban'''
        now = time.time()

        with self.ban_lock:
            # Clean expired bans
            self.bans = [(p, e) for p, e in self.bans if e > now]

            for pattern, expires in self.bans:
                if fnmatch.fnmatch(url, pattern):
                    return True

        return False

    def register_surrogate(self, cache_key: str, surrogate_keys: list[str]):
        '''Register cache entry with surrogate keys'''
        for sk in surrogate_keys:
            if sk not in self.surrogate_keys:
                self.surrogate_keys[sk] = set()
            self.surrogate_keys[sk].add(cache_key)

    def parse_surrogate_header(self, header: str) -> list[str]:
        '''Parse Surrogate-Key header'''
        if not header:
            return []
        return [k.strip() for k in header.split()]

class CDNNode:
    def __init__(self, node_id: str, cache: EdgeCache,
                 invalidator: CacheInvalidator):
        self.node_id = node_id
        self.cache = cache
        self.invalidator = invalidator
        self.peers: list['CDNNode'] = []

    def propagate_purge(self, url: str):
        '''Propagate purge to all peers'''
        self.invalidator.purge(url)

        for peer in self.peers:
            # In production: async HTTP call to peer
            peer.invalidator.purge(url)

    def propagate_ban(self, pattern: str):
        '''Propagate ban to all peers'''
        self.invalidator.ban(pattern)

        for peer in self.peers:
            peer.invalidator.ban(pattern)
```
"""
                },
                "pitfalls": [
                    "Surrogate keys enable efficient invalidation of related content",
                    "Propagation delay: clients may get stale content briefly",
                    "Soft purge (stale-while-revalidate) better than hard purge",
                    "Bans can grow unbounded - implement TTL and cleanup"
                ]
            },
            {
                "name": "Origin Shield & Request Collapsing",
                "description": "Implement origin shielding and request collapsing to reduce origin load",
                "skills": ["Origin protection", "Request deduplication", "Thundering herd"],
                "hints": {
                    "level1": "Origin shield: single edge fetches from origin, others fetch from it",
                    "level2": "Request collapsing: one request to origin, all waiters get result",
                    "level3": """
```python
from dataclasses import dataclass
from typing import Optional, Callable
import threading
import time
from concurrent.futures import Future

@dataclass
class PendingRequest:
    key: str
    future: Future
    created_at: float

class RequestCollapser:
    '''Collapse multiple requests for same resource into one'''

    def __init__(self, timeout: float = 30.0):
        self.pending: dict[str, PendingRequest] = {}
        self.lock = threading.Lock()
        self.timeout = timeout

    def get_or_fetch(self, key: str,
                     fetch_fn: Callable[[], 'Response']) -> 'Response':
        '''Get from pending request or create new one'''

        with self.lock:
            # Check if request already pending
            if key in self.pending:
                pending = self.pending[key]
                # Wait for result
                return pending.future.result(timeout=self.timeout)

            # Create new pending request
            future = Future()
            self.pending[key] = PendingRequest(
                key=key,
                future=future,
                created_at=time.time()
            )

        # Fetch outside lock
        try:
            response = fetch_fn()
            future.set_result(response)
            return response

        except Exception as e:
            future.set_exception(e)
            raise

        finally:
            with self.lock:
                if key in self.pending:
                    del self.pending[key]

class OriginShield:
    '''Shield origin from direct edge requests'''

    def __init__(self, shield_cache: EdgeCache, origin_url: str):
        self.cache = shield_cache
        self.origin_url = origin_url
        self.collapser = RequestCollapser()
        self.request_count = 0
        self.origin_requests = 0

    def fetch(self, path: str, headers: dict) -> 'Response':
        '''Fetch from shield cache or origin'''
        self.request_count += 1

        cache_key = self.cache.generate_cache_key(path, {})

        # Check shield cache
        entry = self.cache.get(cache_key)
        if entry:
            return Response(
                status_code=entry.status_code,
                headers=entry.headers,
                body=entry.body
            )

        # Fetch from origin with request collapsing
        def fetch_origin():
            self.origin_requests += 1
            import httpx
            response = httpx.get(
                f"{self.origin_url}{path}",
                headers=headers,
                timeout=30.0
            )
            return Response(
                status_code=response.status_code,
                headers=dict(response.headers),
                body=response.content
            )

        response = self.collapser.get_or_fetch(cache_key, fetch_origin)

        # Cache response
        self.cache.put(cache_key, response)

        return response

    def get_stats(self) -> dict:
        return {
            'total_requests': self.request_count,
            'origin_requests': self.origin_requests,
            'shield_hit_rate': 1 - (self.origin_requests / max(1, self.request_count))
        }

class CDNEdge:
    '''Edge node with shield support'''

    def __init__(self, edge_id: str, local_cache: EdgeCache,
                 shield: OriginShield, invalidator: CacheInvalidator):
        self.edge_id = edge_id
        self.cache = local_cache
        self.shield = shield
        self.invalidator = invalidator
        self.collapser = RequestCollapser()

    def handle_request(self, request: 'Request') -> 'Response':
        '''Handle incoming request'''
        url = request.url

        # Check bans
        if self.invalidator.is_banned(url):
            return self._fetch_from_shield(request)

        # Generate cache key
        cache_key = self.cache.generate_cache_key(url, {})

        # Check local cache
        entry = self.cache.get(cache_key)
        if entry:
            # Check conditional request
            if self.cache.conditional_get(entry, request.headers):
                return Response(
                    status_code=304,
                    headers={'ETag': entry.etag},
                    body=b''
                )

            return Response(
                status_code=entry.status_code,
                headers=entry.headers,
                body=entry.body
            )

        # Fetch from shield (with collapsing)
        def fetch():
            return self.shield.fetch(url, request.headers)

        response = self.collapser.get_or_fetch(cache_key, fetch)

        # Cache locally
        entry = self.cache.put(cache_key, response)
        if entry:
            # Register surrogate keys
            surrogate_header = response.headers.get('Surrogate-Key')
            if surrogate_header:
                keys = self.invalidator.parse_surrogate_header(surrogate_header)
                self.invalidator.register_surrogate(cache_key, keys)

        return response

    def _fetch_from_shield(self, request: 'Request') -> 'Response':
        return self.shield.fetch(request.url, request.headers)
```
"""
                },
                "pitfalls": [
                    "Request collapsing timeout must be shorter than client timeout",
                    "Shield adds latency but dramatically reduces origin load",
                    "Negative caching: cache 404s briefly to prevent origin storms",
                    "Health checks: bypass shield when origin is unhealthy"
                ]
            }
        ]
    }
}

# Load and update
with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)

if 'expert_projects' not in data:
    data['expert_projects'] = {}

for project_id, project in content_media_projects.items():
    data['expert_projects'][project_id] = project
    print(f"Added: {project_id} - {project['name']}")

with open(yaml_path, 'w') as f:
    yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False, width=120)

print(f"\nAdded {len(content_media_projects)} Content & Media projects")
