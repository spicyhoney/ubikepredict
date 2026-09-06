from __future__ import annotations

"""Materialise the frozen demo assets from private S3 buckets.

The module deliberately imports boto3 only when a real S3 client is needed, so
the verified local demo and unit tests continue to work without AWS packages or
credentials.  Every cloud object has a fixed key, byte size, and SHA256 digest;
an object is never exposed to the application until all three checks pass.
"""

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_ROOT = Path("/tmp/ubikepredict")
DEFAULT_RELEASE_ID = "frozen-v1"
_SAFE_RELEASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class AssetError(RuntimeError):
    """Base class for deterministic asset preparation failures."""


class AssetConfigurationError(AssetError):
    """Raised when an AWS asset setting is missing or unsafe."""


class AssetIntegrityError(AssetError):
    """Raised when a downloaded asset differs from the frozen manifest."""


@dataclass(frozen=True)
class AssetSpec:
    name: str
    relative_path: str
    s3_key_template: str
    bytes: int
    sha256: str
    env_var: str
    scope: str

    def s3_key(self, release_id: str) -> str:
        return self.s3_key_template.format(release_id=release_id)


# This is intentionally an in-code allow-list rather than metadata downloaded
# from S3.  The values are copied from the repository's audited MANIFEST.json.
MANIFEST: Mapping[str, AssetSpec] = {
    "model": AssetSpec(
        name="model",
        relative_path="model/lgbm_full.txt",
        s3_key_template="releases/{release_id}/model/lgbm_full.txt",
        bytes=1_237_897,
        sha256="d2e8ce0738caa0544531d4ac880bab8c1983fbf52686683631cb65475b5d5cf8",
        env_var="UBIKE_MODEL_PATH",
        scope="prediction",
    ),
    "freeze": AssetSpec(
        name="freeze",
        relative_path="config/final_policy_freeze_before_may.json",
        s3_key_template="releases/{release_id}/config/final_policy_freeze_before_may.json",
        bytes=9_282,
        sha256="fef97defc8935a499831f12f1f42dd9d153b1075b61c50f3b57219c0bbccc714",
        env_var="UBIKE_FREEZE_PATH",
        scope="prediction",
    ),
    "protocol": AssetSpec(
        name="protocol",
        relative_path="config/protocol_frozen_before_june.json",
        s3_key_template="releases/{release_id}/config/protocol_frozen_before_june.json",
        bytes=27_099,
        sha256="a65243c5ff986f57b0fb53f26a7649ccbc130a74765890788fe6196640c9825b",
        env_var="UBIKE_PROTOCOL_PATH",
        scope="prediction",
    ),
    "input": AssetSpec(
        name="input",
        relative_path="data/source/dynamic_red_empty_2026_06_input.parquet",
        s3_key_template=(
            "replays/2026-06/input/dynamic_red_empty_2026_06_input.parquet"
        ),
        bytes=2_861_200,
        sha256="b1cb8fdffb1832b0d588bd5d821625cc92f8806d190b15dd5648e36f267af009",
        env_var="UBIKE_API_INPUT_PATH",
        scope="prediction",
    ),
    "full_dock_model": AssetSpec(
        name="full_dock_model",
        relative_path="model/lgbm_full_dock.txt",
        s3_key_template="releases/{release_id}/model/lgbm_full_dock.txt",
        bytes=532_851,
        sha256="730bb702e8c412c66311f046f37d45f8af0c8fc188c6df684b3f7a3fb033a063",
        env_var="UBIKE_FULL_DOCK_MODEL_PATH",
        scope="prediction",
    ),
    "full_dock_freeze": AssetSpec(
        name="full_dock_freeze",
        relative_path="config/final_policy_freeze_full_dock_before_may.json",
        s3_key_template=(
            "releases/{release_id}/config/final_policy_freeze_full_dock_before_may.json"
        ),
        bytes=28_347,
        sha256="02fde25d6833d046ad92c6325e6ac60cc8659ec78c09e51759c48789722e0694",
        env_var="UBIKE_FULL_DOCK_FREEZE_PATH",
        scope="prediction",
    ),
    "full_dock_protocol": AssetSpec(
        name="full_dock_protocol",
        relative_path="config/protocol_full_dock_frozen_before_june.json",
        s3_key_template=(
            "releases/{release_id}/config/protocol_full_dock_frozen_before_june.json"
        ),
        bytes=21_081,
        sha256="dc5cee1e9db099bda81918a218a65c61f3878b95b53651b5611b98efd601da5f",
        env_var="UBIKE_FULL_DOCK_PROTOCOL_PATH",
        scope="prediction",
    ),
    "full_dock_input": AssetSpec(
        name="full_dock_input",
        relative_path="data/source/dynamic_red_full_2026_06_input.parquet",
        s3_key_template=(
            "replays/2026-06/input/dynamic_red_full_2026_06_input.parquet"
        ),
        bytes=507_827,
        sha256="0bfb7fcf13c7f4e1bc0448ab670d6ef89d4f27e849080b91a850557c2fa7774b",
        env_var="UBIKE_FULL_DOCK_API_INPUT_PATH",
        scope="prediction",
    ),
    "stations": AssetSpec(
        name="stations",
        relative_path="data/stations/dim_station.csv",
        s3_key_template="stations/dim_station.csv",
        bytes=193_983,
        sha256="2ea9d705da0d3508fd7e89009398b8b272188937c1e5f7209d0748cb1998d81f",
        env_var="UBIKE_STATIONS_PATH",
        scope="prediction",
    ),
    "truth": AssetSpec(
        name="truth",
        relative_path="data/reference/june_all_eligible_decisions.parquet",
        s3_key_template=(
            "replays/2026-06/truth/june_all_eligible_decisions.parquet"
        ),
        bytes=500_714,
        sha256="5f725b36965e41026679772dd60d79734e939b7a17025b6623f8385727660385",
        env_var="UBIKE_REFERENCE_PATH",
        scope="reveal",
    ),
    "full_dock_truth": AssetSpec(
        name="full_dock_truth",
        relative_path="data/reference/june_full_dock_all_eligible_decisions.parquet",
        s3_key_template=(
            "replays/2026-06/truth/june_full_dock_all_eligible_decisions.parquet"
        ),
        bytes=92_366,
        sha256="e89486879bd6bd8e7872b69da6890c51c2b0b864cbead01c9db1f509b453d73f",
        env_var="UBIKE_FULL_DOCK_REFERENCE_PATH",
        scope="reveal",
    ),
}

PREDICTION_ASSETS = tuple(
    name for name, spec in MANIFEST.items() if spec.scope == "prediction"
)
REVEAL_ASSETS = tuple(name for name, spec in MANIFEST.items() if spec.scope == "reveal")

LOCAL_DEFAULTS: Mapping[str, str] = {
    spec.env_var: spec.relative_path for spec in MANIFEST.values()
}


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_asset(path: Path, spec: AssetSpec) -> None:
    """Raise a stable error if ``path`` is not exactly the frozen asset."""
    if not path.is_file():
        raise AssetIntegrityError(f"Missing frozen asset: {spec.name}")
    actual_bytes = path.stat().st_size
    if actual_bytes != spec.bytes:
        raise AssetIntegrityError(
            f"Frozen asset size mismatch for {spec.name}: "
            f"expected {spec.bytes}, got {actual_bytes}"
        )
    actual_hash = sha256_file(path)
    if actual_hash != spec.sha256:
        raise AssetIntegrityError(
            f"Frozen asset SHA256 mismatch for {spec.name}: "
            f"expected {spec.sha256}, got {actual_hash}"
        )


def _validate_release_id(value: str) -> str:
    release_id = value.strip()
    if not _SAFE_RELEASE_ID.fullmatch(release_id):
        raise AssetConfigurationError(
            "UBIKE_RELEASE_ID may contain only letters, digits, dot, underscore, and hyphen"
        )
    return release_id


def _validate_manifest(manifest: Mapping[str, AssetSpec]) -> None:
    for name, spec in manifest.items():
        if name != spec.name or spec.scope not in {"prediction", "reveal"}:
            raise AssetConfigurationError(f"Invalid asset manifest entry: {name}")
        if spec.bytes < 0 or not re.fullmatch(r"[0-9a-f]{64}", spec.sha256):
            raise AssetConfigurationError(f"Invalid frozen checksum metadata: {name}")
        relative = Path(spec.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise AssetConfigurationError(f"Unsafe asset cache path: {name}")


class S3AssetLoader:
    """Download allow-listed assets into a verified local cache."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        cache_root: Path | str = DEFAULT_CACHE_ROOT,
        manifest: Mapping[str, AssetSpec] = MANIFEST,
    ) -> None:
        _validate_manifest(manifest)
        self._client = client
        self.cache_root = Path(cache_root)
        self.manifest = manifest

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import boto3  # type: ignore[import-not-found]
                from botocore.config import Config  # type: ignore[import-not-found]
            except ImportError as error:  # pragma: no cover - Lambda image supplies boto3
                raise AssetConfigurationError(
                    "boto3 is required only when UBIKE_RUNTIME_BUCKET or "
                    "UBIKE_TRUTH_BUCKET is configured"
                ) from error
            try:
                timeout = float(os.environ.get("UBIKE_S3_TIMEOUT_SECONDS", "8"))
            except ValueError as error:
                raise AssetConfigurationError(
                    "UBIKE_S3_TIMEOUT_SECONDS must be a number"
                ) from error
            if not 1 <= timeout <= 15:
                raise AssetConfigurationError(
                    "UBIKE_S3_TIMEOUT_SECONDS must be between 1 and 15"
                )
            self._client = boto3.client(
                "s3",
                config=Config(
                    connect_timeout=min(timeout, 2.0),
                    read_timeout=timeout,
                    retries={"max_attempts": 1, "mode": "standard"},
                ),
            )
        return self._client

    def materialise(
        self,
        *,
        bucket: str,
        asset_names: Sequence[str],
        release_id: str = DEFAULT_RELEASE_ID,
    ) -> dict[str, Path]:
        bucket = bucket.strip()
        if not bucket:
            raise AssetConfigurationError("S3 bucket name may not be empty")
        release_id = _validate_release_id(release_id)
        results: dict[str, Path] = {}
        for name in asset_names:
            try:
                spec = self.manifest[name]
            except KeyError as error:
                raise AssetConfigurationError(f"Unknown frozen asset: {name}") from error
            results[name] = self._materialise_one(bucket, spec, release_id)
        return results

    # American spelling kept as a convenience for AWS-oriented callers.
    materialize = materialise

    def _materialise_one(
        self,
        bucket: str,
        spec: AssetSpec,
        release_id: str,
    ) -> Path:
        destination = self.cache_root.joinpath(*Path(spec.relative_path).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            try:
                verify_asset(destination, spec)
                return destination
            except AssetIntegrityError:
                # A partial/stale /tmp file is never reused.  It is replaced
                # atomically only after the new object passes verification.
                pass

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".part",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)

            key = spec.s3_key(release_id)
            if hasattr(self.client, "download_file"):
                self.client.download_file(bucket, key, str(temporary_path))
            else:
                response = self.client.get_object(Bucket=bucket, Key=key)
                body = response["Body"]
                with temporary_path.open("wb") as output:
                    for block in iter(lambda: body.read(1024 * 1024), b""):
                        output.write(block)

            verify_asset(temporary_path, spec)
            os.replace(temporary_path, destination)
            temporary_path = None
            return destination
        except AssetError:
            raise
        except Exception as error:
            raise AssetError(f"Unable to download frozen asset {spec.name}") from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


def _resolve_local_assets(
    names: Sequence[str], environ: Mapping[str, str]
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name in names:
        spec = MANIFEST[name]
        configured = Path(environ.get(spec.env_var, LOCAL_DEFAULTS[spec.env_var]))
        paths[name] = configured if configured.is_absolute() else REPO_ROOT / configured
    return paths


def _export_paths(
    paths: Mapping[str, Path], environ: MutableMapping[str, str]
) -> dict[str, Path]:
    for name, path in paths.items():
        environ[MANIFEST[name].env_var] = str(path)
    return dict(paths)


def prepare_prediction_assets(
    *,
    bucket: str | None = None,
    release_id: str | None = None,
    client: Any | None = None,
    cache_root: Path | str = DEFAULT_CACHE_ROOT,
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, Path]:
    """Prepare truth-free model assets, or return local paths without AWS."""
    target_environ = environ if environ is not None else os.environ
    runtime_bucket = bucket if bucket is not None else target_environ.get("UBIKE_RUNTIME_BUCKET")
    if not runtime_bucket:
        return _resolve_local_assets(PREDICTION_ASSETS, target_environ)
    loader = S3AssetLoader(client=client, cache_root=cache_root)
    paths = loader.materialise(
        bucket=runtime_bucket,
        asset_names=PREDICTION_ASSETS,
        release_id=release_id
        or target_environ.get("UBIKE_RELEASE_ID", DEFAULT_RELEASE_ID),
    )
    return _export_paths(paths, target_environ)


def prepare_reveal_assets(
    *,
    bucket: str | None = None,
    client: Any | None = None,
    cache_root: Path | str = DEFAULT_CACHE_ROOT,
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, Path]:
    """Prepare the isolated truth asset, or return its local path without AWS."""
    target_environ = environ if environ is not None else os.environ
    truth_bucket = bucket if bucket is not None else target_environ.get("UBIKE_TRUTH_BUCKET")
    if not truth_bucket:
        return _resolve_local_assets(REVEAL_ASSETS, target_environ)
    loader = S3AssetLoader(client=client, cache_root=cache_root)
    paths = loader.materialise(
        bucket=truth_bucket,
        asset_names=REVEAL_ASSETS,
        release_id=DEFAULT_RELEASE_ID,
    )
    return _export_paths(paths, target_environ)


__all__ = [
    "AssetConfigurationError",
    "AssetError",
    "AssetIntegrityError",
    "AssetSpec",
    "DEFAULT_CACHE_ROOT",
    "DEFAULT_RELEASE_ID",
    "MANIFEST",
    "PREDICTION_ASSETS",
    "REVEAL_ASSETS",
    "S3AssetLoader",
    "prepare_prediction_assets",
    "prepare_reveal_assets",
    "sha256_file",
    "verify_asset",
]
