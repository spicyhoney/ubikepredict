from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from backend.s3_assets import (
    AssetConfigurationError,
    AssetIntegrityError,
    AssetSpec,
    MANIFEST,
    PREDICTION_ASSETS,
    REVEAL_ASSETS,
    S3AssetLoader,
    prepare_prediction_assets,
    verify_asset,
)


class FakeS3Client:
    def __init__(self, objects: dict[tuple[str, str], bytes]) -> None:
        self.objects = objects
        self.calls: list[tuple[str, str]] = []

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        self.calls.append((bucket, key))
        Path(filename).write_bytes(self.objects[(bucket, key)])


def spec_for(content: bytes) -> AssetSpec:
    return AssetSpec(
        name="tiny",
        relative_path="model/tiny.bin",
        s3_key_template="releases/{release_id}/model/tiny.bin",
        bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        env_var="TINY_PATH",
        scope="prediction",
    )


class S3AssetLoaderTest(unittest.TestCase):
    def test_builtin_manifest_matches_repository_assets(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for spec in MANIFEST.values():
            verify_asset(root / spec.relative_path, spec)

    def test_builtin_manifest_separates_truth_from_prediction(self) -> None:
        self.assertEqual(set(PREDICTION_ASSETS), {
            "model",
            "freeze",
            "protocol",
            "input",
            "full_dock_model",
            "full_dock_freeze",
            "full_dock_protocol",
            "full_dock_input",
            "stations",
        })
        self.assertEqual(set(REVEAL_ASSETS), {"truth", "full_dock_truth"})
        self.assertTrue(all(MANIFEST[name].scope == "prediction" for name in PREDICTION_ASSETS))
        self.assertTrue(all(MANIFEST[name].scope == "reveal" for name in REVEAL_ASSETS))

    def test_deployment_allow_lists_every_manifest_asset(self) -> None:
        root = Path(__file__).resolve().parents[1]
        template = (root / "infra/template.yaml").read_text(encoding="utf-8")
        deploy_script = (root / "scripts/deploy-aws.ps1").read_text(encoding="utf-8")
        for spec in MANIFEST.values():
            cloudformation_key = spec.s3_key_template.replace(
                "{release_id}", "${ReleaseId}"
            )
            self.assertIn(cloudformation_key, template)
            self.assertIn(spec.relative_path.replace("/", "\\"), deploy_script)

    def test_download_verifies_and_reuses_valid_cache(self) -> None:
        content = b"frozen model bytes"
        spec = spec_for(content)
        key = "releases/frozen-v1/model/tiny.bin"
        client = FakeS3Client({("runtime", key): content})
        with tempfile.TemporaryDirectory() as directory:
            loader = S3AssetLoader(
                client=client,
                cache_root=directory,
                manifest={"tiny": spec},
            )
            first = loader.materialise(
                bucket="runtime", asset_names=("tiny",), release_id="frozen-v1"
            )["tiny"]
            second = loader.materialize(
                bucket="runtime", asset_names=("tiny",), release_id="frozen-v1"
            )["tiny"]

            self.assertEqual(first.read_bytes(), content)
            self.assertEqual(second, first)
            self.assertEqual(client.calls, [("runtime", key)])

    def test_corrupt_download_is_not_promoted_to_cache(self) -> None:
        expected = b"expected"
        spec = spec_for(expected)
        key = "releases/frozen-v1/model/tiny.bin"
        client = FakeS3Client({("runtime", key): b"wrong"})
        with tempfile.TemporaryDirectory() as directory:
            loader = S3AssetLoader(
                client=client,
                cache_root=directory,
                manifest={"tiny": spec},
            )
            with self.assertRaises(AssetIntegrityError):
                loader.materialise(bucket="runtime", asset_names=("tiny",))
            self.assertFalse((Path(directory) / spec.relative_path).exists())
            self.assertEqual(list(Path(directory).rglob("*.part")), [])

    def test_invalid_release_id_cannot_change_fixed_prefix(self) -> None:
        spec = spec_for(b"ok")
        with tempfile.TemporaryDirectory() as directory:
            loader = S3AssetLoader(
                client=FakeS3Client({}),
                cache_root=directory,
                manifest={"tiny": spec},
            )
            with self.assertRaises(AssetConfigurationError):
                loader.materialise(
                    bucket="runtime", asset_names=("tiny",), release_id="../truth"
                )

    def test_local_mode_requires_neither_boto3_nor_credentials(self) -> None:
        environ: dict[str, str] = {}
        paths = prepare_prediction_assets(environ=environ)
        self.assertEqual(set(paths), set(PREDICTION_ASSETS))
        self.assertTrue(all(path.is_absolute() for path in paths.values()))
        self.assertNotIn("UBIKE_RUNTIME_BUCKET", environ)


if __name__ == "__main__":
    unittest.main()
