"""Run only explicitly data-independent tests. Full tests still require assets."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
loader = unittest.TestLoader()
suite = loader.loadTestsFromName('test_bedrock_summary')
# Only this one S3 integrity test needs the private frozen asset files.
import test_s3_assets
for name in loader.getTestCaseNames(test_s3_assets.S3AssetLoaderTest):
    if name != 'test_builtin_manifest_matches_repository_assets':
        suite.addTest(test_s3_assets.S3AssetLoaderTest(name))
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
