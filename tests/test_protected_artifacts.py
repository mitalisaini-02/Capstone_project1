"""
Verifies that checkpoints, scalers, configs and recorded results still match
the Phase 1 SHA-256 snapshot (outputs/PROTECTED_ARTIFACTS.sha256).
"""
import os
import unittest

from src.utils.integrity import verify_manifest

MANIFEST = "outputs/PROTECTED_ARTIFACTS.sha256"


@unittest.skipUnless(os.path.exists(MANIFEST), "no protected-artifact manifest")
class TestProtectedArtifacts(unittest.TestCase):

    def test_hashes_unchanged(self):
        changed, missing = verify_manifest(MANIFEST)
        self.assertEqual(changed, [], f"Protected files modified: {changed}")
        self.assertEqual(missing, [], f"Protected files missing: {missing}")


if __name__ == "__main__":
    unittest.main()
