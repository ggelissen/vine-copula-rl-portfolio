# Repository hygiene policy

This project separates three classes of material:

1. **Live source**: current R/Python code, contracts, tests, and manuscript
   source. This belongs in Git.
2. **Immutable scientific evidence**: frozen input/release/result archives,
   checksums, manifests, and exact policy checkpoints needed for replay. These
   must remain content-addressed and read-only. Large files belong in Git LFS or
   a DOI-bearing artifact repository.
3. **Regenerable work products**: caches, failed runs, smoke outputs, LaTeX
   intermediates, duplicate archives, and unfrozen generated data. These do not
   belong in Git and may be deleted after their canonical replacement is
   verified.

`local_cleanup_manifest.csv` records every current local cleanup target before
deletion. `cleanup_local_generated_artifacts.ps1` is fail-closed: it reads only
manifest rows marked `delete_local`, resolves every path, and refuses to act
outside the workspace.

The server-side v2/v3/v4 causal run trees are **not ordinary clutter**. They
contain the exact 70/31/29 checkpoints used by the frozen causal result. First
run `freeze_causal_checkpoint_release.py`, verify both the release inventory and
archive checksum, copy the archive off HPC, and only then delete the original
revision-specific run trees.

The local `.git` object database is currently corrupt. This cleanup deliberately
does not touch `.git`. Re-clone the remote repository after preserving any
uncommitted source changes; do not attempt destructive Git repair in place.
