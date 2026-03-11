"""
Weight management for DeepTMHMM 2.

On first use, checkpoint files are downloaded from the GitHub repository
and cached in ~/.cache/deeptmhmm2/. Subsequent runs skip the download.

To update weights to a new version, change WEIGHTS_REF below to a new
commit SHA or tag, or clear ~/.cache/deeptmhmm2/ to force a re-download.
"""

import os
import urllib.request
from pathlib import Path
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────
# Point at the GitHub repo and a specific ref (commit SHA or tag) so the URLs
# are permanently stable even if the branch moves forward.
_GITHUB_REPO = "fteufel/DeepTMHMM2"   # ← fill in once
_WEIGHTS_REF  = "main"                             # ← ideally pin to a tag/SHA

_BASE_URL = f"https://raw.githubusercontent.com/{_GITHUB_REPO}/{_WEIGHTS_REF}/checkpoints"

_TOPOLOGY_FILES = [f"topology_{i}.ckpt" for i in range(5)]
_MEMTYPE_FILES  = [f"memtype_{i}.ckpt"  for i in range(5)]
# ─────────────────────────────────────────────────────────────────────────────


def get_cache_dir() -> Path:
    """Return (and create) the cache directory for model weights."""
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    d = base / "deeptmhmm2"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _download(url: str, dest: Path) -> None:
    """Download *url* to *dest* with a tqdm progress bar."""
    print(f"Downloading {dest.name} ...")
    with tqdm(unit="B", unit_scale=True, unit_divisor=1024,
              miniters=1, desc=dest.name, leave=False) as t:
        def _hook(count, block_size, total_size):
            if total_size > 0:
                t.total = total_size
            t.update(block_size)
        urllib.request.urlretrieve(url, dest, reporthook=_hook)


def ensure_weights() -> tuple:
    """
    Return ``(topology_ckpt_paths, memtype_ckpt_paths)`` as lists of strings.

    Any file not already present in ``~/.cache/deeptmhmm2/`` is downloaded
    from GitHub before the paths are returned.
    """
    cache = get_cache_dir()
    topo_paths, mem_paths = [], []

    for fname in _TOPOLOGY_FILES:
        dest = cache / fname
        if not dest.exists():
            _download(f"{_BASE_URL}/{fname}", dest)
        topo_paths.append(str(dest))

    for fname in _MEMTYPE_FILES:
        dest = cache / fname
        if not dest.exists():
            _download(f"{_BASE_URL}/{fname}", dest)
        mem_paths.append(str(dest))

    return topo_paths, mem_paths
