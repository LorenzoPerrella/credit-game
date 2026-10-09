"""Where `models.engine` used to live, kept so that the fits cached under that name still read.

**This is not dead code, and deleting it empties the fit cache.** The engine was one module of
2,325 lines and is now the `models/engine/` package; a pickle written before that split names
`creditsurv.models.blocks.BlockFit`, and pickle resolves a class by importing the module it was
written from. Without this file every one of the 176 fits on disk raises
`ModuleNotFoundError`, `load_fit` turns that into a miss, and a run starts cold -- which is
exactly the hours the cache exists to protect. `tests/test_blocks.py` holds the name to the
class so that nothing tidies this away.

`BlockFit` is the only name the pickles ask for: a record of what a fit saw and what it cost,
counted from the scan. Nothing new should import from here.
"""

from __future__ import annotations

from creditsurv.models.engine.fit import BlockFit

__all__ = ["BlockFit"]
