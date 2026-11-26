from dataclasses import dataclass, field

from .base import Manipulation, ManipulationName
from .manips import (
    NoopManipulation,
    NoopManipConfig,
    TailPruning,
    TailPruningConfig,
    TokenSkip,
    TokenSkipConfig,
)


@dataclass
class ManipulationConfigs:
    noop: NoopManipConfig = field(default_factory=NoopManipConfig)
    token_skip: TokenSkipConfig = field(default_factory=TokenSkipConfig)
    tail_pruning: TailPruningConfig = field(default_factory=TailPruningConfig)


def load_manipulation(
    name: ManipulationName, cfg: ManipulationConfigs, *args, **kwargs
) -> Manipulation:
    cfg = getattr(cfg, name.value.lower(), None)
    if name == ManipulationName.MISSING:
        raise ValueError(f"Missing transform manipulation: {name}")
    elif name == ManipulationName.NOOP:
        return NoopManipulation(name, cfg=cfg, *args, **kwargs)
    elif name == ManipulationName.TOKEN_SKIP:
        return TokenSkip(name, cfg=cfg, *args, **kwargs)
    elif name == ManipulationName.TAIL_PRUNING:
        return TailPruning(name, cfg=cfg, *args, **kwargs)
    else:
        raise ValueError(f"Unsupported transform manipulation type: {name}")
