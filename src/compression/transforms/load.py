from dataclasses import dataclass

from .base import Transform, TransformName
from .plain import PlainConfig, PlainTransform
from .forced_cot import ForcedCoTConfig, ForcedCoTTransform
from .zero_cot import ZeroCoTConfig, ZeroCoTTransform


@dataclass
class TransformConfigs:
    plain: PlainConfig = PlainConfig()
    zero_cot: ZeroCoTConfig = ZeroCoTConfig()
    forced_cot: ForcedCoTConfig = ForcedCoTConfig()


def load_transform(
    name: TransformName, cfg: TransformConfigs, *args, **kwargs
) -> Transform:
    cfg = getattr(cfg, name.value.lower(), None)
    if name == TransformName.MISSING:
        raise ValueError(f"Missing runtime transform: {name}")
    elif name == TransformName.PLAIN:
        return PlainTransform(name, cfg=cfg, *args, **kwargs)
    elif name == TransformName.ZERO_COT:
        return ZeroCoTTransform(name, cfg=cfg, *args, **kwargs)
    elif name == TransformName.FORCED_COT:
        return ForcedCoTTransform(name, cfg=cfg, *args, **kwargs)
    else:
        raise ValueError(f"Unsupported transform type: {name}")
