from coma import command

from ...io import PathConfig

from .base import GeneralConfig


@command(name="analyze")
class Analyze:
    def __init__(self, general: GeneralConfig, path: PathConfig):
        self.general = general
        self.path = path

    def run(self):
        raise NotImplementedError
