import rawpy
from typing import Any, ContextManager, Tuple
from negpy.domain.interfaces import IImageLoader
from negpy.infrastructure.loaders.helpers import detect_color_space_from_raw


class RawpyLoader(IImageLoader):
    """
    Standard RAW loader (libraw).
    """

    def load(self, file_path: str) -> Tuple[ContextManager[Any], dict]:
        raw = rawpy.imread(file_path)

        metadata = {
            "orientation": 0,
            "raw_flip": 0,
            "color_space": detect_color_space_from_raw(raw) or "Adobe RGB",
        }

        return raw, metadata
