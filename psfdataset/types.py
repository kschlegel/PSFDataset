from typing import Tuple, Dict, Union

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

import numpy as np

KeypointLabelPair = Tuple[np.ndarray, int]
DescriptionDict = Dict[str, Union[str, int, float, bool]]


class KeypointTransformation(Protocol):
    def __call__(self, sample: np.ndarray) -> np.ndarray:
        ...

    def get_description(self) -> DescriptionDict:
        ...
