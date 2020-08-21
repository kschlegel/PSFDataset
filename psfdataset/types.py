from typing import Tuple, Dict, Union, List, TYPE_CHECKING

try:
    from typing import Protocol  # type: ignore
except ImportError:
    from typing_extensions import Protocol  # type: ignore

import numpy as np

if TYPE_CHECKING:
    from .psfdataset import PSFDataset
    from .psfzippeddataset import PSFZippedDataset

KeypointLabelPair = Tuple[np.ndarray, int]
DescriptionDict = Dict[str, Union[str, int, float, bool]]
StructureElement = Union[List[str], int, str]
StructureDescription = List[StructureElement]

PSFDatasetObject = Union["PSFDataset", "PSFZippedDataset"]


class KeypointTransformation(Protocol):
    def __call__(self, sample: np.ndarray) -> np.ndarray:
        ...

    def get_description(self) -> DescriptionDict:
        ...

    def explain(self,
                input_structure: StructureDescription) -> StructureDescription:
        ...
