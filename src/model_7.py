from __future__ import annotations
import collections
from dataclasses import dataclass, asdict
from typing import Optional, Counter, List
import weakref
import sys

@dataclass
class Sample:
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@dataclass
class KnownSample(Sample):
    species: str


@dataclass
class TestingKnownSample(KnownSample):
    classification: Optional[str] = None


@dataclass
class TrainingKnownSample(KnownSample):
    pass

@dataclass
class UnknownSample(Sample):
    classification: Optional[str] = None

class Distance:
    def distance(self, s1:Sample, s2:Sample) -> float:
        raise NotImplementedError
    

@dataclass
class Hyperparameters:
    k: int
    algo: Distance
    data: weakref.ReferenceType["TrainingData"]

    def classify(self, sample: Sample) -> str:
        if not (trainig_data := self.data()):
            raise RuntimeError("No TrainingData object")
        distances = list[tuple[float,TrainingKnownSample]]=sorted(
            (self.algo.distance(sample, known), known)
            for known in trainig_data.training
        )
        k_nearest = (known.species for d, known in distances[: self.k])
        frequency: Counter[str] = collections.Counter(k_nearest)
        best_fit, *others = frequency.most_common()
        species, votes = best_fit
        return species
    
@dataclass
class TrainingData:
    testing: List[TestingKnownSample]
    training: List[TrainingKnownSample]
    tuning: List[Hyperparameters]


__test__ = { names: case for names, case in locals().items() if names.startswith("test_")}