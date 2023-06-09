from __future__ import annotations
from typing import Optional, List, Iterable, cast, Tuple,Any, Dict, Counter,Iterator,TypedDict,overload
import datetime
from math import hypot
from abc import ABC, abstractmethod
import enum
from pathlib import Path
import abc
import csv
import sys
from math import isclose
import random
import itertools
import json
import jsonschema
import yaml
import weakref

class Sample:
    """모든 샘플의 추상 슈퍼 클래스"""

    def __init__(
        self,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
        species: Optional[str] = None,
    ) -> None:
        self.sepal_length = sepal_length
        self.sepal_width = sepal_width
        self.petal_length = petal_length
        self.petal_width = petal_width
        self.species = species
        self.classification: Optional[str] = None

    def hash(self) -> int:
        return(
            sum(
                [
                    hash(self.sepal_length),
                    hash(self.sepal_width),
                    hash(self.petal_length),
                    hash(self.petal_width),
                ]
            )
            % sys.hash_info.modulus
        )
    def __eq__(self, other: any) -> bool:
        if type(other) != type(self):
            return False
        other = cast(Sample,other)
        return all(
            [
                self.sepal_length == other.sepal_length,
                self.sepal_width == other.sepal_width,
                self.petal_length == other.petal_length,
                self.petal_width == other.petal_width,
            ]
        )

    @property
    def attr_dict(self) -> dict[str,str]:
        return dict(
            sepal_length=f"{self.sepal_length!r}",
            sepal_width=f"{self.sepal_width!r}",
            petal_length=f"{self.petal_length!r}",
            petal_width=f"{self.petal_width!r}",
        )

    def __repr__(self) -> str:
        base_attirubutes = self.attr_dict
        attrs = ",".join(f"{k}={v}" for k, v in base_attirubutes.items())
        return f"{self.__class__.__name__}({attrs})"
        

class Purpose(enum.IntEnum):
    Classification = 0
    Testing = 1
    Training = 2
    

class KnownSample(Sample):
    """테스트, 학습 데이터를 위한 추상클래스, 종은 외부에서 설정"""
    def __init__(
        self,
        species: str,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
    ) -> None:
        purpose_enum = Purpose(self.purpose)
        if purpose_enum not in{Purpose.Testing, Purpose.Training}:
            raise ValueError(f"Invalid purpose: {purpose_enum}")
        super().__init__(
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width,
        )
        self.purpose = purpose_enum
        self.species = species
        self._classification: Optional[str] = None

    def matches(self) -> bool:
        return self.species == self.classification
    
    @property
    def classification(self) -> Optional[str]:
        if self.purpose == Purpose.Testing:
            return self._classification
        else:
            raise AttributeError(f"Training sample has no classification")
        
    @classification.setter
    def classification(self, value: str) -> None:
        if self.purpose == Purpose.Testing:
            self._classification = value
        else:
            raise AttributeError(f"Training sample has no classification")

    def __repr__(self) -> str:
        base_attirubutes = self.attr_dict
        base_attirubutes["purpose"] = f"{self.purpose.value}"
        base_attirubutes["species"] = f"{self.species!r}"
        if self.purpose == Purpose.Testing and self._classification:
            base_attirubutes["classification"] = f"{self._classification!r}"
        attrs = ",".join(f"{k}={v}" for k, v in base_attirubutes.items())
        return f"{self.__class__.__name__}({attrs})"
    

class UnknownSample(Sample):
    """분류되지 않은 샘플, 종을 알 수 없음"""

    def __init__(
        self,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
    ) -> None:
        super().__init__(
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width,
        )
        self._classification: Optional[str] = None

    @property
    def classification(self) -> Optional[str]:
        return self._classification
    
    @classification.setter
    def classification(self, value: str) -> None:
        self._classification = value

    def __repr__(self) -> str:
        base_attirubutes = self.attr_dict
        base_attirubutes["classification"] = f"{self._classification!r}"
        attrs = ",".join(f"{k}={v}" for k, v in base_attirubutes.items())
        return f"{self.__class__.__name__}({attrs})"


class TrainingKnownSample(KnownSample):
    """학습 데이터"""
    pass


class TestingKnownSample(KnownSample):
    """테스트 데이터, 정확하거나 정확하지 않은 종 할당"""

    def __init__(
        self,
        /,
        species: str,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
        classification: Optional[str] = None,
    ) -> None:
        super().__init__(
            species=species,
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width,
        )
        self.classification = classification

    def matches(self) -> bool:
        return self.species == self.classification
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"sepal_length={self.sepal_length},"
            f"sepal_width={self.sepal_width},"
            f"petal_length={self.petal_length},"
            f"petal_width={self.petal_width},"
            f"species={self.species!r},"
            f"classification={self.classification!r}"
            f")"
        )


class ClassifiedSample(Sample):
    """사용자가 제공한 샘플을 분류한 결과"""

    def __init__(self, classification: str, sample: Sample) -> None:
        super().__init__(
            sepal_length=sample.sepal_length,
            sepal_width=sample.sepal_width,
            petal_length=sample.petal_length,
            petal_width=sample.petal_width,
        )
        self.classification= classification

    def __repr__(self) -> str:
        return(
            f"{self.__class__.__name__}("
            f"sepal_length={self.sepal_length},"
            f"sepal_width={self.sepal_width},"
            f"petal_length={self.petal_length},"
            f"petal_width={self.petal_width},"
            f"classification={self.classification!r}"
            f")"
        )
    

class Distance:
    """거리 계산 정의"""

    def distance(self, s1: Sample, s2: Sample) -> float:
        raise NotImplementedError


class ED(Distance):
    def distance(self,s1: Sample,s2:Sample) -> float:
        return hypot(
            s1.sepal_length - s2.sepal_length,
            s1.sepal_width - s2.sepal_width,
            s1.petal_length - s2.petal_length,
            s1.petal_width - s2.petal_width,
        )
    

class MD(Distance):
    def distance(self, s1:Sample, s2:Sample) -> float:
        return sum(
            [
                abs(s1.sepal_length - s2.sepal_length),
                abs(s1.sepal_width - s2.sepal_width),
                abs(s1.petal_length - s2.petal_length),
                abs(s1.petal_width - s2.petal_width),
            ]
        )


class CD(Distance):
    def distance(self, s1:Sample, s2:Sample) -> float:
        return max(
            [
                abs(s1.sepal_length - s2.sepal_length),
                abs(s1.sepal_width - s2.sepal_width),
                abs(s1.petal_length - s2.petal_length),
                abs(s1.petal_width - s2.petal_width),
            ]
        )


class SD(Distance):
    def distance(self, s1:Sample, s2:Sample) -> float:
        return sum(
            [
                abs(s1.sepal_length - s2.sepal_length),
                abs(s1.sepal_width - s2.sepal_width),
                abs(s1.petal_length - s2.petal_length),
                abs(s1.petal_width - s2.petal_width),
            ]
        ) / sum(
            [
                s1.sepal_length + s2.sepal_length,
                s1.sepal_width + s2.sepal_width,
                s1.petal_length + s2.petal_length,
                s1.petal_width + s2.petal_width,
            ]
        )

class Chebyshev(Distance):
    """ Chebyshev 거리 계산 정의"""
    def distance(self, s1:Sample, s2:Sample) -> float:
        return max(
            [
                abs(s1.sepal_length - s2.sepal_length),
                abs(s1.sepal_width - s2.sepal_width),
                abs(s1.petal_length - s2.petal_length),
                abs(s1.petal_width - s2.petal_width),
            ]
        )
    

class Minkowski(Distance):
        """Manhantten, Euclidean 구현 추상화"""

        @property
        @abc.abstractmethod
        def m(self) -> float:
            ...
        
        def distance(self, s1:Sample, s2:Sample) -> float:
            return(
                sum(
                    [
                        abs(s1.sepal_length - s2.sepal_length) ** self.m,
                        abs(s1.sepal_width - s2.sepal_width) ** self.m,
                        abs(s1.petal_length - s2.petal_length) ** self.m,
                        abs(s1.petal_width - s2.petal_width) ** self.m,
                    ]    
            ) ** (1 / self.m)
            )
        

class Euclidean(Minkowski):
    m=2


class Manhattan(Minkowski):
    m=1


class Sorensen(Distance):
    def distance(self, s1:Sample, s2:Sample) -> float:
        return sum(
            [
                abs(s1.sepal_length - s2.sepal_length),
                abs(s1.sepal_width - s2.sepal_width),
                abs(s1.petal_length - s2.petal_length),
                abs(s1.petal_width - s2.petal_width),
            ]
        )/ sum(
            [
                s1.sepal_length + s2.sepal_length,
                s1.sepal_width + s2.sepal_width,
                s1.petal_length + s2.petal_length,
                s1.petal_width + s2.petal_width,
            ]
        )
    

class Minkowski_2(Distance):
    """Manhantten, Euclidean, Chebyshev 구현 추상화"""

    @property
    @abc.abstractmethod
    def m(self) -> float:
        ...
    
    @staticmethod
    @abc.abstractstaticmethod
    def reduction(values: Iterable[float]) -> float:
        ...

    def distance(self, s1:Sample, s2:Sample) -> float:
        return(
            self.reduction(
            [
                abs(s1.sepal_length - s2.sepal_length) ** self.m,
                abs(s1.sepal_width - s2.sepal_width) ** self.m,
                abs(s1.petal_length - s2.petal_length) ** self.m,
                abs(s1.petal_width - s2.petal_width) ** self.m,
            ]
            ) ** (1 / self.m)
        )
    

class CD2(Minkowski_2):
    m=1

    @staticmethod
    def reduction(values: Iterable[float]) -> float:
        return max(values)


class MD2(Minkowski_2):
    m=1

    @staticmethod
    def reduction(values: Iterable[float]) -> float:
        return sum(values)
    

class ED2(Minkowski_2):
    m=2

    @staticmethod
    def reduction(values: Iterable[float]) -> float:
        return sum(values)
    

class ED2S(Minkowski_2):
    m=2
    reduction = sum


class Hyperparameter:
    """하이퍼파라미터 값과 분류의 전체 품질"""

    def __init__(self, k:int, training: "TrainingData") -> None:
        self.k=k
        self.data: TrainingData = training
        self.quality: float

    def test(self) -> None:
        """전체 테스트 스위트 실행"""
        pass_count, fail_count= 0, 0
        for sample in self.data.testing:
            sample.classification = self.classify(sample)
            if sample.matches():
                pass_count +=1
            else:
                fail_count +=1
        self.quality = pass_count / (pass_count + fail_count)

    def classify(self, sample: Sample) -> str:
        """k-nn 알고리즘"""
        return ""
    

class SampleDict(TypedDict):

    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    classification: str


class SamplePartition(List[SampleDict],abc.ABC):
    @overload
    def __init__(self, *, training_subset : float = 0.80) -> None:
        ...

    @overload
    def __init__(
        self,
        iterable: Optional[Iterable[SampleDict]] = None,
        *,
        training_subset: float = 0.80,
    ) -> None:
        ...
    
    def __init__(
        self,
        iterable: Optional[Iterable[SampleDict]] = None,
        *,
        training_subset: float = 0.80,
    ) -> None:
        self.training_subset = training_subset
        if iterable :
            super().__init__(iterable)
        else:
            super().__init__()

    
    @abc.abstractproperty
    @property
    def training(self) -> list[TrainingKnownSample]:
        ...

    @abc.abstractproperty
    @property
    def testing(self) -> list[TestingKnownSample]:
        ...
    

class ShufflingSamplePartition(SamplePartition):
    def __init__(
            self,
            iterable: Optional[Iterable[SampleDict]] = None,
            *,
            training_subset: float = 0.80,
    ) -> None:
        super().__init__(iterable, training_subset=training_subset)
        self.shuffle()

    def shuffle(self) -> None:
        if not self.split:
            random.shuffle(self)
            self.split - int(len(self) * self.training_subset)

    @property
    def training(self) -> list[TrainingKnownSample]:
        self.shuffle()
        return [TestingKnownSample(**sd) for sd in self[self.split :]]


class DealingPartition(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        items: Optional[Iterable[SampleDict]],
        *,
        training_subset: tuple[int,int] = (8,10),
    ) -> None:
        ...

    @abc.abstractmethod
    def extend(self,items: Iterable[SampleDict]) -> None:
        ...

    @abc.abstractmethod
    def append(self,item: SampleDict) -> None:
        ...
    
    @property
    @abc.abstractmethod
    def training(self) -> list[TrainingKnownSample]:
        ...
    
    @property
    @abc.abstractmethod
    def testing(self) -> list[TestingKnownSample]:
        ...


class CountingDealingPartition(DealingPartition):
    def __init__(
            self,
            items: Optional[Iterable[SampleDict]],
            *,
            training_subset: tuple[int,int] = (8,10),
    ) -> None:
        self.training_subset = training_subset  
        self.counter = 0
        self._trainning: list[TrainingKnownSample] = []
        self._testing: list[TestingKnownSample] = []
        if items:
            self.extend(items)
    
    def extend(self, items: Iterable[SampleDict]) -> None:
        for item in items:
            self.append(item)

    def append(self, item: SampleDict) -> None:
        n,d = self.training_subset
        if self.count % d<n:
            self._training.append(TrainingKnownSample(**item))
        else:
            self._testing.append(TestingKnownSample(**item))
        self.counter +=1

    @property
    def training(self) -> list[TrainingKnownSample]:
        return self._training
    
    @property
    def testing(self) -> list[TestingKnownSample]:
        return self._testing

import collections.abc
import typing
import sys

if sys.version_info>=(3,9):
    BucketCollection = collections.abc.Collection[Sample]
else:
    BucketCollection = typing.Collection[Sample]


class BucketedCollection(BucketCollection):
    def __init__(self, samples: Optional[Iterable[Sample]] = None) -> None:
        super().__init__()
        self.buckets: dict[str, list[Sample]] = collections.defaultdict(list)
        if samples:
            self.extend(samples)
    
    def extend(self, samples: Iterable[Sample]) -> None:
        for sample in samples:
            self.append(sample)

    def append(self, sample: Sample) -> None:
        b = sample.hash() % 128
        self.buckets[b].append(sample)

    def __contains__(self, target: Any) -> bool:
        b= cast(Sample, target).hash() %128
        return any(existing == target for existing in self.buckets[b])
    
    def __len__(self) -> int:
        return sum(len(b) for b in self.buckets.values())
    
    def __iter__(self) -> Iterable[Sample]:
        return itertools.chain(*self.buckets.values())
    

class BucketedDealingPartition_80(DealingPartition):
    training_subset = (8,10) 


    def __init__(self, items: Optional[Iterable[SampleDict]]) -> None:
        self.counter = 0
        self._training = BucketedCollection()
        self._testing = BucketedCollection()
        if items:
            self.extend(items)

    def extend(self, items: Iterable[SampleDict]) -> None:
        for item in items:
            self.append(item)

    def append(self, item: SampleDict) -> None:
        n,d = self.training_subset
        if self.counter % d < n:
            self._training.append(TrainingKnownSample(**item))
        else:
            candidate = TestingKnownSample(**item)
            if candidate in self._training:
                self._training.append(TestingKnownSample(**item))

            else:
                self._testing.append(candidate)
        self.counter +=1

    @property
    def training(self) -> list[TrainingKnownSample]:
        return cast(list[TrainingKnownSample], list(self._training))
    
    @property
    def testing(self) -> list[TestingKnownSample]:
        return self._testing


class TrainingData:
    """샘플을 로드, 테스트하는 메서드를 가지며 학습 및 테스트 데이터셋을 포함"""

    def __init__(self, name: str)-> None:
        self.name = name
        self.uploaded: datetime.datetime
        self.tested: datetime.datetime
        self.training: List[KnownSample] = []
        self.testing: List[KnownSample] = []
        self.tuning: List[Hyperparameter] = []
    
    def load(
            self,
            raw_data_iter: Iterable[dict[str,str]]
            )-> None:
        """원시데이터 로드 및 분할"""
        partition_class = self.partition_class
        partitioner = partition_class(raw_data_iter, training_subset=(1,2))
        self.training = partitioner.training
        self.testing = partitioner.testing
        self.uploaded = datetime.datetime.now(tz=datetime.timezone.utc)

    def test(
            self, 
            parameter: Hyperparameter) -> None:
        """하이퍼파라미터 값으로 테스트"""
        parameter.test()
        self.tuning.append(parameter)
        self.tested = datetime.datetime.now(tz=datetime.timezone.utc)

    def classify(
            self,
            parameter: Hyperparameter,
            sample: UnknownSample) -> ClassifiedSample:
        return ClassifiedSample(
            classification=parameter.classify(sample), sample=sample
        )
    

class CSVIrisReader:
    def __init__(self, source: Path) -> None:
        self.source = source

    def data_iter(self) -> Iterator[dict[str,str]]:
        with self.source.open() as source_file:
            reader = csv.DictReader(source_file, self.header)
            yield from reader

class CSVIrisReader_2:
    def __init__(self, source: Path) -> None:
        self.source = source

    def data_iter(self) -> Iterator[dict[str,str]]:
        with self.source.open() as source_file:
            reader = csv.DictReader(source_file)
            for row in reader:
                yield dict(
                    sepal_length=row[0],
                    sepal_width=row[1],
                    petal_length=row[2],
                    petal_width=row[3],
                    species=row[4],
                )

class JSONIrisReader:
    def __init__(self, source: Path) -> None:
        self.source = source

    def data_iter(self) -> Iterator[SampleDict]:
        with self.source.open() as source_file:
            data = json.load(source_file)
            yield from iter(data)


class NDJSONIrisReader:
    def __init__(self, source: Path) -> None:
        self.source = source

    def data_iter(self) -> Iterator[SampleDict]:
        with self.source.open() as source_file:
            for line in source_file:
                sample = json.loads(line)
                yield sample


class ValidatingNDJSONIrisReader:
    def __init__(self, source: Path) -> None:
        self.source = source

    def data_iter(self) -> Iterator[SampleDict]:
        with self.source.open() as source_file:
            for line in source_file:
                sample = json.loads(line)
                if self.validator.is_valid(sample):
                    yield sample
                else:
                    print(f"Invalid: {sample}")


class YMALIrisReader:
    def __init__(self, source: Path) -> None:
        self.source = source

    def data_iter(self) -> Iterator[SampleDict]:
        with self.source.open() as source_file:
            yield from yaml.load_all(source_file, Loader=yaml.SafeLoader)

class BadSampleRow(ValueError):
    pass


class SampleReader:


    target_class = Sample
    header = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

    def __init__(self,source :Path) -> None:
        self.source = source
        
    def sample_iter(self) -> Iterator[Sample]:
        target_class = self.target_class
        with self.source.open() as source_file:
            reader = csv.DictReader(source_file, self.header)
            for row in reader:
                try:
                    sample = target_class(
                        sepal_length=float(row["sepal_length"]),
                        sepal_width=float(row["sepal_width"]),
                        petal_length=float(row["petal_length"]),
                        petal_width=float(row["petal_width"]),
                    )

                except ValueError as ex:
                    raise BadSampleRow(f"Invalid {row!r}") from ex
                yield sample


__test__ = {name: case for name, case in locals().items() if name.startswith("test_")}
