import pytest
import sys
sys.path.append("src")
from model_7 import Sample, KnownSample, TrainingKnownSample, TestingKnownSample, UnknownSample, Hyperparameters, TrainingData, Distance
import weakref

testing_samples = [
    TestingKnownSample(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species="setosa"),
    TestingKnownSample(sepal_length=4.9, sepal_width=3.0, petal_length=1.4, petal_width=0.2, species="versicolor"),
    
]
training_samples = [
    TrainingKnownSample(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species="setosa"),
    TrainingKnownSample(sepal_length=4.9, sepal_width=3.0, petal_length=1.4, petal_width=0.2, species="versicolor"),
   
]
training_data = TrainingData(testing=testing_samples, training=training_samples, tuning=[])
tuning_params = [
    Hyperparameters(k=3, algo=Distance(), data=weakref.ref(training_data)),
    Hyperparameters(k=5, algo=Distance(), data=weakref.ref(training_data)),
    
]
training_data = TrainingData(testing=testing_samples, training=training_samples, tuning=tuning_params)

def test_hyperparameters_classify():
    hyperparams = Hyperparameters(k=3, algo=Distance(), data=weakref.ref(training_data))
    sample = Sample(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2)
    result = hyperparams.classify(sample)
    assert isinstance(result, str)

def test_trainingdata():
    testing_samples = [TestingKnownSample(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species="setosa")]
    training_samples = [TrainingKnownSample(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species="setosa")]
    tuning_params = [Hyperparameters(k=3, algo=Distance(), data=weakref.ref(TrainingData([], [], [])))]
    training_data = TrainingData(testing=[], training=[], tuning=tuning_params)

    training_data.training.append(TrainingKnownSample(species="setosa", sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2))
    assert len(training_data.training) == 1

if __name__ == "__main__":
    pytest.main([__file__])
