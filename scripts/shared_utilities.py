import os

import keras
from keras.src.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from pathlib import Path

from sklearn.linear_model import Ridge

DEFAULT_WINDOW = 48
DEFAULT_FEATURES = 2
DEFAULT_STEPS = "n_steps_all"
DEFAULT_SLEEP_SCORE = "n_sleep_score_all"
DEFAULT_TEST_RATIO = 0.1
DEFAULT_DATASET = Path("../data/scalati.csv.gz")
DEFAULT_CHUNK_SIZE = 100000
DEFAULT_CHUNK_START = 0


class Data(ABC):
    def __init__(self, window: int = DEFAULT_WINDOW, test_ratio: float = DEFAULT_TEST_RATIO,
                 dataset: Path = DEFAULT_DATASET, dataset_chunk: int = DEFAULT_CHUNK_SIZE,
                 sleep_score: str = DEFAULT_SLEEP_SCORE, step_type: str = DEFAULT_STEPS,
                 chunk_start: int = DEFAULT_CHUNK_START):
        self._chunk_start = chunk_start
        self._step_type = step_type
        self._sleep_score_type = sleep_score
        self._dataset_chunk = dataset_chunk
        self._dataset = dataset
        self._xtrain, self._xtest, self._ytrain, self._ytest = [], [], [], []
        self._window = window
        self._test_ratio = test_ratio
        self._data = None
        self._ready = False
        self._n_features = self._get_features()
        self._prediction_offset = self._get_prediction_offset()

    def load_dataset(self):
        self._data = pd.read_csv(self._dataset, skiprows=range(1, self._chunk_start), nrows=self._dataset_chunk).dropna(
            subset=["steps"])

    def get_dataset(self):
        if self._data is None:
            self.load_dataset()
        return self._data

    def get_data(self):
        if not self._ready:
            self.prepare_data()
        return self._xtrain, self._ytrain, self._xtest, self._ytest

    def get_training_data(self):
        return self.get_data()[:2]

    def get_test_data(self):
        return self.get_data()[2:]

    def _reshape_data(self):
        self._xtrain = np.array(self._xtrain).reshape(-1, self._window, self._n_features)
        self._ytrain = np.array(self._ytrain).reshape(-1, 1)
        self._xtest = np.array(self._xtest).reshape(-1, self._window, self._n_features)
        self._ytest = np.array(self._ytest).reshape(-1, 1)

    def prepare_data(self):
        self._make_training_test_data()
        self._reshape_data()
        self._ready = True

    def _make_training_test_data(self):
        for uuid, group in self.get_dataset().groupby('uuid')[["n_date", self._step_type, self._sleep_score_type]]:
            split = int(len(group) * (1 - self._test_ratio))
            if split < self._window:
                continue

            inputs = self._get_inputs(group)
            output = self._get_outputs(group)

            for i in range(self._window, split):
                if not np.isnan(output[
                                    i - 1]):  # Ho sicuramente i passi, ma considero la finestra solo se conosco il
                    # valore del sonno alla fine
                    self._xtrain.append(inputs[i - self._window:i])  # Una finestra di <window> osservazioni sui passi
                    self._ytrain.append(
                        output[i - self._prediction_offset])  # La qualità del sonno alla fine della finestra

            for i in range(split, len(group)):
                if not np.isnan(output[i - 1]):
                    self._xtest.append(inputs[i - self._window:i])
                    self._ytest.append(
                        output[i - self._prediction_offset])  # predico l'ennesimo giorno invece dell'n+1-esimo

    @abstractmethod
    def _get_inputs(self, group):
        pass

    @abstractmethod
    def _get_outputs(self, group):
        pass

    @abstractmethod
    def _get_features(self):
        pass

    @abstractmethod
    def _get_prediction_offset(self):
        pass

    def get_window(self):
        return self._window

    def get_step_type(self):
        return self._step_type

    def get_sleep_type(self):
        return self._sleep_score_type


class DateStepData(Data):
    def _get_prediction_offset(self):
        return 0

    def _get_features(self):
        return 2

    def _get_inputs(self, group):
        return np.array(group[["n_date", self._step_type]].values)

    def _get_outputs(self, group):
        return np.array(group[[self._step_type]].values)


class StepData(Data):
    def _get_prediction_offset(self):
        return 0

    def _get_inputs(self, group):
        return np.array(group[[self._step_type]].values)

    def _get_outputs(self, group):
        return np.array(group[[self._step_type]].values)

    def _get_features(self):
        return 1


class DateStepSleepData(Data):
    def _get_prediction_offset(self):
        return 1

    def _get_inputs(self, group):
        return np.array(group[["n_date", self._step_type]].values)

    def _get_outputs(self, group):
        return np.array(group[[self._sleep_score_type]].values)

    def _get_features(self):
        return 2


class StepSleepData(Data):
    def _get_prediction_offset(self):
        return 1

    def _get_inputs(self, group):
        return np.array(group[[self._step_type]].values)

    def _get_outputs(self, group):
        return np.array(group[[self._sleep_score_type]].values)

    def _get_features(self):
        return 1


class Model(ABC):
    @abstractmethod
    def fit(self, xs, ys):
        pass

    @abstractmethod
    def predict(self, xs):
        pass

    @abstractmethod
    def score(self, xs, ys):
        pass

    @abstractmethod
    def predict_batch(self, xs):
        pass


class KerasModel(Model):
    def __init__(self, path: Path, window=DEFAULT_WINDOW, n_features=DEFAULT_FEATURES, load=False):
        self._path = path
        if load and os.path.exists(self._path):
            self._model = keras.models.load_model(path)
        else:
            self._model = self._init_model(window, n_features)
        self._model.compile(loss=keras.losses.MeanSquaredError(), metrics=[keras.metrics.R2Score()])

    @abstractmethod
    def _init_model(self, window, n_features):
        pass

    def fit(self, xs, ys):
        return self._model.fit(xs, ys, epochs=20, verbose=1, validation_split=.2,
                               callbacks=[EarlyStopping(monitor='val_loss', patience=5),
                                          ModelCheckpoint(filepath='best_model.keras', monitor='val_loss',
                                                          save_best_only=True)])

    def predict(self, x):
        return self._model.predict(x, verbose=False)

    def score(self, xs, ys):
        return self._model.evaluate(xs, ys)

    def predict_batch(self, xs):
        return self._model.predict_on_batch(xs)

    def summary(self):
        return self._model.summary()

    def save(self):
        self._model.save(self._path)


class ComposedKerasModel(KerasModel):
    def __init__(self, basemodel, path: Path, load=False):
        self._basemodel = self._load_basemodel(basemodel)
        super().__init__(path, load)

        self._window = 0
        self._n_features = 0
        self._out = 0

    @staticmethod
    def _load_basemodel(basemodel):
        if isinstance(basemodel, Path) or isinstance(basemodel, str):
            basemodel = keras.models.load_model(basemodel)
        return basemodel

    def _init_model(self, window, n_features):
        final_layer = keras.layers.Dense(1)(self._basemodel.layers[-2].output)
        model = keras.Model(inputs=self._basemodel.layers[0].output, outputs=final_layer)
        self._window = model.layers[0].output.shape[1]
        self._n_features = self._basemodel.layers[0].output.shape[2]
        self._out = model.layers[-1].input.shape[1]
        return model


class KerasCNNModel(KerasModel):
    def _init_model(self, window, n_features):
        # L'ho definita secondo l'API funzionale di Keras, in modo da poter estrarre le attivazioni dei layer intermedi
        # con facilità

        inputs = keras.Input(shape=(window, n_features))
        old = inputs
        for rate in (1, 2, 4, 8) * 2:
            old = keras.layers.Conv1D(filters=20, kernel_size=2, padding="causal",
                                      activation="relu", dilation_rate=rate)(old)
        flatten = keras.layers.Flatten()(old)
        output = keras.layers.Dense(1, activity_regularizer=keras.regularizers.L2(0.01))(flatten)

        model = keras.Model(inputs=inputs, outputs=output)
        return model


class KerasLSTMModel(KerasModel):

    def _init_model(self, window, n_features):
        inputs = keras.Input(shape=(window, n_features))
        lstm_1 = keras.layers.LSTM(30)(inputs)
        output = keras.layers.Dense(1)(lstm_1)

        model = keras.Model(inputs=inputs, outputs=output)
        return model


class GenericSklearnModel(Model):
    def fit(self, xs, ys):
        return self._model.fit(xs, ys)

    def predict(self, xs):
        return self._model.fit(xs)

    def score(self, xs, ys):
        return self._model.score(xs, ys)

    def predict_batch(self, xs):
        return self._model.predict(xs)

    def __init__(self, model):
        self._model = model


class RidgeModel(GenericSklearnModel):
    def __init__(self, window: int = DEFAULT_WINDOW):
        super().__init__(Ridge())
        self._window = window

    def fit(self, xs, ys):
        xs = xs.reshape(-1, self._window)
        return self._model.fit(xs, ys)

    def predict(self, xs):
        xs = xs.reshape(-1, self._window)
        return self._model.predict(xs)

    def score(self, xs, ys):
        xs = xs.reshape(-1, self._window)
        return self._model.score(xs, ys)

    def predict_batch(self, xs):
        xs = xs.reshape(-1, self._window)
        return self._model.predict(xs)


class ComposedKerasSklearnModel(GenericSklearnModel):
    def __init__(self, basemodel, topmodel):
        super().__init__(topmodel)
        if isinstance(basemodel, Path) or isinstance(basemodel, str):
            basemodel = keras.models.load_model(basemodel)
        self._basemodel = keras.Model(inputs=basemodel.layers[0].output, outputs=basemodel.layers[-2].output)
        self._window = basemodel.layers[0].output.shape[1]
        self._features = basemodel.layers[0].output.shape[2]
        self._out = basemodel.layers[-2].output.shape[1]

    def predict(self, x):
        x = self._get_activation(x)
        return super().predict(x)

    def score(self, xs, ys):
        xs = self._get_activations(xs)
        ys = ys.reshape(-1)
        return super().score(xs, ys)

    def predict_batch(self, xs):
        xs = self._get_activations(xs)
        return super().predict_batch(xs)

    def fit(self, xs, ys):
        xs = self._get_activations(xs)
        ys = ys.reshape(-1)
        return super().fit(xs, ys)

    def _get_activation(self, x):
        return self._get_activations(x.reshape(-1, self._window, self._features))

    def _get_activations(self, xs):
        return self._basemodel.predict_on_batch(xs)


class Utilities:
    def __init__(self, model, data: Data):
        self.model = model
        self.data = data

    def load_data(self):
        self.data.load_dataset()
        self.data.prepare_data()

    def train_model(self):
        self.model.fit(*self.data.get_training_data())

    def evaluate_model(self):
        return self.model.score(*self.data.get_test_data())


class Visualization(ABC):
    def __init__(self, model: Model, data: Data, users: int = 50):
        self._model = model
        self._data = data
        self.users = users
        self._window = self._data.get_window()
        self._step_type = self._data.get_step_type()
        self._sleep_type = self._data.get_sleep_type()

    def visualize(self):
        df = self._data.get_dataset()

        counter = 0
        if (mean := "m" + self._sleep_type) in df.keys():
            relevant = ["n_date", self._step_type, self._sleep_type, mean]
        else:
            relevant = ["n_date", self._step_type, self._sleep_type]
        for uuid, group in df.groupby('uuid')[relevant]:
            if len(group) < 2 * self._window:
                continue

            self._visualize_group(uuid, group)

            plt.legend()
            plt.show()
            plt.close()

            counter += 1
            if counter >= self.users:
                break

    @abstractmethod
    def _visualize_group(self, uuid, group):
        pass


class DateStepForecastVisualization(Visualization):

    def _visualize_group(self, uuid, group):
        group = group[["n_date", self._step_type]]

        plt.title(f"Forecast for user {uuid}")
        plt.xlabel("n_date")
        plt.ylabel(self._step_type)

        plt.plot(group["n_date"].values[:-self._window], group[self._step_type].values[:-self._window], label="facts")
        plt.plot(group["n_date"].values[-self._window:], group[self._step_type].values[-self._window:], label="truth")

        avg_timestep = (group["n_date"].max() - group["n_date"].min()) / len(group)
        max_timestep = group["n_date"][:-self._window].max()

        prediction_data = group.values[:-self._window]
        for step in range(self._window):
            new_datapoint = [max_timestep + step * avg_timestep,
                             self._model.predict(np.array([prediction_data[-self._window:]]))[0, 0]]

            prediction_data = np.append(prediction_data, new_datapoint).reshape(-1, 2)

        prediction_data = np.array(prediction_data[-self._window:])

        plt.plot(prediction_data[:, 0], prediction_data[:, 1], label="forecast")


class DateStepPredictionVisualization(Visualization):

    def _visualize_group(self, uuid, group):
        group = group[["n_date", self._step_type]]

        plt.title(f"Prediction for user {uuid}")
        plt.xlabel("n_date")
        plt.ylabel(self._step_type)

        plt.plot(group["n_date"].values, group[self._step_type].values, label="facts")

        prediction_data = []

        for i in range(self._window, len(group)):
            prediction_data.append(group[i - self._window:i])

        prediction_data = self._model.predict_batch(np.array(prediction_data))

        plt.plot(group["n_date"].values[self._window:], prediction_data, label="prediction")


class DateStepPredictionForecastVisualization(Visualization):
    def _visualize_group(self, uuid, group):
        group = group[["n_date", self._step_type]]

        plt.title(f"Prediction + Forecast for user {uuid}")
        plt.xlabel("n_date")
        plt.ylabel(self._step_type)

        plt.plot(group["n_date"].values, group[self._step_type].values, label="facts")

        prediction_data = []

        for i in range(self._window, len(group)):
            prediction_data.append(group[i - self._window:i])

        prediction_data = self._model.predict_batch(np.array(prediction_data))

        plt.plot(group["n_date"].values[self._window:], prediction_data, label="prediction")

        avg_timestep = (group["n_date"].max() - group["n_date"].min()) / len(group)
        max_timestep = group["n_date"].max()

        prediction_data = group.values
        for step in range(self._window):
            new_datapoint = [max_timestep + step * avg_timestep,
                             self._model.predict(np.array([prediction_data[-self._window:]]))[0, 0]]

            prediction_data = np.append(prediction_data, new_datapoint).reshape(-1, 2)

        prediction_data = np.array(prediction_data[-self._window:])

        plt.plot(prediction_data[:, 0], prediction_data[:, 1], label="forecast")


class DateSleepPredictionVisualization(Visualization):
    def _visualize_group(self, uuid, group):
        group = group[["n_date", self._sleep_type, self._step_type]]

        plt.title(f"Prediction for user {uuid}")
        plt.xlabel("n_date")
        plt.ylabel(self._sleep_type)

        plt.plot(group["n_date"].values, group[self._sleep_type].values, label="facts")

        prediction_data = []

        for i in range(self._window, len(group)):
            prediction_data.append(group[["n_date", self._step_type]][i - self._window:i])

        prediction_data = self._model.predict_batch(np.array(prediction_data))

        plt.plot(group["n_date"].values[self._window:], prediction_data, label="prediction")


class DateSleepStepPredictionVisualization(Visualization):
    def _visualize_group(self, uuid, group):
        group = group[["n_date", self._sleep_type, self._step_type]]

        steps = plt.subplot(2, 1, 2)
        plt.ylabel(self._step_type)
        plt.plot(group["n_date"], group[self._step_type])
        plt.xlabel("n_date")

        plt.subplot(2, 1, 1, sharex=steps)
        plt.title(f"Prediction for user {uuid}")
        plt.xlabel("n_date")
        plt.ylabel(self._sleep_type)

        plt.plot(group["n_date"].values, group[self._sleep_type].values, label="facts")

        prediction_data = []

        for i in range(self._window, len(group)):
            prediction_data.append(group[["n_date", self._step_type]][i - self._window:i])

        prediction_data = self._model.predict_batch(np.array(prediction_data))

        plt.plot(group["n_date"].values[self._window:], prediction_data, label="prediction")


class DateMeanSleepStepPredictionVisualization(Visualization):
    def _visualize_group(self, uuid, group):
        full = group
        group = group[["n_date", self._sleep_type, self._step_type]]

        steps = plt.subplot(2, 1, 2)
        plt.ylabel(self._step_type)
        plt.plot(group["n_date"], group[self._step_type])
        plt.xlabel("n_date")

        plt.subplot(2, 1, 1, sharex=steps)
        plt.title(f"Prediction for user {uuid}")
        plt.xlabel("n_date")
        plt.ylabel(self._sleep_type)

        plt.plot(group["n_date"].values, group[self._sleep_type].values, label="facts")

        if "m" + self._sleep_type in full.keys():
            plt.plot(full["n_date"].values, full["m" + self._sleep_type].values, label="average")

        prediction_data = []

        for i in range(self._window, len(group)):
            prediction_data.append(group[["n_date", self._step_type]][i - self._window:i])

        prediction_data = self._model.predict_batch(np.array(prediction_data))

        plt.plot(group["n_date"].values[self._window:], prediction_data, label="prediction")


class SleepStepPredictionVisualization(Visualization):
    def _visualize_group(self, uuid, group):
        group = group[[self._sleep_type, self._step_type]]

        steps = plt.subplot(2, 1, 2)
        plt.ylabel(self._step_type)
        plt.plot(range(len(group)), group[self._step_type])

        plt.subplot(2, 1, 1, sharex=steps)
        plt.title(f"Prediction for user {uuid}")
        plt.ylabel(self._sleep_type)

        plt.plot(range(len(group)), group[self._sleep_type].values, label="facts")

        prediction_data = []

        for i in range(self._window, len(group)):
            prediction_data.append(group[[self._step_type]][i - self._window:i])

        prediction_data = self._model.predict_batch(np.array(prediction_data))

        plt.plot(range(self._window, len(group)), prediction_data, label="prediction")
