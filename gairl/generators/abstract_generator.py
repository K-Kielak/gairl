from abc import ABC, abstractmethod


class AbstractGenerator(ABC):

    def __init__(self, data_shape):
        self._data_shape = data_shape

    @abstractmethod
    def train_step(self, expected_output, condition=None):
        """
        Trains generator based on given batch of data.
        :param expected_output: Batch of data that describes the
            distribution that generator should follow.
        :param condition: Provided when generator should generate
            data based on certain conditions. If provided there
            should be exactly one condition for each expected_output.
        :return: None
        """
        pass

    @abstractmethod
    def generate(self, how_many, condition=None):
        """
        Generates data based on the distribution it has learned.
        :param how_many: Describes how many pieces of data it
            should generate.
        :param condition: If conditional generator, provides
            conditions based on which data should be generated.
            There should be as many conditions as specified by how_many.
        :return: Generated pieces of data.
        """
        pass

    @abstractmethod
    def visualize_data(self, data):
        """
        Visualizes provided data in its outputs directory.
        :param data: Data to visualize.
        :return: None.
        """
        pass
