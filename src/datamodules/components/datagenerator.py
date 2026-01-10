from pathlib import Path

class DataGenerator:
    def __init__(self, path: Path):
        hdf_path = path / ""
        pass

    def feature_scale(self, X):
        return (X - self.mean) / self.std

    def generate_train(self):
        """
        returns normalised training and validation features
        """
        pass


class DataGeneratorTest(DataGenerator):
    def __init__(self):
        pass

    def generate_eval(self):
        """
        returns normalised evaluation features
        """
        pass
