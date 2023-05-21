
class DataPreparation:
    def __init__(self, preprocessing_fn, split_ratio):
        self.preprocessing_fn = preprocessing_fn
        self.split_ratio = split_ratio

    def preprocess(self, raw_data):
        return self.preprocessing_fn(raw_data)

    def split(self, data, labels):
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=self.split_ratio, random_state=42)
        return x_train, x_test, y_train, y_test
