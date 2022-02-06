from train import TrainModel


class BaselineModel(TrainModel):
    """
    This model inherits from the basic TrainModel, and differs only in the features it trains on. As this is the baseline,
    it will only train on the tokens. Therefore the word2features function only includes the token.
    """
    def __init__(self):
        super().__init__()
        self.save_directory = 'baseline_model'

    def word2features(self, df_token):
        features = {
            'token': df_token['token'].values[0]
        }

        return features


def main(input_path):
    baseline_class = BaselineModel()
    baseline_class.load_processed_corpus(input_path)
    baseline_class.model_init()
    baseline_class.get_train_test_data()
    baseline_class.split_train_test_data()
    baseline_class.fit_model()
    baseline_class.save_model()
    baseline_class.predict()
    baseline_class.evaluation()

    # baseline_class.load_model()


if __name__ == '__main__':
    processed_corpus_path = "./dataset/processed_corpus.csv"
    main(processed_corpus_path)
