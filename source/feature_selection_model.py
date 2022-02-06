from train import TrainModel


class FeatureSelectedModel(TrainModel):
    """
        This model inherits from the basic TrainModel, and differs only in the features it trains on. As this is the model
        after the feature selection,
        it will only train on the tokens and lemma's (as was found in that experiment). Therefore the word2features function only includes the token and lemma.
    """
    def __init__(self):
        super().__init__()
        self.save_directory = 'features_selected_model'

    def word2features(self, df_token):
        features = {
            'token': df_token['token'].values[0],
            'lemma': df_token['LEMMA'].values[0]
        }

        return features


def main(input_path):
    feature_selected_class = FeatureSelectedModel()
    feature_selected_class.load_processed_corpus(input_path)
    feature_selected_class.model_init()
    feature_selected_class.get_train_test_data()
    feature_selected_class.split_train_test_data()
    feature_selected_class.fit_model()
    feature_selected_class.save_model()
    feature_selected_class.predict()
    feature_selected_class.evaluation()

    # feature_selected_class.load_model()


if __name__ == '__main__':
    processed_corpus_path = "./dataset/processed_corpus.csv"
    main(processed_corpus_path)
