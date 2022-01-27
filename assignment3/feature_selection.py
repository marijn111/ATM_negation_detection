from train import TrainModel


class FeatureSelectedModel(TrainModel):
    def __init__(self):
        super().__init__()
        self.save_directory = 'feature_selected'

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
