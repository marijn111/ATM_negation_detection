from assignment3.main import load_file
from sklearn.base import TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction import DictVectorizer


class Transformer(TransformerMixin):
    """ Prepare dataframe for DictVectorizer """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return (row for _, row in X.iterrows())


def vectorize(df):
    vectorizer = make_pipeline(Transformer(), DictVectorizer())
    # now you can use vectorizer as you might expect, e.g.
    vectorized_features = vectorizer.fit_transform(df)
    return vectorized_features


