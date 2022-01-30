from train_test import TrainModelSVM
from sklearn_pandas import DataFrameMapper, gen_features
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn_crfsuite import metrics


model_class = TrainModelSVM()
model_class.load_model()

gf = gen_features(cat_features, [HashingVectorizer])
mapper = DataFrameMapper(gf)
cat_features_transformed = mapper.fit_transform(df)

target_name_encoded = LabelEncoder().fit_transform(df["State"])



# print(list(model_class.model.classes_.remove('O')))

# model_class.y_test = model_class.load_data('y_test.pkl')
# model_class.y_pred = model_class.load_data('y_pred.pkl')
#
# model_class.evaluation()
#
# labels = model_class.model.classes_
# labels.remove('O')
# print(metrics.flat_f1_score(model_class.y_test, model_class.y_pred,
#                             average='weighted', labels=labels))
#
# print(metrics.flat_precision_score(model_class.y_test, model_class.y_pred,
#                             average='weighted', labels=labels))
#
# print(metrics.flat_recall_score(model_class.y_test, model_class.y_pred,
#                             average='weighted', labels=labels))
