from train import TrainModel
from sklearn_crfsuite import metrics


model_class = TrainModel()
model_class.load_model()

# print(list(model_class.model.classes_.remove('O')))

model_class.y_test = model_class.load_data('y_test.pkl')
model_class.y_pred = model_class.load_data('y_pred.pkl')

model_class.evaluation()
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
