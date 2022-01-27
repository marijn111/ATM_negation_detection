from train import TrainModel
from sklearn_crfsuite import metrics


model_class = TrainModel()
model_class.load_model()


model_class.y_test = model_class.load_data('y_test.pkl')
model_class.y_pred = model_class.load_data('y_pred.pkl')

model_class.evaluation()
