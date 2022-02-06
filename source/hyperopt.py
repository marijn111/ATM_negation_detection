from train import TrainModel

"""
This file loads the model, reads in the train and test data, and calls the hyperopting functions from the model instance.
"""

model_class = TrainModel()
model_class.load_model()

model_class.y_train = model_class.load_data('y_train.pkl')
model_class.X_train = model_class.load_data('x_train.pkl')
model_class.y_test = model_class.load_data('y_test.pkl')
model_class.X_test = model_class.load_data('x_test.pkl')

model_class.hyperopt()
model_class.check_parameter_space()
model_class.check_best_estimator()
model_class.check_classifier_learning()
model_class.check_state_features()
