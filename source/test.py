from train import TrainModel

"""
This file runs the test set and produces the evaluation for the predictions made. 
"""

model_class = TrainModel()
model_class.load_model()

model_class.X_test = model_class.load_data('x_test_cardboard.pkl')
model_class.y_test = model_class.load_data('y_test_cardboard.pkl')
# model_class.y_pred = model_class.load_data('y_pred_circle.pkl')

model_class.predict()
model_class.evaluation()
