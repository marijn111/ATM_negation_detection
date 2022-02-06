from train import TrainModel
from collections import defaultdict
import pandas as pd

import matplotlib.pyplot as plt

"""
This script loads the model and extracts the state feature coefficients, and sums them by feature. This gives us the features 
with the highest overall coefficients. This data is then plotted to produce the barchart with feature importance based
on the coefficients. 
"""

model_class = TrainModel()
model_class.load_model()

feature_coeffs = model_class.model.state_features_
all_coeffs = defaultdict(lambda: 0)

for k, v in feature_coeffs.items():
    feature = k[0].split(':')[0]
    all_coeffs[feature] += v


all_coeffs = dict(all_coeffs)
df = pd.DataFrame(all_coeffs, index=[0])

plot = df.T[0].plot.bar(color=['green', 'yellow', 'turquoise', 'blue', 'purple', 'orange', 'magenta', 'black', 'red', 'pink'])
plt.xlabel("Features")
plt.ylabel("Feature State Coefficients")
plt.title("Feature importances via coefficients")

plt.savefig('./plots/coeffs.png', bbox_inches='tight')
plt.show()
