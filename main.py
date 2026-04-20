import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# dataset (study hours vs scores)
hours = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
scores = np.array([50, 55, 65, 70, 80, 90])

# train model
model = LinearRegression()
model.fit(hours, scores)

# prediction
new_hours = np.array([[4.5]])
predicted_score = model.predict(new_hours)

print("Predicted score for 4.5 hours:", predicted_score[0])

# visualization
plt.scatter(hours, scores)
plt.plot(hours, model.predict(hours))
plt.xlabel("Study Hours")
plt.ylabel("Score")
plt.title("Student Performance Prediction")
plt.show()
