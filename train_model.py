import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# ■■ Training data ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# Each row: [study_hours, prev_score]
X = np.array(
    [
        [1, 40],  # studied 1 hr, scored 40 previously → Fail
        [2, 45],  # studied 2 hrs, scored 45 → Fail
        [3, 50],  # studied 3 hrs, scored 50 → Fail
        [4, 60],  # studied 4 hrs, scored 60 → Pass
        [5, 65],  # studied 5 hrs, scored 65 → Pass
        [6, 70],  # studied 6 hrs, scored 70 → Pass
        [7, 75],  # studied 7 hrs, scored 75 → Pass
        [8, 85],  # studied 8 hrs, scored 85 → Pass
        [2, 80],  # studied 2 hrs but high prev score → Pass
        [1, 30],  # studied 1 hr, low prev score → Fail
    ]
)

# 0 = Fail, 1 = Pass
y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 0])

# ■■ Train the model ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
model = LogisticRegression()
model.fit(X, y)  # this is where learning happens

# ■■ Save to disk ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
joblib.dump(model, "model.pkl")
print("Model trained and saved to model.pkl")

# Quick test on the training data
predictions = model.predict(X)
print("Sample predictions:", predictions)
print("Actual labels:     ", y)