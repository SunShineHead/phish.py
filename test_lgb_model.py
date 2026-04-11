import lightgbm as lgb
import numpy as np
import pytest

# Sample test data - larger dataset to ensure learning
X = np.array([[1, 2], [1, 2], [3, 4], [3, 4], [5, 6], [5, 6], [1, 2], [3, 4], [5, 6], [3, 4]])
y = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 1])

# Create a LightGBM model
model = lgb.LGBMClassifier(
    min_child_samples=1, 
    min_data_in_leaf=1, 
    n_estimators=100,
    verbosity=-1
)

# Fit the model
model.fit(X, y)

# Test if the model predicts correctly

def test_model_prediction():
    predictions = model.predict(X)
    expected_predictions = y
    assert np.array_equal(predictions, expected_predictions), f"Model predictions {predictions} do not match expected outputs {expected_predictions}!"

# Run the test
if __name__ == '__main__':
    pytest.main()
