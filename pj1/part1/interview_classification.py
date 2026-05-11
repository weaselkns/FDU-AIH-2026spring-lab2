import os
import classification

_HERE = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(_HERE, "test")
PARAMS_PATH = os.path.join(_HERE, "classification_params.npz")

if __name__ == "__main__":
    params, layer_dims = classification.load_params(PARAMS_PATH)
    classification.interview(TEST_DIR, layer_dims, params)