import os
import CNN

_HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.normpath(os.path.join(_HERE, 'test'))
_PARAMS = os.path.join(_HERE, 'cnn_params.npz')

if __name__ == '__main__':
    params = CNN.load_params(_PARAMS)
    CNN.interview(EVAL_DIR, params)
