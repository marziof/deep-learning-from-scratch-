import numpy as np

# class tanh_activation:
#     def forward(x):
#         return np.tanh(x)

#     def backward(y):
#         # receives y = tanhx
#         return 1-y**2


class tanh_activation:
    @staticmethod
    def forward(x):
        return np.tanh(x)

    @staticmethod
    def backward(y):
        # y = tanh(a)
        return 1 - y**2