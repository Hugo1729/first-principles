import numpy as np

class Value:
    def __init__(self, val):
        self._val = np.float64(val)
        self._grad = np.float64(0)
        self._children = []

    def __str__(self):
        return str(self._val)

    def __mul__(self, other):
        ans = Value(self._val * other._val)
        ans._grad = self._grad + other._grad
        ans._children.append(self, other)

layer_sizes = (2,3,3)

weight_shapes = [(layer_sizes[i+1],layer_sizes[i]) for i in range(len(layer_sizes)-1)]

W = [(np.random.standard_normal(s)/np.sqrt(s[1])).astype(Value) for s in weight_shapes]
b = [np.zeros((layer_sizes[i],1)) for i in range(1,len(layer_sizes))]

print(Value(5))


z = np.ones(2).astype(Value)

a = W[0] @ z

print(a)

print(a[0]._grad)



class MLP:
    def __init__(self):
        pass
