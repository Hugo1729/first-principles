import numpy as np

class Value:
    def __init__(self, val):
        self.val = val
        self._parents = []
        self.grad = 0.0
        self._backward = None

    def __repr__(self):
        return f"Value({self.val})"

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        ans = Value(self.val * other.val)
        ans._parents += [self, other]

        def _backward():
           self.grad += ans.grad * other.val
           other.grad += ans.grad * self.val

        ans._backward = _backward

        return ans

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        ans = Value(self.val + other.val)
        ans._parents += [self, other]

        def _backward():
            self.grad += ans.grad
            other.grad += ans.grad

        ans._backward = _backward

        return ans

    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        ans = Value(self.val ** other.val)
        ans._parents += [self, other]

        def _backward():
            self.grad += ans.grad * other * (self.val ** (other - 1))
            other.grad += ans.grad * ans.val * np.

        ans._backward = _backward

        return ans

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __neg__(self): # -self
        return self * -1

    def __sub__(self, other): # self - other
        return self + (-other)

    def __radd__(self, other): # other + self
        return self + other

    def backward(self):
        nodes = []
        visited = set()
        def top_sort(v):
            if (v in visited):
                return
            
            visited.add(v)
            for w in v._parents:
                top_sort(w)

            nodes.append(v)

        self._grad = 1.0
        top_sort(self)

        for v in reversed(nodes):
            v._backward()



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
