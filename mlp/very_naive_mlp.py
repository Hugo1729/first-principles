import numpy as np

layer_sizes = (5000,5000,5)

weight_shapes = [(layer_sizes[i+1],layer_sizes[i]) for i in range(len(layer_sizes)-1)]

weights = [np.random.standard_normal(s)/np.sqrt(s[1]) for s in weight_shapes]
biases = [np.zeros((layer_sizes[i],1)) for i in range(1,len(layer_sizes))]

def predict(a):
    for w, b in zip(weights, biases):
        a = activation((w @ a) + b)

    return a

# @staticmethod
def activation(x):
    return 1/(1+np.exp(-x))

print("weights:")

for w in weights:
    print(w)
    print()

print("biases:")

for b in biases:
    print(b)
    print()

print(predict(np.ones((layer_sizes[0],1))))