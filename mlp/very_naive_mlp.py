import numpy as np

class MLP:
    def __init__(self, layer_sizes):
        self.weight_shapes = [(layer_sizes[i+1],layer_sizes[i]) for i in range(len(layer_sizes)-1)]

        self.weights = [np.random.standard_normal(s)/np.sqrt(s[1]) for s in self.weight_shapes]
        self.biases = [np.zeros((layer_sizes[i],1)) for i in range(1,len(layer_sizes))]

    def predict(self, a):
        for w, b in zip(self.weights, self.biases):
            a = self.activation((w @ a) + b)

        return a

    @staticmethod
    def activation(x):
        return 1/(1+np.exp(-x))



layer_sizes = (10,1000,1000,10)

mlp = MLP(layer_sizes)
print(mlp.predict(np.ones((layer_sizes[0],1))))


#notice variance drift in this sigmoid neural net, interesting discovery
#Z0 variance: 0.9666736249917937
#Z1 variance: 0.3070157337172165
#Z2 variance: 0.12832707290078227
#this turns out to be well documented and motivates a different activation
#that preserves variance or we can use batch norm