import numpy as np

class VNMLP:
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

    @staticmethod
    def cost(c, y):
        return np.sum((c - y) ** 2)

    def loss(self, X, Y):
        J = 0

        for x, y in zip(X, Y):
            J += self.cost(self.predict(x), y)

        return J

    def fit(self, X, Y, alpha, h, epocs):
        for iter in range(epocs):
            w_grad = [np.zeros(s) for s in self.weight_shapes]
            b_grad = [np.zeros((layer_sizes[i],1)) for i in range(1,len(layer_sizes))]

            loss1 = self.loss(X, Y)    #f(x)

            for l in range(len(w_grad)):
                for b in range(len(b_grad[l])):
                    self.biases[l][b] += h
                    loss2 = self.loss(X, Y)   #f(x+h)
                    self.biases[l][b] -= h

                    b_grad[l][b] += (loss2 - loss1) / h

                for i in range(w_grad[l].shape[0]):
                    for j in range(w_grad[l].shape[1]):
                        self.weights[l][i][j] += h
                        loss2 = self.loss(X, Y)     #f(x+h)
                        self.weights[l][i][j] -= h
    
                        w_grad[l][i][j] += (loss2 - loss1) / h

            for l in range(len(w_grad)):
                self.biases[l] -= alpha * (1/X.shape[0]) * b_grad[l]  # mult with alpha and 1/4
                self.weights[l] -= alpha * (1/X.shape[0]) * w_grad[l]

            print("#After iteration", iter, "loss is:", self.loss(X, Y))

        return self

layer_sizes = (2,3,2)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape(4,2,1)
Y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]]).reshape(4,2,1)

mlp = VNMLP(layer_sizes)
print(mlp.predict(np.ones((layer_sizes[0],1))))
print()
print("Loss:", mlp.loss(X,Y))

mlp.fit(X, Y, 1, 1e-7, 100)
print(mlp.predict(np.array([1,0]).reshape(2,1)))


#notice variance drift in this sigmoid neural net, interesting discovery
#Z0 variance: 0.9666736249917937
#Z1 variance: 0.3070157337172165
#Z2 variance: 0.12832707290078227
#this turns out to be well documented and motivates a different activation
#that preserves variance or we can use batch norm