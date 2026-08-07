import numpy as np

class NMLP:
    def __init__(self, layer_sizes):
        self.weight_shapes = [(layer_sizes[i+1],layer_sizes[i]) for i in range(len(layer_sizes)-1)]

        self.layer_sizes = layer_sizes

        self.weights = [np.random.standard_normal(s)/np.sqrt(s[1]) for s in self.weight_shapes]
        self.biases = [np.zeros((layer_sizes[i],1)) for i in range(1,len(layer_sizes))]
        self.a = [np.zeros(s) for s in self.layer_sizes]
        self.z = [np.zeros(s) for s in self.layer_sizes]

        #dJ/dz for each layer
        self.partial = [np.zeros(s) for s in self.layer_sizes]

    def forward(self, x):
        self.a[0] = x
        for i, w, b in zip(range(len(self.weights)), self.weights, self.biases):
            self.z[i+1] = (w @ self.a[i]) + b
            self.a[i+1] = self.sigmoid(self.z[i+1])

    def predict(self, x):
        self.forward(x)
        return self.a[-1]

    #applies stochastic gradient descent
    def learn(self, x, y, alpha):
        self.forward(x)

        self.calculate_partials(y)

        for i in range(len(self.weights)):
            b_grad = self.partial[i+1]
            w_grad = self.partial[i+1] @ np.transpose(self.a[i])

            self.weights[i] -= alpha * w_grad
            self.biases[i] -= alpha * b_grad

    def calculate_final_layer_partial(self, y):
        self.partial[-1] = 2*(self.a[-1]-y)
        self.partial[-1] *= self.sigmoid(self.z[-1]) * (1-self.sigmoid(self.z[-1]))

    def calculate_partials(self, y):
        self.calculate_final_layer_partial(y)

        for i in range(len(self.partial)-2,0,-1):
            self.partial[i] = np.transpose(self.weights[i]) @ self.partial[i+1]
            
            self.partial[i] *= self.sigmoid(self.z[i]) * (1-self.sigmoid(self.z[i]))

    @staticmethod
    def sigmoid(x):
        # return (np.exp(x)-np.exp(-x)) / (np.exp(x)+np.exp(-x))
        return 1/(1+np.exp(-x))

    @staticmethod
    def cost(predicted, true):
        return np.sum((predicted - true) ** 2)

    def loss(self, X, Y):
        J = 0

        for x, y in zip(X, Y):
            J += self.cost(self.predict(x), y)

        return J

    def fit(self, X, Y, alpha, epocs):
        self.training_loss = []

        for iter in range(epocs):
            for x, y in zip(X, Y):
                self.learn(x, y, alpha * (1/X.shape[0]))

            self.training_loss.append(self.loss(X, Y))
            print("#After iteration", iter, "loss is:", self.training_loss[-1])

        return self

layer_sizes = (2,10,10,2)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape(4,2,1)
Y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]]).reshape(4,2,1)

mlp = NMLP(layer_sizes)
print(mlp.predict(np.ones((layer_sizes[0],1))))
print()
print("Loss:", mlp.loss(X,Y))

mlp.fit(X, Y, 1, 300000)


for x in X:
    print("input", x)
    print("output", mlp.predict(x))

import matplotlib.pyplot as plt

plt.plot(mlp.training_loss)
plt.show()