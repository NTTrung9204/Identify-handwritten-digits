import numpy as np
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2 

class MultilayerNeuralNetWork:
    def __init__(self, X, Y, Layers = 20, Epoches = 100, BatchSize = 128, LR = 0.001):
        self.X = X
        self.Y = Y
        self.Layers = Layers
        self.Epoches = Epoches
        self.BatchSize = BatchSize
        self.LR = LR
        self.Initialize()

    def Initialize(self):
        self.W1    = np.random.rand(len(self.X[0]), self.Layers) / 10
        self.Bias1 = np.random.rand(1, self.Layers) / 10
        self.W2    = np.random.rand(self.Layers, len(self.Y[0])) / 10
        self.Bias2 = np.random.rand(1, len(self.Y[0])) / 10

        self.HistoryCost = []

    def ReLU(self, Z):
        return np.maximum(Z, 0)
    
    def SoftMax(self, Z):
        E_Z = np.exp(Z - np.max(Z, axis = 1, keepdims = True))
        return E_Z / E_Z.sum(axis = 1, keepdims = True)
    
    def Predict(self, X):
        Z1 = X @ self.W1 + self.Bias1  
        A1 = self.ReLU(Z1)       
        Z2 = A1 @ self.W2 + self.Bias2       
        A2 = self.SoftMax(Z2)  
        return A2                      
    
    def FitData(self):
        for Epoch in range(self.Epoches):
            MixData = np.random.permutation(len(self.X))
            Cost = 0
            for Index in range(len(self.X) // self.BatchSize + 1):
                Batch = MixData[self.BatchSize * Index : min(self.BatchSize * (Index + 1), len(self.X))]

                # FeedForward
                Z1 = self.X[Batch] @ self.W1 + self.Bias1    # (N, d) . (d, h) -> (N, h)
                A1 = self.ReLU(Z1)                           # (N, h)
                Z2 = A1 @ self.W2 + self.Bias2               # (N, h) . (h, C) -> (N, C)
                A2 = self.SoftMax(Z2)                        # (N, C)

                # Backpropagation
                E2     = (A2 - self.Y[Batch])                       # (N, C)
                DW2    = A1.T @ E2                                  # (h, N) . (N, C) -> (h, C)
                DBias2 = np.sum(E2, axis = 0) / self.BatchSize      # (N, C) -> (1, C)
                E1     = (self.W2 @ E2.T)                           # (h, C) . (C, N) -> (h, N)
                E1[Z1.T <= 0] = 0                                   # (h, N)
                DW1    = self.X[Batch].T @ E1.T                     # (d, N) . (N, h) -> (d, h)
                DBias1 = np.sum(E1.T, axis = 0) / self.BatchSize    # (1, h)

                # Gradient Descent Update
                self.W1    -= self.LR * DW1
                self.Bias1 -= self.LR * DBias1
                self.W2    -= self.LR * DW2
                self.Bias2 -= self.LR * DBias2

                # History Cost
                Cost -= np.sum(self.Y[Batch] * np.log(A2))

                # Training
                sys.stdout.write("\rEpoches : {} | {}/{}".format(Epoch, Index, len(self.X) // self.BatchSize + 1))

            self.HistoryCost.append(Cost/len(self.X))
    
    def Accuracy(self, XTest, YTest):
        Result = self.Predict(XTest)
        Accuracy = 0
        for Index in range(len(YTest)):
            if Result[Index][YTest[Index]] == max(Result[Index]):
                Accuracy += 1
        print("\nAccurate :",Accuracy / len(YTest) * 100)

    def ShowCost(self):
        print("\ncost:",self.HistoryCost[-1])
        plt.plot(self.HistoryCost)
        plt.show()
    
if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape(60000, -1)/255
    train_labels = train_labels.reshape(60000, -1)
    test_images = test_images.reshape(10000, -1)/255
    test_labels = test_labels.reshape(10000, -1)

    m = len(train_labels)  # Số hàng của ma trận
    n = 10  # Số cột của ma trận, cộng 1 để tính cả vị trí 0

    # Tạo ma trận one-hot
    test_labels_onehot = np.zeros((m, n))
    test_labels_onehot[np.arange(m), train_labels.flatten()] = 1

    Model = MultilayerNeuralNetWork(train_images, test_labels_onehot, 100, 100, 256, 0.0001)

    Model.FitData()

    np.savetxt("W1", Model.W1)
    np.savetxt("W2", Model.W2)
    np.savetxt("Bias1", Model.Bias1)
    np.savetxt("Bias2", Model.Bias2)
    Model.ShowCost()

    Model.Accuracy(test_images, test_labels)
    Model.Accuracy(train_images, train_labels)

    # Model.W1     = np.loadtxt("W1")
    # Model.W2     = np.loadtxt("W2")
    # Model.Bias11 = np.loadtxt("Bias1")
    # Model.Bias21 = np.loadtxt("Bias2")

    img = cv2.imread('my_image0.png',0)

    TestImage  = img.reshape(1, -1) / 255

    Result = Model.Predict(TestImage)

    print(Result)
    print("Result: ", np.argmax(Result))