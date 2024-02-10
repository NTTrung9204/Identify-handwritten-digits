import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def one_hot_encode(num_classes, input_values):
    encoded_values = np.zeros((len(input_values), num_classes))
    for i, value in enumerate(input_values):
        encoded_values[i, value] = 1 
    return encoded_values

def Vectorize(context):
    vector_text = [0]*len(vectorizer_text)
    for text in str(context[0]).split():
        if text in vectorizer_text:
            vector_text[vectorizer_text.index(text)] += 1
    return vector_text

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
        self.W1    = np.random.rand(len(self.X[0]), self.Layers)
        self.Bias1 = np.random.rand(1, self.Layers)
        self.W2    = np.random.rand(self.Layers, len(self.Y[0]))
        self.Bias2 = np.random.rand(1, len(self.Y[0]))

        self.HistoryCost = [5]

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
                sys.stdout.write("\rEpoches : {} | {}/{} | Loss: {}".format(Epoch + 1, Index + 1, len(self.X) // self.BatchSize + 1, self.HistoryCost[-1]))
            self.HistoryCost.append(Cost/len(self.X))
    
    def Accuracy(self, XTest, YTest):
        Result = self.Predict(XTest)
        Accuracy = accuracy_score(np.argmax(Result, axis=1), np.argmax(YTest, axis=1))
        print(f"{100*Accuracy:.2f}%")

    def ShowCost(self):
        print("\ncost:",self.HistoryCost[-1])
        plt.plot(self.HistoryCost)
        plt.show()
    

df = pd.read_csv('https://raw.githubusercontent.com/amankharwal/Website-data/master/dataset.csv')

df["language"], _ = pd.factorize(df["language"])

X = df.loc[:, ["Text"]].values
Y = df.loc[:, ["language"]].values

vectorizer_text = []

for context in X:
    for text in str(context[0]).split():
        if text not in vectorizer_text: vectorizer_text.append(text)

X_pre = []

for index in range(len(X)):
    sys.stdout.write(f"\rInteration Vectorize: {index+1} | {len(X)}")
    X_pre.append(Vectorize(X[index]))

X_pre = np.array(X_pre)
Y_pre = one_hot_encode(22, Y)

if __name__ == "__main__":
    X_train, X_test, Y_train, Y_test = train_test_split(X_pre, Y_pre, test_size=0.2, random_state=43)

    Model = MultilayerNeuralNetWork(X_train, Y_train, Layers=10, Epoches=10, BatchSize=256, LR=0.0001)

    print()
    print(sum(X_train[0]), X_train[0])
    print()

    Model.FitData()

    np.savetxt("W1", Model.W1)
    np.savetxt("W2", Model.W2)
    np.savetxt("Bias1", Model.Bias1)
    np.savetxt("Bias2", Model.Bias2)
    Model.ShowCost()

    Model.Accuracy(X_train, Y_train)
    Model.Accuracy(X_test, Y_test)

    text_test = Vectorize("Hello, My name is Trung")
    result = Model.Predict(text_test)
    print(np.argmax(result))

    print(np.array(vectorizer_text).shape)