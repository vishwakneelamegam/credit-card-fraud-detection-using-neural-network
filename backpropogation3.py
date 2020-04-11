#numpy for calculation
import numpy as np
from csv import reader
# read csv file as a list of lists
data = []
data1 = []
data2 = []
with open('nonfraud.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    data1 = list(csv_reader)
data1.pop(0)
with open('fraud.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    data2 = list(csv_reader)
data2.pop(0)
for i in range(0,49):
    data.append(data1[i])
    data.append(data2[i])
#print(data)
#sigmoid function ranges squash value x between 0 and 1
def sigmoid(x):
    return 1/(1+np.exp(-x))

#sigmoid_p is for derived sigmoid 0.25 to 0
def sigmoid_p(x):
    return sigmoid(x) * (1-sigmoid(x))

#neuralNetwork function
def neuralNetwork(ipList,weightList,b):
    result = 0
    for i in range(0,30):
        result += float(ipList[i]) * weightList[i]
    return result + b

#trainData to train data
def trainData():
    output = []
    weights = []
    cost_w = []
    for _ in range(0,30):
        weights.append(np.random.rand())
        cost_w.append(0)
        output.append(0)
    b  = np.random.randn()
    #initialize iteration and learning rate
    iteration = 100000
    rate = 0.1
    try:
        for i in range(iteration):
            randomInput = np.random.randint(len(data))
            op      = neuralNetwork(data[randomInput],weights,b)
            predict = sigmoid(op)
            cost_predict = 2.0 * (float(predict) - float(data[randomInput][30]))
            predict_op   = sigmoid_p(op)
            for i in range(0,30):
                cost_w[i]      = float(cost_predict) * float(predict_op) * float(data[randomInput][i])
            cost_b       = float(cost_predict) * float(predict_op) * 1.0
            for i in range(0,30):
                weights[i] = float(weights[i]) - float(rate) * float(cost_w[i])
            b  = float(b)  - rate * float(cost_b)
        for i in range(len(data)):
            print("--------------------")
            print("class : " + str(data[i][30]))
            print(sigmoid(neuralNetwork(data[i],weights,b)))
    except Exception as e:
        return str(e)

trainData()
