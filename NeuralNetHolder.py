import math
import ast


class NeuralNetHolder:

    def __init__(self):
        super().__init__()
        self.TotalWeights = [6.061566716836983, -0.01566640618554844, 0.448756949780269, 0.40574513923883476,
                             0.12044459269285536, 6.902584591317059, 2.6249695397802504, 2.6503061795908396,
                             1.2155140336776662, -1.1880748532583263, -0.6424915596086512, -0.35579608757866965,
                             -1.7800461331583308, 1.253677544100778, -0.18527739950026015, 0.05672286645914846]
        self.lambda1 = 0.8
        self.x1min = 0.000000
        self.x1max = 1.000000
        self.x2min = 0.002126
        self.x2max = 0.999748
        self.y1min = 0.008565
        self.y1max = 1.000000
        self.y2min = 0.012909
        self.y2max = 0.996773

    def ActivationFunction(self, result_V):
        lambda1 = 0.5
        h = []
        for i in range(len(result_V)):
            try:
                value = 1 / (1 + (math.exp((-lambda1) * result_V[i])))
                h.append(value)
            except OverflowError:
                value = 1 - (1 / (1 + (math.exp((lambda1) * result_V[i]))))
                h.append(value)

        return h

    def WeightMultiplication(self, Input1, Input2, weights_From_Input1, weights_From_Input2):
        Vlist = []
        for i in range(len(weights_From_Input1)):
            v = (float(Input1) * weights_From_Input1[i]) + (float(Input2) * weights_From_Input2[i])
            Vlist.append(v)

        return Vlist

    def UpdatedWeightMultiplication(self, result_h, y, z):
        sum1 = 0
        sum2 = 0
        OutPut1 = []
        for i in range(len(y)):
            sum1 = sum1 + result_h[i] * y[i]

        OutPut1.append(sum1)
        for j in range(len(z)):
            sum2 = sum2 + result_h[j] * z[j]

        OutPut1.append(sum2)

        return OutPut1

    def predict(self, input_row):
        # WRITE CODE TO PROCESS INPUT ROW AND PREDICT X_Velocity and Y_Velocity
        input_row = ast.literal_eval(input_row)
        input_row = list(input_row)
        input_row[0] = (input_row[0] - self.x1min) / (self.x1max - self.x1min)
        input_row[1] = (input_row[1] - self.x2min) / (self.x2max - self.x2min)
        input_row = [input_row[0], input_row[1]]

        # print("3333333Total weights",TotalWeights)
        midpoint = len(self.TotalWeights) // 2
        # Creating a new list of the weights from the total weights list by getting the first half - Weights from 2
        # inputs to the hidden neurons
        Input_to_hidden = self.TotalWeights[:midpoint]
        # Further dividing the list of weights from input to the hidden neurons to specify which weights are coming
        # from which input.
        Hidden_to_output = self.TotalWeights[midpoint:]
        # Specifying which weights are coming from which input.
        midpoint1 = len(Input_to_hidden) // 2
        weights_From_Input1 = Input_to_hidden[:midpoint1]
        weights_From_Input2 = Input_to_hidden[midpoint1:]
        midpoint2 = len(Hidden_to_output) // 2
        y = Hidden_to_output[:midpoint2]
        z = Hidden_to_output[midpoint2:]

        v1 = self.WeightMultiplication(input_row[0], input_row[1], weights_From_Input1, weights_From_Input2)
        h1 = self.ActivationFunction(v1)
        v2 = self.UpdatedWeightMultiplication(h1, y, z)
        y = self.ActivationFunction(v2)
        y[0] = y[0] * (self.y1max - self.y1min) + self.y1min
        y[1] = y[1] * (self.y2max - self.y2min) + self.y2min

        y = [math.log(y[0] / (1 - y[0])), math.log(y[1] / (1 - y[1]))]
        return y
