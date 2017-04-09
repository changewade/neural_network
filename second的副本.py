# -- coding: UTF-8 --
#
#Python的默认编码文件是用的ASCII码，你将文件存成了UTF-8也没用。
#解决办法很简单：
#只要在文件开头加入 # -- coding: UTF-8 -- 或者 #coding=utf-8 就行了。
#
from numpy import exp, array, random, dot

class NeuralLayer():
    def __init__(self, number_of_neurons,number_of_input_per_neuron):

        #对单个神经元建模，含有3个输入连接和一个输出连接
        #对一个3 * 1的矩阵赋予随机权重值。范围-1～1，平均值为0
        #self.synaptic_weights = 2 * random.random((3,1)) - 1

        self.synaptic_weights = 2 * random.random((number_of_input_per_neuron, number_of_neurons)) - 1

class NeuralNetwork():
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2
#############################################

    #Sigmoid函数，S形曲线
    #用这个函数对输入对加权总和做正规化，使其范围在0～1
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    #Sigmoid函数的导数
    #Sigmoid函数的梯度
    #表示我们对当前权重的置信程度
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    #通过试错程序训练神经网络
    #每次都调整突触权重
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):

#############################################
            #将训练集导入神经网络
            #output = self.think(training_set_inputs)

            output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)

            #计算第二层的误差
            layer2_error = training_set_outputs - output_from_layer_2
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)

            #计算第一层的误差，得到第一层对第二层对影响
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            #计算误差（实际值与期望值之差）
            #error = training_set_outputs - output

            #计算权重重调整量
            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            #将误差、输入和S曲线梯度相乘
            #对于置信程度低的权重，调整程度也大
            #为0的输入值不会影响权重
            #adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            #调整权重
            #self.synaptic_weights += adjustment

            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment

    #神经网络一思考
    def think(self, inputs):
        #把输入传递给神经网络
        #return self.__sigmoid(dot(inputs, self.synaptic_weights))
        output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights))
        output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
        return output_from_layer1, output_from_layer2
    #输出权重
    def print_weights(self):
        print "Layer 1(4 neurons, each with 3 inputs:"
        print self.layer1.synaptic_weights
        print "Layer 2(1 neuron, each with 4 inputs:"
        print self.layer2.synaptic_weights

if __name__ == "__main__":

    #设定随机种子
    random.seed(1)

    #创建第一层（4神经元，每个3输入）
    layer1 = NeuralLayer(4,3)

    # 创建第二层（单神经元，每个4输入）
    layer2 = NeuralLayer(1, 4)

    #组合成神经网络
    neural_network = NeuralNetwork(layer1, layer2)

    #初始化神经网络
    #neural_network = NeuralNetwork()

    print "Stage1) 随机的初始突触权重："
    #print neural_network.synaptic_weights
    neural_network.print_weights()

    #训练集。七个样本，每个有3个输入和1个输出
    print "训练集为[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]，[0, 0, 0], [1, 0, 0]"
    print "训练集答案为[0, 1, 1, 0，0，1].T"
    training_set_inputs = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    training_set_outputs = array([[0, 1, 1, 1, 1, 0, 0]]).T

    #用训练集训练神经网络
    #重复60000次，每次做微小的调整
    neural_network.train(training_set_inputs, training_set_outputs, 60000)

    print "Stage2) 训练后的突触权重："
    neural_network.print_weights

    #用新数据测试神经网络
    print "stage3) 考虑新的形势 【1，1，0】 -> ?: "
    hidden_stage, output = neural_network.think(array([1, 1, 0]))
    print output
