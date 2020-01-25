import argparse
import os
import wave
import soundfile as np
import numpy
from pydub import AudioSegment
from matplotlib import pyplot as plt
import scipy.io.wavfile

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

# Make a prediction with weights
def predict(row, weights):
    activation = weights[0]
    for i in range(len(row) - 1):
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else 0.0

# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, l_rate, n_epoch):
    weights = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0.0
        for row in train:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            sum_error += error ** 2
            weights[0] = weights[0] + l_rate * error
            for i in range(len(row) - 1):
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    return weights

parser = argparse.ArgumentParser(description="Separating silent waves from louder waves")
parser.add_argument('-p', '--path', type=dir_path, metavar='', required=True, help="give path to the file with waves.")
args = parser.parse_args()

if __name__ == '__main__':
    os.chdir(args.path)
    print(os.getcwd() + "\n\n")
    training_list = []
    testing_list = []
    sciList = []
    trainbool = []


    flag = True
    for root, sub, files in os.walk(args.path):
        for f in files:

            if 'wav' in f:
                abs_path = os.path.abspath(root)
                # Populate training list
                if abs_path.endswith('silence_training'):
                    samplerate, data = scipy.io.wavfile.read(os.path.join(abs_path, f))
                    for i in data:
                        if i<0:
                    training_list.append([numpy.sum(data),0])
                    trainbool.append(-1000000)

                elif abs_path.endswith('voiced_training'):
                    samplerate, data = scipy.io.wavfile.read(os.path.join(abs_path, f))
                    training_list.append([numpy.sum(data),1])
                    trainbool.append(1000000)

                # Populate testing list
                elif abs_path.endswith('silence'):
                    pass
                elif abs_path.endswith('voiced'):
                    pass


#numpy.random.shuffle(training_list)

for row in training_list:
    print(row)

plt.plot(training_list,'r')
plt.plot(trainbool)
plt.show()

l_rate = 0.01
n_epoch = 10000
#weights = train_weights(training_list,l_rate, n_epoch)

#for row in training_list:
#    print(predict(row,[5,0.2,0.2]))