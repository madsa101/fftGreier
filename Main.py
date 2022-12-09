import numpy as np
import scipy as scp
from scipy import fftpack
from matplotlib import pyplot as plt
from sklearn.preprocessing import scale


def getAvgs(file):  # outputs high/low avgs for each kline and sends to npArray:
    output = []
    for line in file:
        high = float(line.split(',')[2])
        low = float(line.split(',')[3])
        output.append((high + low) / 2)
    return np.array(output)


def percentWise(array1d):  # outputs percentwise rise/decline of value compared to the previous
    output = []
    index = 0
    for i in range(array1d.size):
        if i == 0:
            output.append(0)
        else:
            output.append(round((array1d[i] / array1d[i - 1]) - 1, 6))
        index += 1
    return np.array(output)


def fftWithDenoise(vector1d, time, limit):
    index = 0
    segment = vector1d[index:index + time]
    fft = np.fft.fft(segment[0:time])
    freqDom = abs(fft * np.conj(fft) / time)
    limit = float(np.amax(freqDom)) * limit
    freq = (1 / time) * np.arange(time)

    freqDom0 = np.where(freqDom < limit, 0, freqDom)
    fft = np.where(freqDom < limit, 0, fft)
    ifftDenoised = np.fft.ifft(fft)
    ifftDenoised = ifftDenoised.real
    return np.array([ifftDenoised, freq, fft, freqDom, freqDom0])


n = 3
file = open("LINKUSDT-1m-2022-11.csv")
values = percentWise(getAvgs(file))
limit = 1/n
"""
# test
vectors = fftWithDenoise(segment, fftSize)
plotVector = np.hstack((vectors[0], vectors[0][:10]))
plt.plot(x, plotVector)
plt.plot(x, segment)
plt.show()
"""
totalAvg = []
for i in range(10, 100):
    disparities = []
    index = 1

    while index < (values.size - i):
        segment = values[index:index + i]
        vectors = fftWithDenoise(segment, i-1, limit)
        plotVector = np.hstack((vectors[0], vectors[0][:1]))
        plotVector = plotVector.real

        if ((plotVector[-1] < 0 and segment[-1] < 0) or (plotVector[-1] > 0 and segment[-1] > 0)):
            disparities.append(100)
        elif ((plotVector[-1] < 0 and segment[-1] > 0) or (plotVector[-1] > 0 and segment[-1] < 0)):
            disparities.append(0)

        """
        plt.plot(x, plotVector, color='r', label='ifft')
        plt.plot(x, segment, color='b', label='percent')
        plt.plot(x, np.zeros(segSize))
        plt.pause(0.1)
        plt.clf()
        """
        index += 1

    length = len(disparities)
    if len(disparities) == 0:
        length = 1
    avgDisp = (sum(disparities)/length)
    print("Accuracy for samplesize " + str(i) + ": " + str(avgDisp))
    if i % 10 == 0:
        totalAvg.append([])
        totalAvg[-1].append(avgDisp)
    else:
        totalAvg[-1].append(avgDisp)

for list in totalAvg:
    print(str(totalAvg.index(list)) + ": " + str(sum(list)/len(list)))
