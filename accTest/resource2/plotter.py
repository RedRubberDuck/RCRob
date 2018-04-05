import json
from matplotlib import pyplot as plt


filename = 'datKalmanF2'


def readFromFile():

    fileName = filename+'.csv'
    fileIn = open(fileName, 'r')
    data = fileIn.read()

    data = data.replace("'", '"')
    data = data.replace("True", '1')
    data = data.replace("False", '0')
    data = data.replace("(", '[')
    data = data.replace(")", ']')
    dataJson = json.loads(data)
    fileIn.close()
    return dataJson


def main():
    s = 0
    dataJson = readFromFile()

    X_acc = []
    Y_acc = []
    Z_acc = []

    dataJ = []
    for dataI in dataJson:
        dataV = {}

        dataV['gyro'] = dataI['gyro']
        dataV['accel'] = dataI['accel']
        dataV['compass'] = dataI['compass']
        dataV['timestamp'] = dataI['timestamp']
        dataV['fusionPose'] = dataI['fusionPose']

        dataJ.append(dataV)
    # plt.figure()
    # plt.subplot(311)
    # plt.plot(X_acc)
    # plt.subplot(312)
    # plt.plot(Y_acc)
    # plt.subplot(313)
    # plt.plot(Z_acc)
    # plt.show()

    # print(dataJ)
    output = json.dumps(dataJ)

    outFile = open(filename+'.json', 'w')
    outFile.write(output)
    outFile.close()


if __name__ == "__main__":
    main()
