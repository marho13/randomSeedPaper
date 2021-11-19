import matplotlib.pyplot as plt
import re
import numpy as np
import os

bestSeeds = [694917, 113557, 226206, 40383, 912849, 29179, 659976, 736984, 508253, 254464, 2533, 529007]
worstSeeds = [847475, 199636, 513294, 358264, 724059, 677934, 165072, 7782, 648138, 113326, 610725, 312899]

def listNSeeds(list):
    output = [("randomnessData/rewards{}.txt".format(seed)) for seed in list]
    return output

def readFile(fileName):
    return open(fileName).read()

def splitNewLine(text):
    return text.split("\n")

def splitComma(text):
    output = []
    del text[-1]
    for line in text:
        output.append([])
        comma = line.split(",")
        del comma[-1]
        for c in comma:
            output[-1].append(float(c))
    return np.asarray(output)

def createSeedLists(text):
    output = []
    print(len(text), len(text[0]))
    for _ in text[0]:
        output.append([])
    for t in text:
        for i in range(len(t)):
            output[i].append(t[i])
    print(len(output), len(output[0]))
    return output

def plotter(data, files):
    plt.clf()
    for a in range(len(data)):
        this = []
        for d in data[a]:#[a]:
            temp = sum(d)/len(d)
            this.append(temp)
        plt.plot(this, label="{}".format(files[a]))# plt.plot(this, label="{}".format(files[a]))
        print(files[a])
    # plt.plot(this)
    plt.ylabel("Reward")
    plt.xlabel("Backpropagation")
    leg = plt.legend(loc='best')
    plt.show()

def calcSum(data):
    output = 0.0
    for d in data:
        for i in d:
            output += i
    return output

# files = os.listdir("randomnessData")
# listofFiles = [("randomnessData/{}".format(f)) for f in files]
best = listNSeeds(bestSeeds)
worst = listNSeeds(worstSeeds)


def formatFiles(list, bw):
    totRewards = []
    plottingFiles = []
    for l in list:
        print("Seed {}: gives avg reward of".format(l), end=" ")
        filey = readFile(l)
        lines = splitNewLine(filey)
        linesComma = splitComma(lines)
        totSum = (calcSum(linesComma))
        totRewards.append(totSum)
        print(totRewards[-1])
        plottingFiles.append(linesComma)

    plotter(plottingFiles, bw)
    print(max(totRewards))
    print(min(totRewards))
    print(sum(totRewards)/len(totRewards))
    print(plottingFiles[0].shape)

formatFiles(best, bestSeeds)
formatFiles(worst, worstSeeds)
pass
# maxy = findMaxVal(splitFile)
# miny = findMinVal(splitFile)
# distance = (maxy - miny)/1000
# print(splitFile)
# removed = removeExtremes(splitFile)
# # normalised = normalise(splitFile, maxy)
# # print(normalised)
#
# plotter(removed)