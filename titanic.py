import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import random

import math

#get data
trainingData = pd.read_csv('titanic/train.csv', delimiter = ',')
testData = pd.read_csv('titanic/test.csv', delimiter = ',')

#clean up data to be used
cleanData = trainingData.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
shortData = testData.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
maxVar = int(math.sqrt(cleanData.shape[1]-1))

#value 5 is arbritary
shortData.insert(0, 'Survived', 5)

#implement machine learning model

#list for decision trees
submitTree = list()
submit = list()
finalParsed = list()
final = list()
train = list()
numberOfWays = dict()
chooseCol = []

indexToName = list(cleanData.columns)

def init():
    #by index - e.g. 1 -> Pclass
    numberOfWays[1]=[1,2,3]
    numberOfWays[2]=['male', 'female']
    numberOfWays[3]=[(0,29),(30,59),(60,120)]
    numberOfWays[4]=[(0,2), (3,8)]
    numberOfWays[5]=[(0,2), (3,6)]
    numberOfWays[6]=[(0,200),(201,1000)]
    numberOfWays[7]=['C', 'Q', 'S']
    
 
    #...

def calculate(alive, dead):
    #so if Pclass has lowest gigiIndex(one case - 2); the next one will again just give 1.01 cause the subset is empty, so that means that the leaf node is a pclass of 2; Which is correct! And thus this works
    if(alive == 0 and dead == 0):
        return 1.01
    giniIndex = 1 - ((alive)/(alive+dead))**2 - ((dead)/(alive+dead))**2
    return giniIndex

def GiniIndex(subset, icol):
    alive = subset.loc[subset["Survived"]==1]
    dead = subset.loc[subset["Survived"]==0]
    #this removes the rows with NaN values so we don't have trouble comparing; We need to do this again because this is used to calculate giniImpurity for new columns for sub-subset
    print(indexToName[icol])
    a = alive.loc[alive[indexToName[icol]].notnull()]
    d = dead.loc[dead[indexToName[icol]].notnull()]
    print(a)
    print(d)
    total = 0
    if(icol==1 or icol==2 or icol == 7):
        for i in range(len(numberOfWays[icol])):
            g = len(a.loc[a[indexToName[icol]]==numberOfWays[icol][i]].index)
            h = len(d.loc[d[indexToName[icol]]==numberOfWays[icol][i]].index)
            total += calculate(g,h)
    else:
        for i in range(len(numberOfWays[icol])):
            #get all the data that != NaN
            b = a.loc[a[indexToName[icol]] >= numberOfWays[icol][i][0]]
            c = len(b.loc[b[indexToName[icol]] <= numberOfWays[icol][i][1]].index)
            e = d.loc[d[indexToName[icol]] >= numberOfWays[icol][i][0]] 
            f = len(e.loc[e[indexToName[icol]] <= numberOfWays[icol][i][1]].index)
            total += calculate(c,f)
    return total/len(numberOfWays[icol])

def whichCol(subset):
    col = []
    pickCol=[]
    pickCol = chooseCol.copy()
    if(len(chooseCol)>maxVar):
        for i in range(maxVar):
            num = random.choice(pickCol)
            pickCol.remove(num)
            col.append(num)
    else:
        col = pickCol.copy()
    #We need 1.02 to be > 1.01 because if the GiniIndex gives 1.01 and the minimum = 1.01 throughout each iteration of the loop; bestCol will = None. This will give an error. However, this doesn't affect anything because when bestCol is returned to BuildTree, it's GiniIndex will be checked. And if it equals 1.01, then it will be greater than the previous node; the leaf node. 
    minimum = 1.02
    bestCol = None
    for a in range(len(col)):
        if(minimum!=min(minimum, GiniIndex(subset, col[a]))):
            minimum = GiniIndex(subset, col[a])
            bestCol = col[a]
    print(col)
    print(bestCol)
    print(chooseCol)
    chooseCol.remove(bestCol)
    return bestCol

def decideSurvived(subset):
    alive = len(subset.loc[subset["Survived"]==1].index)
    dead = len(subset.loc[subset["Survived"]==0].index)
    if(alive>dead):
        train.append(1)
    elif(dead>alive):
        train.append(0)
    else:
        guess = random.randint(0, 1)
        train.append(guess)

def BuildTree(startNode, subset, leafNode): 
    cases = numberOfWays[startNode]
    #every time a new node comes - startNode -> we need to cut the rows the NaN values so we can compare with cases
    notNull = subset.loc[subset[indexToName[startNode]].notnull()]
    for i in range(len(cases)):
        #append to our list of decision trees
        train.append((startNode, cases[i]))
        #discrete
        if(startNode == 1 or startNode == 2 or startNode == 7):
            nextSubset = notNull.loc[notNull[indexToName[startNode]]==cases[i]]
        #continuous
        else:
            forward = notNull.loc[notNull[indexToName[startNode]] >= cases[i][0]]
            nextSubset =  forward.loc[forward[indexToName[startNode]] <= cases[i][1]]
        if(len(chooseCol)==0):
            #sorry but there is nothing else to choose for you
            decideSurvived(nextSubset)
            train.append("^")
            continue
        #col = find whichCol of this subset
        col = whichCol(nextSubset)
        if(leafNode<=GiniIndex(nextSubset, col)):
            decideSurvived(nextSubset)
            train.append("^")
            chooseCol.append(col)
            continue
        leafNode = GiniIndex(nextSubset, col)
        #and then BuildTree(col, subset)
        BuildTree(col, nextSubset, leafNode)
        

#this function is the start of making our random forest
def DecisionTree(numberTrain):
    init()
    for i in range(numberTrain):
        train.clear()
        empty = list()
        #replace with 1,2,3,4,5,6,7 -> only for testing
        chooseCol.extend([1,2,3,4,5,6,7])
        for i2 in range(cleanData.shape[0]):
            rowNum = random.randint(1, cleanData.shape[0]-1)
            row = cleanData.iloc[rowNum]
            empty.append(row)
        bootstrappedData = pd.DataFrame(empty)
        startNode = whichCol(bootstrappedData)
        BuildTree(startNode, bootstrappedData, 1.01)    
        final.append(train) 

def parseData(final):
    for i in range(len(final)):
        DecisionTree = []
        startingIndex = 0
        for a in range(len(final[i])):
            if(final[i][a]=="^"):
                DecisionTree.append(final[i][startingIndex:a])
                startingIndex = a + 1
        finalParsed.append(DecisionTree)

def findCol(arr, i, count, col):
    while(True):
        if(count>len(arr[i])-1):
            # 100 is an arbritary number
            return 100
        elif(arr[i][count][0][0]==col):
            return count
        count+=1

#this function is for a row of the data; arr = finalParsed
def predict(arr, row):
    #number of decision trees
    for i in range(len(arr)):
        #arr[i] is a decision tree
        count = 0
        index = 0
        while(True):
            if(count==100):
                guess = random.randint(0,1)
                submit.append(guess)
                break
            elif(type(arr[i][count][index]) is not tuple):
                submit.append(arr[i][count][index])
                break
            else:
                if(pd.isnull(row[arr[i][count][index][0]])==True):
                    #input random data in table
                    if(arr[i][count][index][0]==1 or arr[i][count][index][0]==2 or arr[i][count][index][0]==7):
                        num = random.randint(0, len(numberOfWays[arr[i][count][index][0]])-1)
                        row[arr[i][count][index][0]] = numberOfWays[arr[i][count][index][0]][num]
                    else:
                        num = random.randint(0, len(numberOfWays[arr[i][count][index][0]])-1)
                        findInRange = random.randint(numberOfWays[arr[i][count][index][0]][num][0], numberOfWays[arr[i][count][index][0]][num][1])
                        row[arr[i][count][index][0]] = findInRange
                #discrete
                if(arr[i][count][index][0]==1 or arr[i][count][index][0]==2 or arr[i][count][index][0]==7):
                    #check if satisfies constraint; then index+=1
                    if(row[arr[i][count][index][0]]==arr[i][count][index][1]):
                        index += 1
                    #else reset index to 0; and set findCol; loop until arr[i][count][index][0]==findCol where count+=1 every iteration
                    else:
                        col = arr[i][count][index][0]
                        count += 1
                        nextCase = findCol(arr,i,count,col)
                        count = nextCase
                        index = 0
                #continuous
                else:
                    if(row[arr[i][count][index][0]]>=arr[i][count][index][1][0] and row[arr[i][count][index][0]] <= arr[i][count][index][1][1]):
                        index += 1
                    else:
                        col = arr[i][count][index][0]
                        count += 1
                        nextCase = findCol(arr,i,count,col)
                        count = nextCase
                        index = 0

def startPrediction():
    #use shortData; iterate through rows and pass to predict function
    for i in range(shortData.shape[0]):
        predict(finalParsed, shortData.iloc[i, :])

#num = number of decision trees
def majority(num):
    for i in range(int(len(submit)/num)):
        count_1 = 0
        count_0 = 0
        minIndex = num*i
        maxIndex = num*i + num
        predictedVals = submit[minIndex: maxIndex]
        for i2 in range(len(predictedVals)):
            if(predictedVals[i2]==1):
                count_1 += 1
            else:
                count_0 +=1
        if(count_1 > count_0):
            submitTree.append(1)
            print("1")
        else:
            submitTree.append(0)
            print("0")

DecisionTree(100)
parseData(final)
startPrediction()
majority(100)

df = pd.DataFrame(submitTree, columns=['Survived'])
print(df)

#This is one example of what we got - for one decision tree:
#[[(3, (0, 30)), (5, (0, 2)), 1, '^', (5, (2, 6)), 1, '^', (3, (30, 60)), (4, (0, 2)), 1, '^', (4, (2, 8)), 1, '^', (3, (60, 120)), (6, (0, 200)), (2, 'male'), (1, 1), 0, '^', (1, 2), 0, '^', (1, 3), 0, '^', (2, 'female'), 1, '^', (6, (200, 1000)), 0, '^']]

#This is our finalParsed:
#[[[(6, (0, 200)), (1, 1), (3, (0, 30)), (2, 'male'), 0], [(2, 'female'), 1], [(3, (30, 60)), 1], [(3, (60, 120)), 0], [(1, 2), 0], [(1, 3), (4, (0, 2)), (5, (0, 2)), 0], [(5, (2, 6)), 0], [(4, (2, 8)), 0], [(6, (200, 1000)), 1]], [[(6, (0, 200)), (1, 1), (3, (0, 30)), (2, 'male'), 0], [(2, 'female'), 1], [(3, (30, 60)), 1], [(3, (60, 120)), 0], [(1, 2), 0], [(1, 3), (4, (0, 2)), (5, (0, 2)), 0], [(5, (2, 6)), 0], [(4, (2, 8)), 0], [(6, (200, 1000)), 1]]]

#split at "^"
