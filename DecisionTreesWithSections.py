import pandas as pd
import numpy as np
import pprint as pp
import copy as cp
import matplotlib.pyplot as plt
from pandas_ml import ConfusionMatrix
from sklearn.metrics import confusion_matrix


dataset = pd.read_csv("letter-recognition.data",delimiter=',', header=None)
dataset.columns = ["lettr", "x-box", "y-box", "width-box", "height-box", "total pixels", "x-bar", "y-bar", "x2bar", "y2bar", "xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"]
dataset

#Calculate entropy of a target class
def entropy(target):
    element,count = np.unique(target,return_counts=True)
    entropy = np.sum([(-count[i]/np.sum(count))*np.log2(count[i]/np.sum(count)) for i in range(len(element))])
    return entropy
#Calculate information gain for the data with a specific attribute
def InfoGain(data, attribute):
    totalEntropy = entropy(data["lettr"])
    element,count=np.unique(data[attribute],return_counts=True)
    weightedEntropy = np.sum([(count[i]/np.sum(count))*entropy(data.where(data[attribute]==element[i]).dropna()["lettr"]) for i in range(len(element))])
    return totalEntropy - weightedEntropy
#split a dataframe into 2 smaller dataframes on a value of an attribute
def testSplit(value, dataset, attribute):
    smaller, bigger = [], []
    for index, row in dataset.iterrows():
        if row[attribute] < value:
            smaller.append(row)
        else:
            bigger.append(row)
    smaller = pd.DataFrame(smaller)
    bigger = pd.DataFrame(bigger)
    return smaller, bigger
#find the gini score of 2 data frames and calculate gini score for all target classes
def gini(dataGroups, targets):
    groupsNo = float(sum([len(g) for g in dataGroups]))
    giniO = 0
    for g in dataGroups:
        rows = len(g)
        if rows == 0:
            continue
        score = 0
        counts = g["lettr"].value_counts(normalize=True)
        for target in targets:
            try:
                a = counts[target]
                score += a * a
            except:
                continue
        giniO += (1-score) * (rows/groupsNo)
    return giniO
#Evaluate possible splits and return best split
def getSplit(dataset, minSize):
    attributef, value, score, groupsf = None, 999, 999, None
    for attribute in dataset.columns[1:]:
        for row in dataset[attribute].unique().tolist():
            groups = testSplit(row, dataset, attribute)
            giniO = gini(groups, dataset["lettr"].unique().tolist())
            if giniO < score:
                attributef, value, score, groupsf = attribute, row, giniO, groups
    return {'attribute':attributef, 'value':value, 'groups':groupsf}
#build tree
def buildTree(trainingData, maxDepth,minSize, depth = 1):
    root = getSplit(trainingData, minSize)
    tree = {root["feature"]:{}} #root node for iteration
    left, right = root['groups']
    right = right.reset_index(drop=True)
    left = left.reset_index(drop=True)
    if len(left) == 0 or len(right) == 0: #collapse if no split
        tree = np.unique(trainingData["lettr"])[np.argmax(np.unique(trainingData["lettr"],return_counts=True)[0])]
        return tree
    if depth >= maxDepth: #collapse if greater than max Depth
        tree[root["attribute"]]["L" + str(root["value"])] = np.unique(left["lettr"])[np.argmax(np.unique(left["lettr"],return_counts=True)[0])]
        tree[root["attribute"]]["B" + str(root["value"])] = np.unique(right["lettr"])[np.argmax(np.unique(right["lettr"],return_counts=True)[0])]
        return tree
    if len(left) <= minSize: #collapse leaf if data set is smaller than min size
        tree[root["attribute"]]["L" + str(root["value"])] = np.unique(left["lettr"])[np.argmax(np.unique(left["lettr"],return_counts=True)[0])]
    else: #add another branch
        tree[root["attribute"]]["L" + str(root["value"])] = buildTree(left, maxDepth, minSize, depth+1)
    if len(right) <= minSize: #collapse leaf if data set is smaller than min size
        tree[root["attribute"]]["B" + str(root["value"])] = np.unique(right["lettr"])[np.argmax(np.unique(right["lettr"],return_counts=True)[0])]
    else: #add another branch
        tree[root["attribute"]]["B" + str(root["value"])] = buildTree(right, maxDepth, minSize, depth+1)
    return tree
#predict by recursively iterating through the continuous tree
def predictC (query,tree):
    for key, value in query.items():
        if key in tree.keys():
            result = None
            for key2 in tree[key].keys():
                string = key2 #either B/L-number
                number = float(string[1:]) # number
                if string[0] == 'B':
                    if value >= number:
                        result = tree[key][key2]
                else:
                    if value < number:
                        result = tree[key][key2]
            if isinstance(result, dict):
                return predictC(query, result)
            else:
                return result
#test algorithm for continuous tree            
def testC (data,tree):
    queries = data.iloc[:,1:].to_dict(orient = "records")
    
    predicted = pd.DataFrame(columns=["lettr"]) 
    
    for i in range(len(data)):
        predicted.loc[i,"lettr"] = predictC(queries[i],tree)
    
    finalPercent = np.sum(predicted["lettr"]==data["lettr"])/len(data)*100
    print('accuracy = ',finalPercent,'%')
    return finalPercent
#test algorithm for continuous trees
def testsC (data,trees):
    queries = data.iloc[:,1:].to_dict(orient = "records")
    percent = []
    predictions = []
    predicted = pd.DataFrame(columns=["lettr"])
    finalPredicted = pd.DataFrame(columns=["lettr"])
    default = np.unique(data["lettr"])[np.argmax(np.unique(data["lettr"],return_counts=True)[0])]
    for k in range(len(trees)):
        for i in range(len(data)): #get all tree's predictions
            predicted.loc[i,"lettr"] = predictC(queries[i],trees[k])
        predictions.append(predicted.copy(deep=True))
        percent.append(np.sum(predicted["lettr"]==data["lettr"])/len(data))
        
    for i in range(len(data)): #vote for all predictions
        predicted.loc[i,"lettr"] = vote(predictions, percent, i)
    confusionMatrix = confusion_matrix(data["lettr"], predicted["lettr"])
    print(confusionMatrix)
    finalPercent = np.sum(predicted["lettr"]==data["lettr"])/len(data)*100
    print('accuracy = ',finalPercent,'%')
    return finalPercent, confusionMatrix
#Create a random forest of continuous trees
def randomForestC (data, splits, maxDepth, minSize):
    trees = []
    splitData = dataSplit(data,splits)
    for d in range(splits):
        trees.append(buildTree(splitData[d], maxDepth, minSize))
    return trees
#Generate a discrete decision tree using the ID3 algorithm
def ID3(data,originalData,features,parent=None):
    if len(np.unique(data["lettr"])) == 1: #leaf
        return np.unique(data["lettr"])[0]
    elif len(data)==0: #leaf
        return np.unique(originalData["lettr"])[np.argmax(np.unique(originalData["lettr"],return_counts=True)[0])] #was 1
    elif len(features) ==0: #leaf
        return parent
    else: #add another branch
        parent = np.unique(data["lettr"])[np.argmax(np.unique(data["lettr"],return_counts=True)[0])] #was 1
        
        items = [InfoGain(data,feature) for feature in features] #calculate info gain for all features
        bestFeature = features[np.argmax(items)]
        
        tree = {bestFeature:{}} #root node for branch
        
        features = [i for i in features if i != bestFeature] 
        
        for value in np.unique(data[bestFeature]): #add another branch for each feature except the best one
            subData = data.where(data[bestFeature] == value).dropna()
            subTree = ID3(subData,originalData,features,parent)
            tree[bestFeature][value] = subTree
            
        return (tree)
#predict using a discrete tree
def predict (query,tree):
    default = modeOfBranch(tree)
    for key in query.keys():
        if key in tree.keys():  
            try: 
                result = tree[key][query[key]]
            except:
                return default
            result = tree[key][query[key]]
            if isinstance(result,dict):
                return predict(query,result)
            else:
                return result
#split training data for use in training random forests
def dataSplit(dataset, splitsNo): 
    splits = []
    for i in range(splitsNo):
        fraction = 1/(splitsNo-i)
        data = dataset.sample(frac=fraction)
        dataset = dataset.drop(data.index)
        data = data.reset_index(drop=True)
        dataset = dataset.reset_index(drop=True)
        splits.append(data)
    return splits
#initial data split into training and testing
def trainTestSplit(dataset):
    training_data = dataset.sample(frac=0.8)
    testing_data = dataset.drop(training_data.index)
    training_data = training_data.reset_index(drop=True)
    testing_data = testing_data.reset_index(drop=True)
    return training_data,testing_data

temp = trainTestSplit(dataset)
trainingData = temp[0]
testingData = temp[1]
#return unique values in an array
def unique (l):
    u = []
    for x in l:
        if x not in u:
            u.append(x)
    return u
#test a discrete tree's accuracy with testing data
def test(data,tree):
    queries = data.iloc[:,1:].to_dict(orient = "records")
    
    predicted = pd.DataFrame(columns=["lettr"]) 
    
    for i in range(len(data)):
        predicted.loc[i,"lettr"] = predict(queries[i],tree) 
    
    finalPercent = np.sum(predicted["lettr"]==data["lettr"])/len(data)*100
    print('accuracy = ',finalPercent,'%')
    return finalPercent
#Make tree's vote using a set of predictions and their percent accuracies
def vote(predictions, percents, index):
    predList = []
    predWeights = []
    for i in range(len(percents)): 
        predList.append(predictions[i].loc[index,"lettr"])
        predWeights.append(percents[i])
    uniqueList = unique(predList)
    uniqueCounts = []
    uniqueWeights = [0] * len(uniqueList)
    for y in predList: #add up all the accuracies
        if y in uniqueList:
            uniqueWeights[uniqueList.index(y)] += predWeights[predList.index(y)]
    
    for x in uniqueList: #average accuracy
        uniqueCounts.append(predList.count(x))
        uniqueWeights[uniqueList.index(x)] /= uniqueCounts[uniqueList.index(x)]
        
    maxCount = 1
    letter = ''
    for f in uniqueList: #find the mode
        if uniqueCounts[uniqueList.index(f)] > maxCount:
            maxCount = uniqueCounts[uniqueList.index(f)]
            letter = f
        elif uniqueCounts[uniqueList.index(f)] == maxCount:
            letter = ''
    if letter != '':
        return letter
    
    maximum = 0
    for z in uniqueWeights: #return higher accuracy of the 2 voter parties if equal parts
        if z > maximum:
            maximum = z
    return uniqueList[uniqueWeights.index(z)]

confusionMatrix = None
#test discrete random forest
def tests(data,trees):
    queries = data.iloc[:,1:].to_dict(orient = "records")
    percent = []
    predictions = []
    predicted = pd.DataFrame(columns=["lettr"])
    finalPredicted = pd.DataFrame(columns=["lettr"])
    for k in range(len(trees)):
        for i in range(len(data)):
            predicted.loc[i,"lettr"] = predict(queries[i],trees[k])
        predictions.append(predicted.copy(deep=True))
        percent.append(np.sum(predicted["lettr"]==data["lettr"])/len(data))
        
    for i in range(len(data)):
        predicted.loc[i,"lettr"] = vote(predictions, percent, i)
        
    confusionMatrix = confusion_matrix(data["lettr"], predicted["lettr"])
    print(confusionMatrix)
    finalPercent = np.sum(predicted["lettr"]==data["lettr"])/len(data)*100
    print('accuracy = ',finalPercent,'%')
    return finalPercent, confusionMatrix
#return the mode of a branch by recursively collapsing the leaf nodes
def modeOfBranch(branch,b=False):
    track = []
    if b:
        pp.pprint(branch)
    if isinstance(branch,dict):
        for value in branch.values():
            if isinstance(value,str) and len(value) == 1:
                track.append(value)
            elif isinstance(value,dict):
                track.append(modeOfBranch(value))
    if track == []:
        return branch
    return max(set(track), key=track.count)            
#create a discrete random forest
def randomForest(data, splits):
    trees = []
    splitData = dataSplit(data,splits)
    for d in range(splits):
        trees.append(ID3(splitData[d],splitData[d],data.columns[1:]))
    return trees

#all code following was used to test the data set and attempt to find the classifier's highest accuracy on the dataset
plotDiscrete = {}
plotDiscrete["splits"] = [1,2,5,10,20,30]
accuracies = []
confusionMatricesD = []
for d in plotDiscrete["splits"]:
    trees = randomForest(trainingData, d)
    testResults = tests(testingData, trees)
    accuracies.append(testResults[0])
    confusionMatricesD.append(testResults[1])
    print(d)
plotDiscrete["accuracy"] = accuracies
discreteDataFrame = pd.DataFrame(plotDiscrete)
discreteDataFrame.plot(kind='bar', x='splits', y='accuracy', ylim=100)
plt.savefig("Discrete.png")
plt.show()

features = ["splits", "maxDepth", "minSize", "accuracy"]
splits = [1,5,10,20]
maxDepths = [10,20,30]
minSize = [5,10,20]
results = []
treesC = []
confusionMatricesC = []
for s in splits:
    for d in maxDepths:
        for m in minSize:
            trees = randomForestC(trainingData, s, d, m)
            treesC.append(trees)
            testResults = testsC(testingData, trees)
            plot = {}
            plot[features[0]] = s
            plot[features[1]] = d
            plot[features[2]] = m
            plot[features[3]] = testResults[0]
            confusionMatricesC.append(testResults[1])
            results.append(plot)
            print(s,d,m, testResults[0])
continuousDataFrame = pd.DataFrame(results)
print(continuousDataFrame)

print(discreteDataFrame)

depths2 = [40,50]
size2 = [1,5]
for s in splits:
    for d in depths2:
        for m in size2:
            trees = randomForestC(trainingData, s, d, m)
            treesC.append(trees)
            testResults = testsC(testingData, trees)
            plot = {}
            plot[features[0]] = s
            plot[features[1]] = d
            plot[features[2]] = m
            plot[features[3]] = testResults[0]
            confusionMatricesC.append(testResults[1])
            results.append(plot)
            print(s,d,m, testResults[0])
df2 = pd.DataFrame(results)
print(df2)

depths3 = [25,30,35,45]
for s in splits:
    for d in depths3:
        trees = randomForestC(trainingData, s, d, 1)
        treesC.append(trees)
        testResults = testsC(testingData, trees)
        plot = {}
        plot[features[0]] = s
        plot[features[1]] = d
        plot[features[2]] = m
        plot[features[3]] = testResults[0]
        confusionMatricesC.append(testResults[1])
        results.append(plot)
        print(s,d,m, testResults[0])
df3 = pd.DataFrame(results)
print(df3)

print(results[52])

for i in range(16):
    results[52+i][features[2]] = 1

df3 = pd.DataFrame(results)

print(df3)


from pandas.plotting import table

ax = plt.subplot(111, frame_on=False) # no visible frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis
table(ax, df3)
plt.savefig('continuousTable.png')

def plotConfusionMatrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=False):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(12, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]).replace(".0", ""),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(title +".png", bbox_inches='tight')
    plt.show()

conf = testsC(testingData,treesC[53])
plotConfusionMatrix(conf[1], np.unique(dataset["lettr"]), 'continuousConfusionMatrix')

testResults = tests(testingData, trees)
plotConfusionMatrix(testResults[1], np.unique(dataset["lettr"]))

df3.to_html('tableC.html')

print(df3.columns)
df3 = df3.sort_values('minSize', ascending=True)
df4 = df3[:24].reset_index(drop=True).drop('minSize', axis=1)
ax = plt.gca()
df4 = df4.sort_values('accuracy', ascending=False)
df4 = df4[:6].sort_values('maxDepth')
print(df4)
df4.plot(kind='bar', x = 'maxDepth', y = 'accuracy', ax=ax)
plt.show()

df3[:30].reset_index(drop=True).to_html('tableC.html')

def label_by_id(v, label, ID):
    num = v.get_label_by_id(ID).get_text()
    v.get_label_by_id(ID).set_text(label + "\n\n" + num)

import sys
get_ipython().system('{sys.executable} -m pip install matplotlib_venn')
def plotVenDiagram(Ab, aB, AB, ab, letter):

    # Ab  A and not B
    # aB - not A and B
    # AB - A and B
    # ab - neither A or B 

    plt.figure(figsize=(5, 5))
    v = venn2(subsets=(Ab, aB, AB), set_labels=('System predicts letter ' + letter, 'Actually letter ' + letter))

    label_by_id(v, 'False positives', '01')
    label_by_id(v, 'False negatives', '10')
    label_by_id(v, 'True positives', '11')

    v.get_patch_by_id('01').set_color('green')
    v.get_patch_by_id('10').set_color('blue')


    plt.figtext(0.1, 0.85, 'True Negatives')
    plt.figtext(0.1, 0.8, str(ab))
    plt.savefig(letter + '_venn diagram.png')
    plt.show()

from matplotlib_venn import venn2
TP = conf[1][7][7]
print(TP)
FP = 0
for i in range(26):
    if i != 7:
        FP = FP + conf[1][i][7]
print(FP)
FN = 0
for i in range(26):
    if i != 7:
        FN = FN + conf[1][7][i]
print(FN)
TN = 4000 - FN - FP - TP
plotVenDiagram(FN, FP, TP, TN, 'H')

TP = conf[1][6][6]
print(TP)
FP = 0
for i in range(26):
    if i != 6:
        FP = FP + conf[1][i][6]
print(FP)
FN = 0
for i in range(26):
    if i != 6:
        FN = FN + conf[1][6][i]
print(FN)
TN = 4000 - FN - FP - TP
plotVenDiagram(FN, FP, TP, TN, 'G')

TP = conf[1][15][15]
print(TP)
FP = 0
for i in range(26):
    if i != 15:
        FP = FP + conf[1][i][15]
print(FP)
FN = 0
for i in range(26):
    if i != 15:
        FN = FN + conf[1][15][i]
print(FN)
TN = 4000 - FN - FP - TP
plotVenDiagram(FN, FP, TP, TN, 'O')

TP = testResults[1][7][7]
print(TP)
FP = 0
for i in range(26):
    if i != 7:
        FP = FP + testResults[1][i][7]
print(FP)
FN = 0
for i in range(26):
    if i != 7:
        FN = FN + testResults[1][7][i]
print(FN)
TN = 4000 - FN - FP - TP
plotVenDiagram(FN, FP, TP, TN, 'H')

TP = testResults[1][6][6]
print(TP)
FP = 0
for i in range(26):
    if i != 6:
        FP = FP + testResults[1][i][6]
print(FP)
FN = 0
for i in range(26):
    if i != 7:
        FN = FN + testResults[1][6][i]
print(FN)
TN = 4000 - FN - FP - TP
plotVenDiagram(FN, FP, TP, TN, 'G')

discreteDataFrame.to_html('ds.html')





