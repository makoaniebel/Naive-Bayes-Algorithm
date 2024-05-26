import sys
from collections import defaultdict

def load(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    attrs = defaultdict(list)
    data = []
    stop = True
    for i in lines:
        i = i.strip()
        if i.lower().startswith('@attribute'):
            parts = i.split()
            name = parts[1]
            vals = parts[2].replace('{', '').replace('}', '').split(',')
            vals = [val.strip() for val in vals if val.strip()]
            attrs[name] = vals
        elif not stop and i:
            data.append([item.strip() for item in i.split(',')])
        elif i.lower().startswith('@data'):
            stop = False
    return attrs, data

def predict(numFeature, data, numClass, attrs, numSamples):
    predicts = []
    for sample in data:
        probs = defaultdict(float)
        for label, count in numClass.items():
            prob = count / numSamples
            prob = getProb(sample, attrs, label, numFeature, prob)
            probs[label] = prob
        sumProb = sum(probs.values())
        newProbs = {label: prob / sumProb  for label, prob in probs.items()}
        predicts.append(newProbs)
    return predicts

def getProb(sample, attrs, label, numFeature, prob):
    for i, feature in enumerate(sample[:-1]):
        if i in attrs:
            featCount = numFeature[i][feature].get(label, 0)
            sumCount = sum(numFeature[i][val][label] for val in attrs[i])
            prob *= (featCount + 1) / (sumCount + len(attrs[i]))
    return prob

def train(data, numClass, numFeature):
    for sample in data:
        label = sample[-1]
        numClass[label] += 1
        for i, feature in enumerate(sample[:-1]):
            numFeature[i][feature][label] += 1
    return numClass, numFeature, len(data)

def output(outFile, predictions):
    with open(outFile, 'w') as f:
        f.write("predicted probability of being prescribed contact-lenses:\n")
        for i in predictions:
            for label, prob in i.items():
                f.write(f"{label}: {prob:.4f} |")
            f.write('\n')
def main():
    trainFile = sys.argv[1]
    inFile = sys.argv[2]
    outFile = sys.argv[3]
    attrs, trainData = load(trainFile)
    numClass = defaultdict(int)
    numFeature = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    numClass, numFeature, numSamples = train(trainData, numClass, numFeature)
    x, inputData = load(inFile)
    predictions = predict(numFeature, inputData, numClass, attrs, numSamples)
    output(outFile, predictions)

if __name__ == "__main__":
    main()
