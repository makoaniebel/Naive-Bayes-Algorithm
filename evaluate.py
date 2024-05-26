import os 
import sys 
from naivebayes import load
from collections import defaultdict 

def generate(train, test, trainF, testF): 
    with open(trainF, 'w') as f: 
        f.write('\n'.join([','.join(sample) for sample in train])) 
    with open(testF, 'w') as f: 
        f.write('\n'.join([','.join(sample) for sample in test])) 

def evaluate(dataFile): 
    x, data = load(dataFile) 
    matrix = defaultdict(lambda: defaultdict(int)) 
    predicts = 0 
    for i in range(len(data)): 
        test = data[i] 
        train = data[:i] + data[i+1:]
        generate(train, [test], 'training.txt', 'test.txt') 
        os.system('python3 naivebayes.py training.txt test.txt result.txt') 
        with open('result.txt', 'r') as f: 
            predictClass = f.read().strip() 
            realClass = test[-1] 
        matrix[realClass][ predictClass] += 1 
    print("confusion matrix:") 
    strMatrix = [f'{key} : {matrix[key]}' for key in matrix]
    with open('result.txt', 'w') as f:
        f.write("confusion matrix:\n")
        [f.write(f'{st}\n') for st in strMatrix]
    for actual, preds in matrix.items():   
        print(actual, preds) 
    print("overall accuracy:", getAccuracy(matrix, predicts, data)) 

def getAccuracy(matrix, predicts, data):
    for key, val in matrix.items():
        if key == 'none':
            noneNum = val['predicted probability of being prescribed contact-lenses:']
            predicts += noneNum
        elif key == 'soft': 
            softNum = val['predicted probability of being prescribed contact-lenses:']
            predicts += softNum
        elif key == 'hard':
            hardNum = val['predicted probability of being prescribed contact-lenses:']
            predicts += hardNum
    return predicts / len(data) 

def main(): 
    dataFile = sys.argv[1] 
    evaluate(dataFile) 
if __name__ == "__main__": 
    main()