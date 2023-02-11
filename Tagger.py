import csv
import time

# Define a list of panctuations
panctuations = [',', '"', '.', '(', ')', ':', ';', '?', '[', ']', '@', '!', '%']

def processTrainData(filename) -> tuple :
    print("Training...............")
    data = {} # Store all data as a dictionary of work and token
    freq = {} # Frequencies per word per tag
    prefix_freq = {} # Frequency of prefix(2) of a word
    totalTagFequency = {} # Frequency of each tag
    taglist=[] # Store a list of all tags
    testDictionaryTag={} # Dictionary of test data tags 

    with open(filename) as trainFile:
        csv_reader = csv.reader(trainFile) # Create an object of the csv reader
        next(csv_reader) # Skip the Header of the CSV file
        for row in csv_reader:
            word = row[0].lower()
            # Skip word if it is a panctuation
            if word not in panctuations:
                data[word] = row[1]

                # Mark the end of a paragraph
                if(row[1] == ""):
                    testDictionaryTag["<E>"] = 0
                else:
                    testDictionaryTag[row[1]] = 0
                taglist.append(row[1])

                if row[1] not in totalTagFequency:
                    totalTagFequency[row[1]] = 1
                else:
                    totalTagFequency[row[1]] += 1
                
                # Frequency of each tag per word
                # Stored as dictionaries of tags per word dictionary 
                if word not in freq:
                    freq[word] = {row[1] : 1}
                else:
                    if row[1] in freq[word]:
                        freq[word][row[1]] += 1
                    else:
                        freq[word][row[1]] = 1
                
                # Prefix table
                if word[0:2] not in prefix_freq:
                    prefix_freq[word[0:2]] = {row[1] : 1}
                else:
                    if row[1] in prefix_freq[word[0:2]]:
                        prefix_freq[word[0:2]][row[1]] += 1
                    else:
                        prefix_freq[word[0:2]][row[1]] = 1

    # Return a tuple of useful variables
    return prefix_freq, freq, totalTagFequency, taglist, testDictionaryTag

def emissionTable(prefix_freq, freq, testDictionaryTag, totalTagFequency) -> tuple : 
    # Emission table of probabilities
    for word in freq:
        for tag in testDictionaryTag:
            if tag not in freq[word]:
                freq[word][tag] = 0
            else:
                temp_freq = freq[word][tag] / totalTagFequency[tag]
                freq[word][tag] = temp_freq

    # Prefix words table of probabilities
    for word in prefix_freq:
        for tag in testDictionaryTag:
            if tag not in prefix_freq[word]:
                prefix_freq[word][tag] = 0
            else:
                temp_freq = prefix_freq[word][tag] / totalTagFequency[tag]
                prefix_freq[word][tag] = temp_freq

    return prefix_freq, freq

# Transition Table
def transitionTable(taglist, totalTagFequency) -> tuple:
    # Transition properties for each tag
    tTable={}
    temp = "<S>" 
    for tag in taglist:
        # Check end of a sentence and assign it <E>
        if tag == "": 
            tag="<E>"

        if temp not in tTable:
            tTable[temp]=dict(testDictionaryTag) 
            tTable[temp][tag]= 1
        else:
            if tag not in tTable[temp]:
                tTable[temp][tag]=1
            else:
                tTable[temp][tag]+=1

        # Cater for last word of last sentence not to affect the new sentence
        if tag=="<E>":       
            temp="<S>"
        else:
            temp=tag
    

    # Transition table probabilities        
    for row in tTable:
        rowCount=0
        totalSymbol = 0
        for col in tTable[row]:
            # Calculate row total
            rowCount += tTable[row][col] 
            totalSymbol += 1
        for col in tTable[row]:
            #Divide each column in row by row total and use laplace (Add 1) smoothing
            # tTable[row][col] = (tTable[row][col]+1) / (rowCount+len(testDictionaryTag))  
            #Divide each column in row by row total and use k-smoothing
            tTable[row][col] = (tTable[row][col] + 0.5) / (rowCount + 0.5 * len(testDictionaryTag))

    # Move start tag from transition table
    sTable = tTable["<S>"] 
    # Remove start tag from transition table
    tTable.pop("<S>") 

    # Prepare tags
    tags = []
    for i in totalTagFequency:
        if i != '':
            tags.append(i)


    return  sTable, tTable, tags


#Viterbi algorithm for decoding and handling of unknowns
def viterbi(observation, tags, sTrans, transitions, emissionTableFreq, prefixTableFreq, totalTagFequency, test) -> tuple:
    #Calculate the probability of the most recuring tag
    highestProbabilityTag= totalTagFequency[max(totalTagFequency, key=totalTagFequency.get)] / len(totalTagFequency)
    average=0
    sCount=0
    vPaths = [{}]

    for tag in tags:
        # Condition if the word is known
        if observation[0] in emissionTableFreq:
            probability = sTrans[tag]                                     
            previous=None
            pTagForWord = emissionTableFreq[observation[0]][tag]
            vPaths[0][tag] = {"prev": previous,"prob": (probability * pTagForWord)}   
        
        # If the word is unknown
        else:
            # Use prefix table probabilities 
            if observation[0][0:2] in prefixTableFreq:
                probability = sTrans[tag]
                previous=None
                pPrefixTagForWord = prefixTableFreq[observation[0][0:2]][tag]
                vPaths[0][tag] = {"prev": previous, "prob": (probability * pPrefixTagForWord)}
            else:
                # Or use use the highest probability tag
                probability = sTrans[tag]
                previous=None
                vPaths[0][tag] = {"prev": previous, "prob": (probability * highestProbabilityTag)}    

    #loop through the words in the sentence
    for obsWord in range(1, len(observation)):
        vPaths.append({})
        #Loop through each possible tag
        for tag in tags:        
            path=vPaths[obsWord - 1][tags[0]]["prob"]
            trans=transitions[tags[0]][tag]
            maxProbability = (path * trans)
            previousTagS = tags[0]

            # Check for transition probabilities
            for previousTag in tags[1:]:       
                pVPaths=(vPaths[obsWord - 1][previousTag]["prob"])
                prevTrans=transitions[previousTag][tag]
                trainingProbability = pVPaths * prevTrans

                if  maxProbability < trainingProbability:
                    maxProbability = trainingProbability
                    previousTagS = previousTag

            # If the word is in training set
            if observation[obsWord] in emissionTableFreq:                                                  
                pTagForWord = emissionTableFreq[observation[obsWord]][tag]
                maximumProb = (maxProbability * pTagForWord)
            # If the word is unknown
            else: 
                # Check if word prefix is in prefix table
                if observation[obsWord][0:2] in prefixTableFreq:   
                    pPrefixTagForWord = prefixTableFreq[observation[obsWord][0:2]][tag]
                    maximumProb = (maxProbability * pPrefixTagForWord)               
                # Treat the word as entirely unknown
                else:
                    maximumProb = maxProbability * highestProbabilityTag

            vPaths[obsWord][tag] = {"prev": previousTagS,"prob": maximumProb}

    closestTag = None
    outputTags = []
    maximumProb = 0.0

    #Probabilities Backtrack
    for (tag, tagProb) in vPaths[-1].items():
        if  maximumProb < tagProb["prob"]:
            maximumProb = tagProb["prob"]
            closestTag = tag

    outputTags.append(closestTag)
    previous = closestTag

    # Backtracking
    vPathLen=len(vPaths) - 2
    for tagProb in range(vPathLen, -1, -1):
        outputTags.insert(0, vPaths[tagProb + 1][previous]["prev"])
        previous = vPaths[tagProb + 1][previous]["prev"]
    
    #Calculating average accurate for sentences
    count = 0
    total = 0
    for i in range(len(outputTags)-1):
        if (outputTags[i] == test[i]):
            count +=1
        total +=1

    if total != 0:
        average +=(count/total)*100
    sCount+=1

    return outputTags, average, sCount

#Running Development set 
def runDevTests(tags, sTable, tTable, emissionTableFreq, prefixTableFreq, totalTagFequency) :
    print("Testing................")
    test = []
    with open('Dataset/TrainData.csv') as devfile:
        csv_reader = csv.reader(devfile)
        next(csv_reader)
        observation=[]
        start = time.time()
        for row in csv_reader:
            word = row[0].lower()
            if word not in panctuations:
                if word != "":
                    observation.append(word)
                    test.append(row[1])
                else:
                    # Call viterbi algorithm to tag every word in the corpus
                    outputTags, average, sCount = viterbi(observation,tags,sTable,tTable,emissionTableFreq, prefixTableFreq, totalTagFequency, test)
                    observation=[]
                    test=[]
        end = time.time()
        totaltime = (end-start)
    print(f"Total time(seconds) taken for testing: {totaltime}")            
    print(f"Development Set Accuracy: {round(average/sCount, 2)}%")


def runTestSet(tags, sTable, tTable, emissionTableFreq, prefixTableFreq, totalTagFequency) : 
    print("Testing................")
    test = []
    file = open("TestOutput.txt","w")
    file.close()
    with open('Dataset/TestData.csv') as testfile:
        
        csv_reader = csv.reader(testfile)
        next(csv_reader)
        sentence = []
        for row in csv_reader:
            word = row[0].lower()
            if word not in [',', '"', '.', '(', ')', ':', ';', '?', '[', ']', '@', '!', '%']:
                if word != "":
                    sentence.append(word)
                    test.append(row[1])
                else:
                    start = time.time()
                    outputTags, average, sCount = viterbi(sentence,tags,sTable,tTable,emissionTableFreq, prefixTableFreq, totalTagFequency, test)
                    end = time.time()
                    totaltime = (end-start)

                    file_object = open('TestOutput.txt', 'a')
                    for i in range(0,len(sentence)):
                        file_object.write("Word: "+sentence[i]+", POS: "+outputTags[i]+" Original_POS: "+test[i]+"\n")
                    file_object.close()

                    sentence=[]
                    test=[]
                    file_object.close()

    print("\nThe tagged words are written to TestOutput.tx \n")
    print(f"\nTotal time(seconds) taken for testing: {totaltime}")                
    print(f"Test Set Accuracy: {round(average/sCount, 2)}%")


# Main Method
if __name__ == "__main__":
    # Load Train dataset
    trainingFile = 'Dataset/TrainData.csv'
    # Process training dataset and do some computations
    prefix_freq, freq, totalTagFequency, taglist, testDictionaryTag = processTrainData(trainingFile)
    
    # Compute emission table
    prefixTableFreq, emissionTableFreq = emissionTable(prefix_freq, freq, testDictionaryTag, totalTagFequency)
    
    # Compute Transition table
    sTable, tTable, tags = transitionTable(taglist, totalTagFequency)

    # Print development tests
    # runDevTests(tags, sTable, tTable, emissionTableFreq, prefixTableFreq, totalTagFequency)

    # Print testing tests
    runTestSet(tags, sTable, tTable, emissionTableFreq, prefixTableFreq, totalTagFequency)