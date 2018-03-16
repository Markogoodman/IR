from porter import PorterStemmer
import re
import os
import math

class Document:
    def __init__(self, docID, filePath):
        '''
        __documentRecord = {
            'term': {
                'index': int,
                'tf': int # term frequency
            }

        }
        '''
        self.__documentRecord = {}
        self.docID = docID
        self.filePath = filePath
        # init
        tokens = self.getTokens(self.filePath)
        tokens = self.removeStopWords(tokens, 'stop-word-list.txt')
        stemmedTokens = self.stem(tokens)
        for term in stemmedTokens:
            # +1 if term exists else set to 1
            record = self.__documentRecord.get(term, None)
            if record:
                record['tf'] += 1 
            else:
                self.__documentRecord[term] = {
                    'index': 0,
                    'tf': 1
                }

    def getTokens(self, path):
        """
        receive file path
        return a lower case token list
        """
        tokens = []
        with open(path, 'r') as f:
            tokens = list(filter(None, re.split(r'\W', f.read())))
        tokens = [token.lower() for token in tokens if len(token) > 1 and token.isalpha()]

        return tokens

    def stem(slef, tokens):
        """
        receive tokens
        return stemmedTokens
        """
        stemmedTokens = []
        stemmer = PorterStemmer()
        for token in tokens:
            '''
            if stemmer.stem(token, 0, len(token)-1) == 'el':
                print(token)
            '''
            stemmedTokens.append(stemmer.stem(token, 0, len(token)-1))

        return stemmedTokens

    def removeStopWords(self, tokens, path):
        """
        give tokens and stop-word-list filepath
        return tokens without stop words
        """
        stopWords = []
        with open(path, 'r') as f:
            stopWords = set(f.read().splitlines())
        tokens = [token for token in tokens if token not in stopWords]

        return tokens

    def getDocumentRecord(self):
        return self.__documentRecord
    '''
    def getUnitVector(self):
        return self.__unitVector

    def updateIndex(self,dictionary):
        dictionaryRecord = dictionary.getDictionaryRecord()
        for term, record in self.__documentRecord.items():
            record['index'] = dictionaryRecord[term]['index']
    '''



class Dictionary:
    def __init__(self, documents):
        '''
        __dictionaryRecord = {
            'term': {
                'index': int,
                'df': int,  #documentFrequency
                'idf': 0    #inverse document Frequency

                'tf': int # hw3 add
            }
        }
        '''
        self.__dictionaryRecord = {}
        self.__documentList = []
        self.totalLength = 0
        self.documentNum = len(documents)
        #count document frequency
        for docID, document in documents.items():
            #set doc id list
            self.__documentList.append(docID)
            documentRecord = document.getDocumentRecord()
            if documentRecord is not None:
                for term, term_info in documentRecord.items():
                    record = self.__dictionaryRecord.get(term)
                    self.totalLength += term_info['tf']
                    if record:
                        record['df'] += 1
                        record['tf'] += term_info['tf']
                    else:
                        self.__dictionaryRecord[term] = {
                            'index': 0,
                            'df': 1,
                            'tf': term_info['tf'],
                        }
        #After getting every term to dict, assign each term an index
        for i, term in enumerate(sorted(self.__dictionaryRecord)):
            self.__dictionaryRecord[term]['index'] = i + 1

    def getDictionaryRecord(self):
        return self.__dictionaryRecord

    def getDocumentList(self):
        return self.__documentList

    def updateFeature(self, features):
        #delete terms that are not in features
        #upadte __dictionaryRecord and totalLength
        newDictionaryRecord = {}
        newTotalLength = 0
        for feature in features:
            tmp = self.__dictionaryRecord.get(feature, {'index': 0, 'df': 0, 'tf': 0})
            newDictionaryRecord[feature] = tmp
            newTotalLength += tmp['tf']

        self.__dictionaryRecord = newDictionaryRecord
        self.totalLength = newTotalLength

class NBClassifier:
    def __init__(self, summary_dict, cls_dicts):
        self.prior = {} # P(c) { cls_num:　p}
        self.term_prob = {} # P(t|c) { cls_num: {term: p}}
        self.summary_dict = summary_dict
        self.cls_dicts = cls_dicts
        self.class_num = len(cls_dicts)

        self.calPrior()
        self.calTermProb()

    def calPrior(self):
        for cls, cls_info in self.cls_dicts.items():
            self.prior[cls] = cls_info.documentNum / self.summary_dict.documentNum

    def calTermProb(self):
        summary_record = self.summary_dict.getDictionaryRecord()
        V = len(summary_record) #training summary vocabulary size
        for cls in range(1, self.class_num+1): #1 to class_count
            self.term_prob[cls] = {}
            cls_dict = self.cls_dicts[cls]
            cls_record = cls_dict.getDictionaryRecord()
            for term, term_info in summary_record.items():
                #cal term's prob in this class
                self.term_prob[cls][term] = (cls_record[term]['tf'] + 1) / (cls_dict.totalLength + V)

    def predict(self, document):
        record = document.getDocumentRecord()
        class_score = {}
        for cls in range(1, self.class_num+1):
            #prior prob
            class_score[cls] = math.log(self.prior[cls], 10)
            score = 0
            for term, term_info in record.items():
                if term in self.term_prob[cls]:
                    #add score if term is in feature
                    log_prob = math.log(self.term_prob[cls][term], 10)
                    for i in range(0, term_info['tf']):
                        score += log_prob
            class_score[cls] += score

        candidate_class = 1
        candidate_score = class_score[1]
        for cls in range(2, self.class_num+1):
            if class_score[cls] > candidate_score:
                candidate_class = cls
                candidate_score = class_score[cls]

        return candidate_class

def featureSelection(training_summary, training_class_dictionary, class_count, feature_num):
    feature_score = {
        #term: float
    }
    N = training_summary.documentNum
    cls_summary_record = training_summary.getDictionaryRecord()
    for cls in range(1, class_count + 1):
        cls_dict = training_class_dictionary[cls]
        cls_record = cls_dict.getDictionaryRecord()
        cls_doc_count = cls_dict.documentNum
        for term, term_info in cls_record.items():
            n11 = term_info['df']
            n01 = cls_summary_record[term]['df'] - n11
            n10 = cls_doc_count - n11
            n00 = N - n11 - n10 - n01
            #
            pt = (n11 + n01) / N
            p1 = n11 / (n11 + n10)
            p2 = n01 / (n01 + n00)
            #
            H1 = pt ** n11 * (1 - pt) ** n10 * pt ** n01 * (1 - pt) ** n00
            H2 = p1 ** n11 * (1 - p1) ** n10 * p2 ** n01 * (1 - p2) ** n00
            LLR = -2 * math.log(H1 / H2, 2)
            feature_score[term] = feature_score.get(term, 0) + LLR / class_count
    feature_rank = sorted(feature_score.items(), key=lambda x: x[1])
    features = [feature for feature, score in feature_rank[-feature_num:]]

    return features

def main():
    #store all training doc object
    training_documents = {}
    #store { class_id(int):[docId, ..] }
    training_class_list = {}
    # {all training doc id}
    training_set = set() # To record training doc nums
    # store each class's summary info {class_id(int): Dictionary}
    training_class_dictionary = {}
    class_count = 0
    localPath = os.getcwd()
    corpusPath = os.path.join(localPath, 'IRTM')
    
    #read training info
    with open('training.txt', 'r') as f:
        for line in f.readlines():
            class_count += 1
            line = line.split()
            cls = int(line[0]) # first num in each line is class #
            training_class_list[cls] = []
            for docID in line[1:]:
                training_class_list[cls].append(int(docID))
                training_set.add(int(docID))

    #read training docs
    for cls, doc_list in training_class_list.items():
        cls_documents = {}
        for docID in doc_list:
            fileName = str(docID) + '.txt'
            filePath = os.path.join(corpusPath, fileName)
            #read for training summary dictionary
            training_documents[docID] = Document(docID, filePath)
            # read for each class dictionary
            cls_documents[docID] = Document(docID, filePath)

        training_class_dictionary[cls] = Dictionary(cls_documents)

    #store total training doc summary
    training_summary = Dictionary(training_documents)

    #Feature Selection
    features = featureSelection(training_summary, training_class_dictionary, class_count, 500)
    #update features, delete terms that are not in features
    training_summary.updateFeature(features)
    for i in training_class_dictionary:
        training_class_dictionary[i].updateFeature(features)

    #train a classfier
    classifier = NBClassifier(training_summary, training_class_dictionary)
    result = {}
    #walk through all doc
    for subdir, dirs, files in os.walk(corpusPath):
        for file in files:
            docID = file.split('.')
            docID = int(docID[0])
            if docID not in training_set:
                doc = Document(docID, os.path.join(corpusPath, file))
                result[docID] = classifier.predict(doc)

    with open('R06725038.txt', 'w') as f:
        for docID, cls in sorted(result.items()):
            f.write('%d \t %d \n' % (docID, cls))

if __name__ == '__main__':
    main()