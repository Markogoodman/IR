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
        '''
        __unitVector = {
            'term': {
                'index': int,
                'tf_idf': float
            }

        }
        '''
        self.__unitVector = {}

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

    def getUnitVector(self):
        return self.__unitVector

    def updateIndex(self,dictionary):
        dictionaryRecord = dictionary.getDictionaryRecord()
        for term, record in self.__documentRecord.items():
            record['index'] = dictionaryRecord[term]['index']

    def updateUnitVector(self, dictionary):
        #Do this after update documents'index
        vectorLength = 0
        dictionaryRecord = dictionary.getDictionaryRecord()
        for term, record in self.__documentRecord.items():
            self.__unitVector[term] = {
                'index': record['index'],
                'tf_idf': record['tf'] * dictionaryRecord[term]['idf']
            }
            vectorLength += self.__unitVector[term]['tf_idf'] ** 2
        vectorLength =  math.sqrt(vectorLength)
        #normalize to unit
        for term, record in self.__unitVector.items():
            record['tf_idf'] /= vectorLength


class Dictionary:
    def __init__(self, documents):
        '''
        __documentRecord = {
            'term': {
                'index': int,
                'df': int,  #documentFrequency
                'idf': 0    #inverse document Frequency
            }
        }
        '''
        self.__dictionaryRecord = {}
        self.documentNum = len(documents)
        #count document frequency
        for docID, document in documents.items():
            documentRecord = document.getDocumentRecord()
            if documentRecord is not None:
                for term in documentRecord:
                    record = self.__dictionaryRecord.get(term)
                    if record:
                        record['df'] += 1
                    else:
                        self.__dictionaryRecord[term] = {
                            'index': 0,
                            'df': 1,
                            'idf': 0
                        }
        #After getting every term to dict, assign each term an index
        for i, term in enumerate(sorted(self.__dictionaryRecord)):
            self.__dictionaryRecord[term]['index'] = i + 1

        self.calculateIDF()

    def getDictionaryRecord(self):
        return self.__dictionaryRecord

    def calculateIDF(self):
        for term, record in self.__dictionaryRecord.items():
            record['idf'] = math.log(self.documentNum / record['df'], 10)

def cosine(doc_x, doc_y):
    sim = 0
    with open(doc_x, 'r') as f1, open(doc_y, 'r') as f2:
        '''
        doc = {
            index(int) : tf_idf
        }
        '''
        doc1 = {}
        doc2 = {}
        for pair in f1.read().splitlines()[1:]:
            pair = pair.split()
            index = int(pair[0])
            tf_idf = float(pair[1])
            doc1[index] = tf_idf
        
        for pair in f2.read().splitlines()[1:]:
            pair = pair.split()
            index = int(pair[0])
            tf_idf = float(pair[1])
            doc2[index] = tf_idf
        
        for index, tf_idf in doc1.items():
            sim += tf_idf * doc2.get(index, 0)

    return sim


def main():
    documents = {}
    localPath = os.getcwd()
    corpusPath = os.path.join(localPath, 'IRTM')

    for subdir, dirs, files in os.walk(corpusPath):
        for file in files:
            docId = file.split('.')
            docId = int(docId[0])
            documents[docId] = Document(docId, os.path.join(corpusPath, file))

    #open a result dir
    result = os.path.join(localPath, 'result')
    if not os.path.exists(result):
        os.mkdir(result)   

    dictionary = Dictionary(documents)
    with open('dictionary.txt', 'w') as f:
        #tf = doc.getTermFrequency()
        dictionaryRecord = dictionary.getDictionaryRecord()
        for term, info in sorted(dictionaryRecord.items()):
            #f.write(str(info['index']) + '\t\t' + term + '\t\t' + str(info['df']) + '\n')
            f.write('%s \t %25s \t %5s \n' % (str(info['index']), term, str(info['df'])))

    for docId, document in sorted(documents.items()):
        document.updateIndex(dictionary)
        document.updateUnitVector(dictionary)
        with open(result + '/' + str(docId) + ".txt", 'w') as f:
            unitVector = document.getUnitVector()
            f.write(str(len(unitVector)) + '\n')
            for term in sorted(unitVector):
                #f.write(term + '\t\t' + str(unitVector[term]['tf_idf']) + '\n')
                f.write('%5s \t %4.3f \n' % (str(unitVector[term]['index']), unitVector[term]['tf_idf']))
    
    sim = cosine('result/8.txt', 'result/9.txt')
    print(sim)
    sim = cosine('result/926.txt', 'result/928.txt')
    print(sim)
    sim = cosine('result/195.txt', 'result/229.txt')
    print(sim)
    sim = cosine('result/564.txt', 'result/565.txt')
    print(sim)

if __name__ == '__main__':
    main()