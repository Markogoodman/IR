from porter import PorterStemmer
from MaxHeap import MaxHeap
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

class Cluster:
    def __init__(self, ID):
        self.clusterID = ID
        '''
        record similarity to other cluster
        priority queue, stores Similarity object
        index1 is self, index2 is other
        '''
        self.sim_heap = MaxHeap()
        self.doc_list = [ID]

    def merge(self, other):
        # merge another cluster to this cluster
        self.doc_list += other.doc_list



class Similarity:
    def __init__(self, sim, index):
        # similarity of current cluster to cluster index
        self.sim = sim
        self.index = index

    def __lt__(self, other):
        return self.sim < other.sim
    def __gt__(self, other):
        return self.sim > other.sim
    def __le__(self, other):
        return self.sim <= other.sim
    def __ge__(self, other):
        return self.sim >= other.sim

    def __str__(self):
        return  'sim to ' + str(self.index) + ' is ' + str(self.sim)

def cosine(doc_x, doc_y):
    sim = 0
    unitVector_x = doc_x.getUnitVector()
    unitVector_y = doc_y.getUnitVector()
    for term, term_info in unitVector_x.items():
        # every idf in doc_x is multiplied by idf in dox_y
        sim += term_info['tf_idf'] * unitVector_y.get(term,{'tf_idf':0})['tf_idf']

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
    
    dictionary = Dictionary(documents)
    for docId, document in sorted(documents.items()):
        document.updateIndex(dictionary)
        document.updateUnitVector(dictionary)
        
    docNum = len(documents)
    '''
    store { clusterid: Cluster}
    '''
    clusters = {}
    # calculate every sim between clusters
    for i in range(1, docNum+1):
        clusters[i] = Cluster(i)
        for j in range(1, docNum+1):
            if i != j:
                sim = cosine(documents[i], documents[j])
                clusters[i].sim_heap.insert(Similarity(sim, j))
    # do docNum-1 times merge
    for _ in range(docNum-1):
        #find two most similar cluster to merge
        max_sim = -1
        k1 = -1
        for id, cluster in clusters.items():
            if max_sim < cluster.sim_heap.max():
                max_sim = cluster.sim_heap.max()
                k1 = id
        k2 = clusters[k1].sim_heap.elements[0].index
        #ff.write(str(max_sim) + ', ' + str(k1) + ' and ' + str(k2) + '\n')
        clusters[k1].sim_heap = MaxHeap()
        #merge index to max_id
        clusters[k1].merge(clusters[k2])
        clusters.pop(k2)

        # update similarity between clusters
        for id, cluster in clusters.items():
            if id == k1:
                continue
            sim1 = cluster.sim_heap.delete(k1)
            sim2 = cluster.sim_heap.delete(k2)
            sim = sim1 if sim1 < sim2 else sim2
            cluster.sim_heap.insert(Similarity(sim, k1))
            clusters[k1].sim_heap.insert(Similarity(sim, id))

        if len(clusters) in [8, 13, 20]:
            k = len(clusters)
            with open(str(k) + '.txt', 'w') as f:
                for id, cluster in sorted(clusters.items()):
                    for docID in sorted(cluster.doc_list):
                        f.write(str(docID) + '\n')
                    f.write('\n')


if __name__ == '__main__':
    main()