from porter import PorterStemmer
import re

def getTokens(path):
    """
    receive file path
    return a lower case token list
    """
    tokens = []
    with open(path, 'r') as f:
        tokens = list(filter(None, re.split(r'\W', f.read())))
    tokens = [token.lower() for token in tokens if len(token) > 1 ]
    return tokens

def removeStopWords(tokens, path):
    """
    give tokens and stop-word-list filepath
    return tokens without stop words
        """
    stopWords = []
    with open(path, 'r') as f:
        stopWords = set(f.read().splitlines())
    tokens = [token for token in tokens if token not in stopWords]

    return tokens

def stem(tokens):
    """
    receive tokens
    return stemmedTokens
    """
    stemmedTokens = []
    stemmer = PorterStemmer()
    for token in tokens:
        stemmedTokens.append(stemmer.stem(token, 0, len(token)-1))

    return stemmedTokens

def main():
    tokens = []
    stemmedTokens = []
    terms = []
    tokens = getTokens('28.txt')
    tokens = removeStopWords(tokens, 'stop-word-list.txt')
    stemmedTokens = stem(tokens)
    # remove dup elements and sort list
    terms = sorted(list(set(stemmedTokens)))
    # write terms to result.txt
    with open("result.txt", 'w') as f:
        for term in terms:
            f.write(term + '\n')

if __name__ == '__main__':
    main()