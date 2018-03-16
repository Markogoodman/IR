class MaxHeap:
    def __init__(self):
        # store element as heap
        self.elements = []
        # use element's index for mapping to elements's index
        # index can not duplicate  
        self.map = {}

    def insert(self, element):
        self.elements.append(element)
        self.map[element.index] = len(self.elements) - 1

        i = len(self.elements) - 1
        while i > 0:
            j = (i-1)//2
            if self.elements[i] > self.elements[j]:
                # need to change
                self.elements[i], self.elements[j] = self.elements[j], self.elements[i]
                self.map[self.elements[i].index], self.map[self.elements[j].index] = self.map[self.elements[j].index], self.map[self.elements[i].index]
                i = j
            else:
                break

    def sift_down(self, i):
        length = len(self.elements)
        while i*2 + 1 < length:
            j = i*2 + 1
            if j < length - 1 :
                if self.elements[j+1] > self.elements[j]:
                    j += 1
            if self.elements[i] < self.elements[j]:
                self.elements[i], self.elements[j] = self.elements[j], self.elements[i]
                self.map[self.elements[i].index], self.map[self.elements[j].index] = self.map[self.elements[j].index], self.map[self.elements[i].index]
                i = j
            else:
                break

    def max(self):
        # return max element
        return self.elements[0].sim

    def pop(self):
        # pop out max element
        e = self.elements[0]
        # exchange
        self.elements[0], self.elements[-1] = self.elements[-1], self.elements[0]
        self.map[self.elements[0].index], self.map[self.elements[-1].index] = self.map[self.elements[-1].index], self.map[self.elements[0].index]

        self.map.pop(e.index)
        del self.elements[-1]
        self.sift_down(0)

        return e

    def delete(self, index):
        i = self.map[index]
        self.elements[i], self.elements[-1] = self.elements[-1], self.elements[i]
        self.map[self.elements[i].index], self.map[self.elements[-1].index] = self.map[self.elements[-1].index], self.map[self.elements[i].index]

        self.map.pop(self.elements[-1].index)
        sim = self.elements[-1].sim
        del self.elements[-1]
        self.sift_down(i)
        return sim

    def heapify(self):
        i = (len(self.elements) - 1) // 2
        while i >= 0:
            self.sift_down(i)
            i -= 1







