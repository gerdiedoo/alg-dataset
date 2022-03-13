class LinkedList:
    def __init__(self):
        self.head = None
    def addToStart(self, data):
        tempNode = Node(data)
        tempNode.setLink(self.head)
        self.head = tempNode
        del tempNode
    def addToEnd(self, data):
        start = self.head
        tempNode = Node(data)
        while start.getNextNode():
            start = start.getNextNode()
        start.setLink(tempNode)
        del tempNode
        return True
    def display(self):
        start = self.head
        if start is None:
            print("Empty List!!!")
            return False
        while start:
            print(str(start.getData()), end=" ")
            start = start.link
            if start:
                print("-->", end=" ")
        print()
    def length(self):
        start = self.head
        size = 0
        while start:
            size += 1
            start = start.getNextNode()
        return size
    def index(self, data):
        start = self.head
        position = 1
        while start:
            if start.getData() == data:
                return position
            else:
                position += 1
                start = start.getNextNode()
    def remove(self, item):
        start = self.head
        previous = None
        found = False
        while not found:
            if start.getData() == item:
                found = True
            else:
                previous = start
                start = start.getNextNode()
        if previous is None:
            self.head = start.getNextNode()
        else:
            previous.setLink(start.getNextNode())
        return found
    def Max(self):
        start = self.head
        largest = start.getData()
        while start:
            if largest < start.getData():
                largest = start.getData()
            start = start.getNextNode()
        return largest
    def Min(self):
        start = self.head
        smallest = start.getData()
        while start:
            if smallest > start.getData():
                smallest = start.getData()
            start = start.getNextNode()
        return smallest
    def push(self, data):
        self.addToEnd(data)
        return True
    def pop(self):
        start = self.head
        previous = None
        while start.getNextNode():
            previous = start
            start = start.getNextNode()
        if previous is None:
            self.head = None
        else:
            previous.setLink(None)
            data = start.getData()
            del start
            return data
    def atIndex(self, position):
        start = self.head
        position = int(position)
        pos = 1
        while pos != position:
            start = start.getNextNode()
            pos += 1
        data = start.getData()
        return data
    def copy(self):
        temp = LinkedList()
        start = self.head
        temp.addToStart(start.getData())
        start = start.getNextNode()
        while start:
            temp.addToEnd(start.getData())
            start = start.getNextNode()
        return temp
    def clear(self):
        self.head = None
        return True
    def removePosition(self, position):
        data = self.atIndex(position)
        self.remove(data)
        return data
    def toString(self, seperator=""):
        start = self.head
        finalString = ""
        while start:
            tempString = start.getData()
            finalString += str(tempString)
            start = start.getNextNode()
            if start:
                finalString += seperator
        return finalString
    def count(self, element):
        start = self.head
        count1 = 0
        while start:
            if start.getData() == element:
                count1 += 1
            start = start.getNextNode()
        return count1
    def toList(self):
        start = self.head
        tempList = []
        while start:
            tempElement = start.getData()
            tempList.append(tempElement)
            start = start.getNextNode()
        return tempList
    def toSet(self):
        start = self.head
        tempSet = set()
        while start:
            tempElement = start.getData()
            if tempElement not in tempSet:
                tempSet.add(tempElement)
            start = start.getNextNode()
        return tempSet
    def reverse(self):
        start = self.head
        tempNode = None
        previousNode = None
        while start:
            tempNode = start.getNextNode()
            start.setLink(previousNode)
            previousNode = start
            start = tempNode
        self.head = previousNode
        return True
    def sort(self):
        start = self.head
        beginNode = start
        while beginNode:
            tempNode = beginNode
            tempNode2 = beginNode
            smallest = beginNode.getData()
            while tempNode:
                if smallest > tempNode.getData():
                    smallest = tempNode.getData()
                    tempNode2 = tempNode
                tempNode = tempNode.getNextNode()
            temp = beginNode.getData()
            beginNode.updateData(tempNode2.getData())
            tempNode2.updateData(temp)
            beginNode = beginNode.getNextNode()
    def sorted(self):
        start = self.head
        tempList = self.copy()
        tempList.sort()
        return tempList
class Node:
    def __init__(self, data=None, link=None):
        self.data = data
        self.link = link
    def updateData(self, data):
        self.data = data
    def setLink(self, node):
        self.link = node
    def getData(self):
        return self.data
    def getNextNode(self):
        return self.link
myList = LinkedList()
myList.addToStart(5)
myList.addToStart(4)
myList.addToStart(3)
myList.addToStart(2)
myList.addToStart(1)
myList.display()
myList.addToEnd(12)
myList.addToEnd(13)
myList.addToEnd(3)
myList.display()
print(myList.length())
print(myList.index(3))
print(myList.atIndex(5))
print(myList.remove(12))
myList.removePosition(2)
myList.display()
print(myList.Max())
print(myList.Min())
print(myList.push(31))
myList.display()
print(myList.pop())
myList.display()
myList2 = myList.copy()
myList2.display()
myList2.clear()
myList2.display()
print(myList.toString(","))
print(myList.count(3))
newList = myList.toList()
print(newList)
newSet = myList.toSet()
print(newSet)
myList.reverse()
myList.display()
myList3 = myList.sorted()
myList3.display()
myList.sort()
myList.display()