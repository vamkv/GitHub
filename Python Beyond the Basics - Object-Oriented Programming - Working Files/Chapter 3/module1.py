
class MaxSizeList(object):
    def __init__(self,len=0):
        self.list = []
        self.le = len
    def push(self,string):
        if self.list.__len__() < self.le:
            self.list.append(string)
        else:
            self.list.pop(0)
            self.list.append(string)
    def get_list(self):
        return self.list





