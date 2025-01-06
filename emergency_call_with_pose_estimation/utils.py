import time

def select_color(angle):
    if angle < 180:
        color = (0,255,0)
    elif angle < 300:
        color = (0,102,255)
    else:
        color = (0,0,255)
    return color

class Result():

    def __init__(self,result,id):
        self.result = result
        self.time = time.time()
        self.id = id
    
    def getx(self,point = "15"):
        return self.result[point][0]
    
    def gety(self,point = "15"):
        return self.result[point][1]

    def __sub__(self,array):
        if array.gety() < array.gety("12"):
            if self.gety() < self.gety("12"):
                return abs(self.getx() - array.getx())
        return 0
    
    def __str__(self) -> str:
        return f"ID: {self.id} ---- {self.time} \n {self.result}"
    
    def calculate_scalefactor(self):
        return abs(self.result["11"][0] - self.result["12"][0])
        


class ResultsManager():

    def __init__(self,timelimit = 2,call_help= 5):
        self.timelimit = timelimit
        self.results = []
        self.idcount = 0
        self.call_help = call_help
    def add(self,result):
        if len(result) >2:
            self.results.append(Result(result,self.idcount))
            self.idcount+=1
        #print(self.results[-1])
    
    def calculateDistance(self):
        distance = 0
        tmpresult = None
        scalefactor = 0
        result: Result
        if len(self.results) > 1:
            for result in self.results:
                if tmpresult is None:
                    tmpresult = result
                    scalefactor = result.calculate_scalefactor()
                    continue
                distance += result-tmpresult
                tmpresult = result
            return min(self.call_help,distance/scalefactor)
        return 0

    def filter_results_bytime(self):
        now = time.time()
        for idx,result in enumerate(self.results):
            if now-result.time > self.timelimit:
                self.results.pop(idx)