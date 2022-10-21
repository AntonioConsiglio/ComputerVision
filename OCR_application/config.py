import os

HOME_PATH = os.getcwd()

class Point():
    def __init__(self,cordinates):
        self.x1 = cordinates[0]
        self.y1 = cordinates[1]
        self.x2 = cordinates[2]
        self.y2 = cordinates[3]
    
    def getfirst(self):
        return (self.x1,self.y1)

    def getsecond(self):
        return (self.x2,self.y2)

## IMAGE ZONE WHERE TO PREDICT
NAME = Point([176,470,341,590])
TARGA = Point([460,310,645,362])
DATA = Point([189,610,401,660])
CF = Point([432,600,760,660])
VIA = Point([185,715,690,749])
