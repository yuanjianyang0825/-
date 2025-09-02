class Car:
    def __init__(self,name):
        self.name=name
        self.remain_mile=0
    def fill_fuel(self,miles):#加燃料里程
        self.remain_mile+=miles
    def run(self,miles):#跑miles英里
        print(self.name,end=':')
        if self.remain_mile>=miles:
            self.remain_mile-=miles
            print(f" run {miles} miles")
        else:
            print(" fuel out!")
class GasCar(Car):
    def __init__(self,name,capacity):
        Car.__init__(self,name)
        self.capacity=capacity
    def fill_fuel(self,gas):#加汽油gas升
        self.remain_mile=gas*6.0#每升跑6英里
class ElecCar(GasCar):
    def fill_fuel(self,power):#充电power度
        self.remain_mile=power*3.0#每度跑3英里
gcar=GasCar("BMW",200)
gcar.fill_fuel(50)
gcar.run(200)