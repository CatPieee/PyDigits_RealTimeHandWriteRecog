from PySide6.QtCore import Slot
class A:
    def __init__(self):
        self.a = 1
    
    @Slot()
    def hello(self):
        print('hello I am A')

class B(A):
    def __init__(self):
        super().__init__()
        self.b = 2
    
    # def hello(self):
    #     print('hello I am B')

b = B()
b.hello()