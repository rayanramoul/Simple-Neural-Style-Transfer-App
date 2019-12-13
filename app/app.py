import sys
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QFileDialog
from app.model import Transf
class App:
    def __init__(self):
        app = QApplication(sys.argv)

        self.window = QWidget()
        
        mainbox = QVBoxLayout()
        self.window.setLayout(mainbox)
        self.window.setWindowTitle("Simple Neural Style Transfer")
        hbox1 = QHBoxLayout() # Input image
        hbox2 = QHBoxLayout() # Style Image
        hbox3 = QHBoxLayout() # Output image

        label1 = QLabel("Your input picture ")
        label2 = QLabel("Style image ")
        label3 = QLabel("Output Image ")
        
        button1 = QPushButton("Pick")
        button2 = QPushButton("Pick")
        button3 = QPushButton("Generate")

        hbox1.addWidget(label1)
        hbox1.addWidget(button1)

        hbox2.addWidget(label2)
        hbox2.addWidget(button2)
        
        hbox3.addWidget(label3)
        hbox3.addWidget(button3)

        button1.clicked.connect(lambda:self.pickfile("input"))
        button2.clicked.connect(lambda:self.pickfile("style"))
        button3.clicked.connect(self.transform)

        mainbox.addLayout(hbox1)
        mainbox.addLayout(hbox2)
        mainbox.addLayout(hbox3)

        self.window.show() 
        app.exec_()
    
    def pickfile(self, typ):
        fname = QFileDialog.getOpenFileName(self.window, 'Open file', '~')
        if fname[0]:
            if typ=="input":
                print(" new input image "+str(fname[0]))
                self.input = fname[0]
            else:
                print(" new source image "+str(fname[0]))
                self.style = fname[0]

    def transform(self):
        print("Generate..")
        trans = Transf()
        trans.apply(self.input, self.style)