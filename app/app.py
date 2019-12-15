import sys
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QFileDialog, QLineEdit
from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore
from PyQt5 import Qt
from app.model import Transf
class App:
    def __init__(self):
        app = QApplication(sys.argv)
        

        import os.path
        from os import path
        
        if not path.exists("./generated") and path.isdir('./generated'):
            os.mkdir('./generated')
        if not path.exists("./sources") and path.isdir('./sources'):
            os.mkdir('./sources')

        self.window = QWidget()
        self.window.setStyleSheet("background-color: #231f20")
        mainbox = QHBoxLayout()
        self.window.setLayout(mainbox)
        self.window.setWindowTitle("Simple Neural Style Transfer")
        hbox1 = QVBoxLayout() # Input image
        hbox2 = QVBoxLayout() # Style Image
        hbox3 = QVBoxLayout() # Output image

        label1 = QLabel("Input picture")
        label1.setStyleSheet("color: #e8e6e7")
        label1.setAlignment(Qt.Qt.AlignCenter)
        label2 = QLabel("Style image")
        label2.setAlignment(Qt.Qt.AlignCenter)
        label2.setStyleSheet("color: #e8e6e7")
        label4 = QLabel("File Name")
        label4.setAlignment(Qt.Qt.AlignCenter)
        label4.setStyleSheet("color: #e8e6e7")
        label5 = QLabel("Number of Iterations")
        label5.setAlignment(Qt.Qt.AlignCenter)
        label5.setStyleSheet("color: #e8e6e7")
        self.label3 = QLineEdit("name.jpg")
        self.label3.setStyleSheet("background-color: white")
        self.num_iterations = QLineEdit("300")
        self.num_iterations.setStyleSheet("background-color: white")
        button1 = QPushButton("Pick")
        button1.setStyleSheet("background-color: #e6dd90")
        button2 = QPushButton("Pick")
        button2.setStyleSheet("background-color: #e6dd90")
        button3 = QPushButton("Generate")
        button3.setStyleSheet("background-color: #e6dd90")
        
        self.img1 = QLabel()
        self.img2 = QLabel()

        pixmap = QPixmap(400, 400)
        self.img1.setPixmap(pixmap)
        self.img1.setFixedWidth(400)
        self.img1.setFixedHeight(400)
        self.img1.setScaledContents(True)

        pixmap = QPixmap(400, 400)
        self.img2.setPixmap(pixmap)
        self.img2.setFixedWidth(400)
        self.img2.setFixedHeight(400)
        self.img2.setScaledContents(True)

        hbox1.addWidget(label1)
        hbox1.addWidget(button1)
        hbox1.addWidget(self.img1)

        hbox2.addWidget(label2)
        hbox2.addWidget(button2)
        hbox2.addWidget(self.img2)

        hbox3.addWidget(label4)
        hbox3.addWidget(self.label3)
        hbox3.addWidget(label5)
        hbox3.addWidget(self.num_iterations)
        hbox3.addWidget(button3)

        button1.clicked.connect(lambda:self.pickfile("input"))
        button2.clicked.connect(lambda:self.pickfile("style"))
        button3.clicked.connect(self.transform)

        mainbox.addLayout(hbox1)
        mainbox.addLayout(hbox2)
        mainbox.addLayout(hbox3)
        self.window.setFixedSize(1024, 480)
        self.window.show() 
        app.exec_()
    
    def pickfile(self, typ):
        fname = QFileDialog.getOpenFileName(self.window, 'Open file', '~')
        if fname[0]:
            if typ=="input":
                print(" new input image "+str(fname[0]))
                self.input = fname[0]
                pixmap = QPixmap(self.input)
                pixmap.scaled(400, 400, QtCore.Qt.KeepAspectRatio)
                self.img1.setPixmap(pixmap)

            else:
                print(" new source image "+str(fname[0]))
                self.style = fname[0]
                pixmap = QPixmap(self.style)
                pixmap.scaled(400, 400, QtCore.Qt.KeepAspectRatio)
                self.img2.setPixmap(pixmap)

    def transform(self):
        print("Generate..")
        trans = Transf()
        trans.apply(self.input, self.style, './generated/'+str(self.label3.text()), int(self.num_iterations.text()))