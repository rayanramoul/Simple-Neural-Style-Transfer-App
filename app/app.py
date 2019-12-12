import sys
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel

class App:
    def __init__(self):
        app = QApplication(sys.argv)

        window = QWidget()
        
        mainbox = QVBoxLayout()
        window.setLayout(mainbox)

        hbox1 = QHBoxLayout() # Input image
        hbox2 = QHBoxLayout() # Style Image
        hbox3 = QHBoxLayout() # Output image

        label1 = QLabel("Your input picture ")
        label2 = QLabel("Style image ")
        label3 = QLabel("Output Image ")
        
        hbox1.addWidget(label1)
        hbox2.addWidget(label2)
        hbox3.addWidget(label3)
        
        mainbox.addLayout(hbox1)
        mainbox.addLayout(hbox2)
        mainbox.addLayout(hbox3)

        window.show() 
        app.exec_()