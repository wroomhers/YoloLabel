import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QVBoxLayout, QCheckBox, QLabel, QLineEdit
from PyQt5 import QtCore, QtGui
import cv2
import numpy as np
import os
import math 
from ultralytics import YOLO

classNames = [ "Saga_Donulmez", "Mecburu_Saga", "Sola_Donulmez", "Mecburi_Sola", "Ileri_Mecburi",
               "Ileri_veya_Saga_Mecburi", "Ileri_veya_Sola_Mecburi", "Dur", "Kapalı_Yol", "Durak",
                "Park", "Park_Yasak", "Duraklamak_ve_Park_Yasa", "Yaya_Gecidi", "Yesil_Trafik_Isigi", 
                "Sari_Trafik_Isigi", "Kirmizi_Trafik_Isigi", "Ilderden_Saga", "Ilerden_Sola", "Kapali_Yol", "Hatırlamiyom", "Park_Yasak_Yanlis"
              ]


class DosyaSecmeEkrani(QWidget):
    def __init__(self):
        super().__init__()
        self.etiketlenecek_veriler_yolu = "./"
        self.agirlik_dosyasi = "./best.pt"
        self.txt_yolu = None
        self.setFixedSize(330, 330)
        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, 330, 330)
        self.setWindowTitle('YOLOLabel')
        self.setWindowIcon(QtGui.QIcon('tetra_logo.png'))

        
        self.klasor_label = QLabel('Etiketlenecek verilerin konumu', self)
        self.klasor_label.setGeometry(30, 20, 1000, 30)

        self.klasor_textbox = QLineEdit(
            "", parent=self
        )
        self.klasor_textbox.setGeometry(30, 50, 160, 30)

        self.klasor_button = QPushButton('Klasör Seç', self)
        self.klasor_button.clicked.connect(self.klasor)
        self.klasor_button.setGeometry(200, 50, 100, 30)



        self.txt_klasor_button = QPushButton('Klasör Seç', self)
        self.txt_klasor_button.clicked.connect(self.txt_klasor)
        self.txt_klasor_button.setGeometry(200, 130, 100, 30)

        self.txt_checkbox = QCheckBox('txt dosyalarını farklı bir konuma kaydet', self)
        self.txt_checkbox.setGeometry(30, 100, 300, 30)
        self.txt_checkbox.stateChanged.connect(self.toggle_button)
        self.txt_klasor_button.setDisabled(True)
        self.txt_klasor_textbox = QLineEdit(
            "", parent=self
        )
        self.txt_klasor_textbox.setGeometry(30, 130, 160, 30)
        self.txt_klasor_textbox.setDisabled(True)

        


        self.agirlik_label = QLabel('Ağırlık Dosyası', self)
        self.agirlik_label.setGeometry(30, 180, 1000, 30)

        self.agirlik_textbox = QLineEdit(
            "", parent=self
        )
        self.agirlik_textbox.setGeometry(30, 210, 160, 30)

        self.agirlik_button = QPushButton('Dosya Seç', self)
        self.agirlik_button.clicked.connect(self.agirlik_doyasi)
        self.agirlik_button.setGeometry(200, 210, 100, 30)




        self.etikete_basla_button = QPushButton('Etiketle!', self)
        self.etikete_basla_button.clicked.connect(self.derleyici)
        self.etikete_basla_button.setGeometry(200, 270, 100, 30)

        self.show()


    def gorsel_koyan(self):
        self.derleyici()
        
    def derleyici(self):
        if self.txt_checkbox.isChecked():
            oto_etiket(self.etiketlenecek_veriler_yolu, self.agirlik_dosyasi, self.txt_yolu)
        else:
            oto_etiket(self.etiketlenecek_veriler_yolu, self.agirlik_dosyasi)
    def agirlik_doyasi(self):
        self.agirlik_dosyasi, _ = QFileDialog.getOpenFileName(self, 'Ağırlık Dosyası', '', 'Ağırlık Dosyası (*.pt);;Tüm Dosyalar (*.*)')
        if self.agirlik_dosyasi:
            print("Seçilen dosya:", self.agirlik_dosyasi)
            self.agirlik_textbox.setText(self.agirlik_dosyasi)
    def txt_klasor(self):
        #self.txt_yolu, _ = QFileDialog.getOpenFileName(self, 'Etiketlenecek Verilerin Konumu', '', 'Ağırlık Dosyası (*.pt);;Tüm Dosyalar (*.*)')
        self.txt_yolu= QFileDialog.getExistingDirectory(self, 'txt dosyalarının kaydedileceği konum', '')
        if self.txt_yolu:
            print("Seçilen Klasör:", self.txt_yolu)
            self.txt_klasor_textbox.setText(self.txt_yolu)


    def klasor(self):
        self.etiketlenecek_veriler_yolu= QFileDialog.getExistingDirectory(self, 'Etiketlenecek Verilerin Bulunduğu Klasör', '')
        if self.etiketlenecek_veriler_yolu:
            print("Seçilen Klasör:", self.etiketlenecek_veriler_yolu)
            self.klasor_textbox.setText(self.etiketlenecek_veriler_yolu)

    def toggle_button(self):
        if self.txt_checkbox.isChecked():
            self.txt_klasor_button.setEnabled(True)
            self.txt_klasor_textbox.setEnabled(True)
        else:
            self.txt_klasor_button.setDisabled(True)
            self.txt_klasor_textbox.setDisabled(True)

def convert(size, box):
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (box[0] + box[1]) / 2.0
        y = (box[2] + box[3]) / 2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return [x, y, w, h]

def boxlar(boxes, img_file, img, txt_yolu):
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            #print("Confidence --->",confidence)
            # class name
            cls = int(box.cls[0])
            #print("Class name -->", classNames[cls])
            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

            b = (float(x1), float(x2), float(y1), float(y2))
            yolo_box = convert((img.shape[1], img.shape[0]), b)

            f = open(txt_yolu + os.path.basename(img_file)[:-4] + ".txt", "a")
            f.write(str(cls) + " " + str(yolo_box[0])[0:8] + " " + str(yolo_box[1])[0:8] + " " + str(yolo_box[2])[0:8] + " " + str(yolo_box[3])[0:8] + " " + "\n")
            f.close()

            #cv2.imshow("Slide Show", img)

def oto_etiket(klasor, agirlik_dosyasi, txt_yolu = None):

    klasor = klasor + "/"
    if txt_yolu:
        txt_yolu = txt_yolu + "/"
    else:
        txt_yolu = klasor
    images = os.listdir(klasor)    # Get their names in a list
    length = len(images)
    result = np.zeros((301, 300, 3), np.uint8)        # Image window of size (360, 360)
    i = 1
    for i in range(length):
        if images[i][-4:] != ".txt":
            #img = cv2.imread(klasor + images[i])
            img = cv2.imread(klasor + images[i])
            print("agirlik dosyam: " + agirlik_dosyasi)
            model = YOLO(agirlik_dosyasi)
            print("resim dosyam: " + klasor + images[i])
            #cv2.imshow("of", img)
            results = model(img)
            boxlar(results[0].boxes, (klasor + images[i]), img, txt_yolu)
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DosyaSecmeEkrani()
    sys.exit(app.exec_())
