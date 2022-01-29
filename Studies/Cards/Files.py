from msilib.schema import File
import os
import time
import cv2
from glob import glob

#Pegar os nomes dos arquivos
FileNames = os.listdir('./Videos')
FileNames = FileNames

#Passar por cada arquivo
for File in FileNames:
    print(File)
    time.sleep(3)
    #Retirar frames
    vidcap = cv2.VideoCapture('./Videos/'+File)
    success, image = vidcap.read()
    count = 0
    Where = '.\\'+File[:-3]+'\\'
    os.mkdir(File[:-3]) #Criar diretorio
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*50))
        cv2.imwrite(Where+'frame%d.jpg' % count, image)
        success, image = vidcap.read()
        print('Read a new frame: ', success, count)
        count += 1