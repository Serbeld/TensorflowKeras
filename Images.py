# encoding: utf-8 

# Images.py
# Este código va orientado a la toma de imágenes de video mediante la librería open cv en Python...
# Genera un número finito de imagenes en la carpeta enumerandolas del 1 al final de la toma de datos
# Para finalizar la toma de datos pulsa la tecla Q minúscula "q"

# Programador Sergio Luis Beleño Díaz
# 29.Octubre.2019

#Para empezar se importa la librería de Open cv para visión Artificial

import cv2

# Asignamos la cámara ingresando cv2.VideoCapture(0)
# Si quiere asignar una segunda cámara externa puede usar cv2.VideoCapture(1)
cap = cv2.VideoCapture(0)

formato = '.png'
name = 0

while(True):

	# Toma parámetros de captura de la cámara
	[rec, camara] = cap.read()
	name = name + 1 # Contador de imagenes tomadas

	# Muestra la imagen tomada en una ventana
	cv2.imshow('Capturas', camara)

	file = str(name) + formato
	# Reescala la imagen a 80x80px
	camara = cv2.resize(camara, (80,80))
	# Guarda la imagen tomada
	cv2.imwrite(file,camara)

	print("Imagen numero: " + str(name))

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
		
cap.release()
cv2.destroyAllWindows()