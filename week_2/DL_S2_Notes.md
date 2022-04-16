### clase sobre convnet

----

[rev] practica

* Siempre hacer un validation y test
  * Estratificado cuando la clasificación hay muy pocas muestras de una clase (ISLAND)

* probad medidas de regularizacion:
  * L2
  * Dropout
  * BatchNorm
* meter métricas (f1, recall, prec, acc)
* cuidado de las gráficas de loss accuracy
  * bajar el learning rate
  * subir batch size (?)

----

### ConvNet

1. Casos de uso
   1. Imágenes

A. LA CONVOLUCIÓN

* Detecta patrones

* Aplica diferentes filtros, cada filtro crea un mapa de activación. Los filtros son creados por la propia rnn

* Hiperparámetros_
  * Kernel size (matrix size)
  * Stride
  * ???
* Explicabilidad



B. POOLING

​	* Redución del mapa de activación mediante una agregación ("pooling")

	* Max pooling
	* Min pooling 
	* etc



---

Problemas de imagenes

* variabilidad de luz (occlusion)
* posicion (angulo)
* posturas (deformación, expresiones faciales)
* variabilidad de label (breed dogs)

---

### Transfer Learning



[link](!)

