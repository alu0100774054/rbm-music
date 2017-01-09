# 			Generador de música

Proyecto final de la asignatura de Sistemas Inteligentes del itinerario de Computación de Ingeniería Informática de la Universidad de La Laguna



​	 ![](src/media/portada.jpg)



|            Nombre            |         Contacto         |
| :--------------------------: | :----------------------: |
| Erik Andreas Barreto de Vera | alu0100774054@ull.edu.es |
|    Jorge Alonso Hernández    | alu0100767803@ull.edu.es |



1. # Resumen

   El objetivo del generador de música es diseñar un sistema de aprendizaje basado en muestras. El sistema consta de una base de datos musical que forman parte de la entrada del sistema, a partir de esta información de entrada el sistema es capaz de aprender qué notas musicales son buenas para producir música parecida a la de entrada. De esta forma el sistema es capaz de producir música.

2. # Estado del arte

   1. # Magenta

      **¿Qué es Magenta?**

      Magenta es un proyecto del equipo de Google Brain. Magenta tiene dos objetivos. 

      En primer lugar, es un proyecto de investigación para avanzar el estado del arte en la inteligencia de la máquina para la música y la generación del arte. El aprendizaje automático ya se ha utilizado ampliamente para comprender el contenido, como en el reconocimiento del habla o la traducción.

      En segundo lugar, Magenta es un intento de construir una comunidad de artistas, codificadores y investigadores de aprendizaje automático. El equipo principal de Magenta construirá una infraestructura de código abierto alrededor de TensorFlow para hacer arte y música. Comenzaremos con soporte de audio y video, herramientas para trabajar con formatos como MIDI y plataformas que ayudan a los artistas a conectarse a modelos de aprendizaje automático. Por ejemplo, queremos que sea super simple para reproducir música junto con un modelo de rendimiento Magenta.

      **Evaluación de los resultados del sistema**

      Evaluar el resultado de los modelos generados es difícil. Llegará el momento en que Magenta tiene 20 modelos diferentes de generación de música disponibles en código abierto. ¿Cómo decidimos cuáles son buenas? Una opción es comparar el resultado del modelo con los datos de entrenamiento midiendo la verosimilitud.

      **Posibles mejoras del sistema**

      Para la música y el arte, esto no funciona muy bien. Es fácil generar resultados que están cerca en términos de probabilidad, pero lejos en términos de atractivo (y viceversa). Esto motiva el trabajo en adversarios artificiales como Generar redes neuronales diferentes. Al final, para responder a la pregunta de evaluación necesitamos que las herramientas de Magenta estén en manos de artistas y músicos, y *Magenta media* disponible para los espectadores y oyentes.

      **Enlace al proyecto**

      https://magenta.tensorflow.org/welcome-to-magenta

3. # Introducción

   Este sistema generador de música es un sistema inteligente capaz de generar música aprendiendo de diferentes muestras. A medida que el sistema se nutre de una mayor cantidad de muestras el sistema consigue mejorar sus resultados en términos de probabilidad. Esto es diferente a los resultados que nosotros esperemos, más adelante se comentará. A continuación se procede a explicar todo el proceso de creación, estructura, conocimientos, etc que compone este proyecto.

4. # Métodos

   En este apartado vamos a explicar aquellos conocimientos que hacen falta para comprender el sistema. No vamos a entrar a nivel de matemáticas ni estadísticas sino a nivel de herramientas o modelos. 

   {0}.  # MIDI

      **¿Qué es MIDI?**

      MIDI (abreviatura de Musical Instrument Digital Interface) es un estándar tecnológico que describe un protocolo, una interfaz digital y conectores que permiten que varios instrumentos musicales electrónicos, computadoras y otros dispositivos relacionados se conecten y comuniquen entre sí.

      **¿Qué es un archivo .midi?**

      El formato estándar MIDI (SMF) permite una manera estandarizada para almacenar, transportar y abrir secuencias en otros sistemas. El compacto tamaño de estos archivos ha permitido que sean implementados de manera numerosa en computadoras. Fueron creados para su uso universal e incluir información como el valor de las notas, tiempo y nombre de las pistas. La lírica puede ser incluida como metadata

      **¿Por qué MIDI en el proyecto?**

      Será el formato en el que vamos a trabajar para generar la música. La base de muestras del sistema se encuentra en formato .*midi*, compone la base de entrenamiento.

      **Más información**

      En estos enlaces se puede encontrar más información acerca de el formato de notación musical .*midi* :

      http://www.css-audiovisual.com/areas/guias/midi-mensajes.htm

      https://es.wikipedia.org/wiki/MIDI

   {0}.  # Restricted Boltzmann machine

      **¿Qué es? Definición y estructutra**

      Inventado por Geoff Hinton, las Máquinas Boltzmann Restringidas (RBMs) son redes neurales superficiales de dos capas que constituyen los bloques de construcción de redes de creencias profundas. La primera capa del RBM se denomina capa visible, o input, y la segunda es la capa oculta.

   ![https://deeplearning4j.org/img/two_layer_RBM.png](https://deeplearning4j.org/img/two_layer_RBM.png)

      Cada círculo en el gráfico anterior representa una unidad similar a una neurona llamada nodo, y los nodos son simplemente donde los cálculos tienen lugar. Los nodos están conectados entre sí a través de capas, pero no hay dos nodos de la misma capa conectados.

      Es decir, no hay comunicación intra-capa - esta es la restricción en una máquina Boltzmann restringida. Cada nodo es un locus de computación que procesa la entrada, y comienza haciendo decisiones estocásticas sobre si transmitir esa entrada o no. (Estocástico significa "determinado al azar", y en este caso, los coeficientes que modifican las entradas se inicializan al azar).

      Cada nodo visible toma una característica de bajo nivel de un elemento del conjunto de datos que se va a aprender. Por ejemplo, a partir de un conjunto de datos de imágenes en escala de grises, cada nodo visible recibiría un valor de píxel para cada píxel en una imagen. (Las imágenes MNIST tienen 784 píxeles, por lo que las redes neuronales que las procesan deben tener 784 nodos de entrada en la capa visible).

      Ahora vamos a seguir ese valor de píxel único, x, a través de la red de dos capas. En el nodo 1 de la capa oculta, x se multiplica por un peso y se añade a un llamado sesgo. El resultado de estas dos operaciones se alimenta a una función de activación, que produce la salida del nodo, o la intensidad de la señal que pasa a través de ella, dada la entrada x.

   ```python
      activation f((weight w * input x) + bias b ) = output a
   ```
   ![](https://deeplearning4j.org/img/input_path_RBM.png)
   ![](https://deeplearning4j.org/img/weighted_input_RBM.png)
   ![](https://deeplearning4j.org/img/multiple_inputs_RBM.png)
   ![](https://deeplearning4j.org/img/multiple_hidden_layers_RBM.png)
   Pero en esta introducción a las máquinas Boltzmann restringidas, nos centraremos en cómo aprenden a reconstruir los datos por sí mismos de una manera no supervisada (medios sin supervisión sin etiquetas de verdad en un conjunto de prueba), haciendo varios pasos hacia adelante y hacia atrás entre la capa visible Y la capa oculta sin implicar una red más profunda.
   En la fase de reconstrucción, las activaciones de la capa oculta se convierten en la entrada en un paso hacia atrás. Se multiplican por los mismos pesos, uno por borde de entrenudo, así como x se ajustó en peso en el paso hacia adelante. La suma de esos productos se añade a un sesgo de capa visible en cada nodo visible, y la salida de esas operaciones es una reconstrucción; Es decir, una aproximación de la entrada original. Esto puede representarse mediante el siguiente diagrama:
   ![](https://deeplearning4j.org/img/reconstruction_RBM.png)

   **Generación de cada nota**

   ![](/Users/zephyr/Desktop/3.png)![](/Users/zephyr/Desktop/2.png)![](/Users/zephyr/Desktop/1.png)
   **Enlace a un tutorial para más información**

   https://deeplearning4j.org/restrictedboltzmannmachine.html#

5. # Herramientas

   * ## Python

     Se trata de un lenguaje de programación multiparadigma, ya que soporta orientación a objetos, programación imperativa y, en menor medida, programación funcional. Es un lenguaje interpretado, usa tipado dinámico y es multiplataforma.

     **¿Por qué python?**

     Porque TensorFlow es una API sólo para python.

     **Instalación**

     [descarga: https://www.python.org/downloads/](https://www.python.org/downloads/)

     * ### Pip

       pip Es un sistema de gestión de paquetes utilizado para instalar y administrar paquetes de software escritos en Python.

       **Instalación**

       A partir de la versión 2.6 pip viene instalado junto con la versión de python.

     * ### dependecies

       Todos los paquetes que aparecen a continuación se instalan mediante el comando pip que vimos anteriormente.

       ```shell
       $ pip install
       ```

       * #### pandas

         Pandas es una librería de python destinada al análisis de datos, que proporciona unas estructuras de datos flexibles y que permiten trabajar con ellos de forma muy eficiente. Pandas ofrece las siguientes estructuras de datos:

         * **Series:** Son arrays unidimensionales con indexación (arrays con índice o etiquetados), similar a los diccionarios. Pueden generarse a partir de diccionarios o de listas.

         * **DataFrame:** Son estructuras de datos similares a las tablas de bases de datos relacionales como SQL. 

         * **Panel, Panel4D y PanelND:** Estas estructuras de datos permiten trabajar con más de dos dimensiones. Dado que es algo complejo y poco utilizado trabajar con arrays de más de dos dimensiones no trataremos los paneles en estos tutoriales de introdución a Pandas.

           **Instalación**

           ```shell
           $ pip install pandas
           ```

       * #### numpy

         NumPy es una extensión de Python, que le agrega mayor soporte para vectores y matrices, constituyendo una biblioteca de funciones matemáticas de alto nivel para operar con esos vectores o matrices.

         **Instalación**

         ```shell
         $ pip install numpy
         ```

       * #### msgpack

         MessagePack es un formato de serialización binaria rápido y compacto, adecuado para datos similares a JSON. Este paquete proporciona enlaces CPython para leer y escribir datos de MessagePack.

         **Instalación**

         ```shell
         $ pip install msgpack-python
         ```

       * #### glob

         El módulo glob encuentra todos los pathnames que coinciden con un patrón especificado de acuerdo con las reglas utilizadas por el shell de Unix, aunque los resultados se devuelven en orden arbitrario. Esto facilita la lectura de la base de datos musical.

         **Instalación**

         ```shell
         $ pip install glob2
         ```

       * #### tqdm

         Permite usar barras de progreso en una tarea en la linea de comandos.

         ```shell
         $ pip install tqdm
         ```


*  ## TensorFlow

   Es una librería open-source dedicada para machine learning.

   **Instalación**

   En el siguiente enlace se muestra la instalación en cada sistema operativo.

   [https://www.tensorflow.org/versions/r0.10/get_started/os_setup](https://www.tensorflow.org/versions/r0.10/get_started/os_setup)

*  ## Github

              GitHub es una plataforma de desarrollo colaborativo para alojar proyectos utilizando el sistema de control de versiones Git.

   * ## Bitbucket

     Bitbucket es un servicio de alojamiento basado en web, para los proyectos que utilizan el sistema de control de revisiones Mercurial y Git. 

   * ## HipChat

     HipChat es un servicio Web para chat interno / privado y mensajería instantánea. Además de un chat individual o en grupo / tema, también ofrece almacenamiento en archivos basado en la nube, videollamadas, historial de mensajes de búsqueda y visualización de imágenes en línea.

6. # Proceso

7. # Resultados

8. # Observaciones

9. # Conclusiones

10. # Referencias

[https://www.python.org](https://www.python.org)

[https://www.tensorflow.org/versions/r0.10/get_started/os_setup](https://www.tensorflow.org/versions/r0.10/get_started/os_setup)

https://es.wikipedia.org/wiki/Pip

http://pandas.pydata.org

[https://en.wikipedia.org/wiki/Pandas_(software)](https://en.wikipedia.org/wiki/Pandas_(software))

[http://msgpack.org](http://msgpack.org)

[https://github.com/tqdm/tqdm](https://github.com/tqdm/tqdm)

[https://github.com](https://github.com)

https://bitbucket.org