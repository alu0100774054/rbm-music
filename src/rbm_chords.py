import numpy as np
import pandas as pd
import msgpack
import glob
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tqdm import tqdm

###################################################
# El siguiente archivo es capaz de manipular formato
# midi.
import midi_manipulation

# Este metodo obtiene las canciones del directorio gracias a la libreria glob.
def get_songs(path):
    files = glob.glob('{}/*.mid*'.format(path))
    songs = []
    for f in tqdm(files):
        try:
            song = np.array(midi_manipulation.midiToNoteStateMatrix(f))
            if np.array(song).shape[0] > 50:
                songs.append(song)
        except Exception as e:
            raise e           
    return songs

#Estas canciones ya estan convertidas de formato midi a msgpack.
songs = get_songs('Pop_Music_Midi')
print "{} songs processed".format(len(songs))

# Red neuronal:

# Indice de la nota mas bajas de una pianola.
lowest_note = midi_manipulation.lowerBound 

# Indice de la nota mas alta de una pianola.
highest_note = midi_manipulation.upperBound 

# Rango de notas.
note_range = highest_note-lowest_note 

# numero de pasos que genera.
num_timesteps  = 100 

# Tamanio de la capa visible.
n_visible      = 2*note_range*num_timesteps

# Tamaio de la capa oculta.
n_hidden       = 50 

# El numero de entrenamiento que vamos a correr. Para cada uno examinamos todo el conjunto de datos.
num_epochs = 200

# El número de ejemplos de entrenamiento que vamos a enviar a través del RBM a la vez.
batch_size = 100 

# La tasa de aprendizaje de nuestro modelo
lr         = tf.constant(0.005, tf.float32)

# Variables:

# La variable de marcador de posicion que contiene nuestros datos.
x  = tf.placeholder(tf.float32, [None, n_visible], name="x")

# La matriz de peso que almacena los pesos de las conexiones.
W  = tf.Variable(tf.random_normal([n_visible, n_hidden], 0.01), name="W") 

# Bias de la capa oculta. 
bh = tf.Variable(tf.zeros([1, n_hidden],  tf.float32, name="bh"))

# Bias de la capa visible.
bv = tf.Variable(tf.zeros([1, n_visible],  tf.float32, name="bv"))

# Devuelve un ejemplo de 0s y 1s en funcion de probabilidades.
def sample(probs):
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))

# Esta funcion ejecuta la cadena de gibbs. Llamaremos esta funcion en dos lugares:
# - Cuando definimos el paso de actualizacion del entrenamiento.
# - Cuando muestreamos nuestros segmentos musicales del RBM entrenado.
def gibbs_sample(k):

    # Ejecuta una cadena de gibbs k-pasos para muestrear la distribucion de probabilidad de la RBM definida por W, bh, bv
    def gibbs_step(count, k, xk):
        
        # Ejecuta un solo paso gibbs. Los valores visibles se inicializan en xk
        # Propagar los valores visibles para muestrear los valores ocultos
        hk = sample(tf.sigmoid(tf.matmul(xk, W) + bh)) 

        # Propagar los valores ocultos para muestrear los valores visibles
        xk = sample(tf.sigmoid(tf.matmul(hk, tf.transpose(W)) + bv)) 
        return count+1, k, xk

    # contador.
    ct = tf.constant(0)
    [_, _, x_sample] = control_flow_ops.While(lambda count, num_iter, *args: count < num_iter,
                                         gibbs_step, [ct, tf.constant(k), x], 1, False)
    x_sample = tf.stop_gradient(x_sample) 
    return x_sample

# ENTRENAMIENTO:

x_sample = gibbs_sample(1)
h = sample(tf.sigmoid(tf.matmul(x, W) + bh))
h_sample = sample(tf.sigmoid(tf.matmul(x_sample, W) + bh)) 

# A continuación, actualizamos los valores de W, bh y bv, 
# basados en la diferencia entre las muestras que dibujamos 
# y los valores originales
size_bt = tf.cast(tf.shape(x)[0], tf.float32)
W_adder  = tf.mul(lr/size_bt, tf.sub(tf.matmul(tf.transpose(x), h), tf.matmul(tf.transpose(x_sample), h_sample)))
bv_adder = tf.mul(lr/size_bt, tf.reduce_sum(tf.sub(x, x_sample), 0, True))
bh_adder = tf.mul(lr/size_bt, tf.reduce_sum(tf.sub(h, h_sample), 0, True))

# Ejecuta los 3.
updt = [W.assign_add(W_adder), bv.assign_add(bv_adder), bh.assign_add(bh_adder)]


### Run the graph!
# Now it's time to start a session and run the graph! 

with tf.Session() as sess:
    
    # Iniciamos el modelo.
    init = tf.initialize_all_variables()
    sess.run(init)
    
    # Ejecutar todos los datos de entrenamiento num_epochs veces
    for epoch in tqdm(range(num_epochs)):
        for song in songs:
            
            # Las canciones se almacenan en un formato de tiempo x notas. El tamaioo de cada cancion es timesteps_in_song x 2 * note_range
            # Aqui remodelamos las canciones para que cada ejemplo de entrenamiento sea un vector con num_timesteps x 2 * elementos note_range
            song = np.array(song)
            song = song[:np.floor(song.shape[0]/num_timesteps)*num_timesteps]
            song = np.reshape(song, [song.shape[0]/num_timesteps, song.shape[1]*num_timesteps])
            
            # Entrenar la RBM con ejemplos(tantos como batch_size)
            for i in range(1, len(song), batch_size): 
                tr_x = song[i:i+batch_size]
                sess.run(updt, feed_dict={x: tr_x})

    sample = gibbs_sample(1).eval(session=sess, feed_dict={x: np.zeros((10, n_visible))})
    for i in range(sample.shape[0]):
        if not any(sample[i,:]):
            continue
        
        # Aqui vamos a remodelar el vector hora x notas, y luego guardar el vector como un archivo midi
        S = np.reshape(sample[i,:], (num_timesteps, 2*note_range))
        midi_manipulation.noteStateMatrixToMidi(S, "generated_chord_{}".format(i))
            