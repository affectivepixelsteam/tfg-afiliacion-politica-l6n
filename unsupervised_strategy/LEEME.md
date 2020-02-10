# politicsInTheWild LEEME.md

Fichero readme.md en castellano

## Estructura de ficheros

Dado el carácter modular de este proyecto, se ha dividido el pipeline en varios apartados, totalmente independientes entre sí, y que operan por separado si se necesitara. Así, todos los scripts se desarrollan admitiendo argumentos que definen los parámetros de funcionamiento de esa parte específica. Dichos argumentos de entrada son definidos en un
sistema de *clases* en `arguments.py`. Como se observa, cada tarea tiene asociada una clase de argumentos específica. Los únicos argumentos que se llaman siempre para todos los programas son:
- output_dir (str): Directorio donde se guardarán los resultados gráficos y de cómputo derivados de los programas. Por default, 'results/'
- quiet (bool): Silencia los mensajes mostrados por pantalla. Por default, se muestra información  de los procesos ejecutados así como de los parámetros de entrada.

Con todo, para llamar, por ejemplo, al procedimiento `procedimiento.py`, uno introduciría en terminal:
```
python procedimiento.py <argumentos-positional> --option
``` 

Todos estos scripts trabajan *a nivel programa*, por lo que un wrapper sería necesario para ejecutar estos procesos para un conjunto de programas.

## Scripts (EN DESARROLLO)

Hasta el momento los procesos desarrollados incluyen:
1. `arguments.py` - Definición de las clases argumentos de entrada
2. `split_in_frames.py` - Extracción de frames a partir de programas concretos
3. `detection_and_encoding.py` - Detección de rostros con MTCNN y codificación Facenet
4. `dbscan.py` - DBSCAN hparams search y clustering (EN DESARROLLO)

De nuevo, se incluye una explicación detallada de los parámetros de entrada para cada script. Por lo general, todos los hiperparámetros, así como directorios de entrada y salida, pueden considerarse modificables por el usuario.

### arguments.py

Nothing to declare

### split_in_frames.py

A partir del path a un programa concreto, extrae usando *ffmpeg* una serie de frames que serán las unidades de trabajo a partir de ese momento.

### detection_and_encoding.py

Debido a la estrecha relación entre los procesos de detección y codificación, se han juntado en una sola llamada frame-by-frame. Por tanto, para cada frame extraído previamente, detectamos rostros y guardamos su codificación Facenet en un fichero externo. Opcionalmente se pueden guardar también el resultado de las detecciones. 

Para acceder a estos elementos, se guarda la información de rutas de bounding box (si se han guardado), de embeddings, tamaño de rostro y confianza del modelo de detección MTCNN en un fichero resumen con el nombre:
```
<output_dir>/<video_name>/detection_and_encoding.csv
```

### dbscan.py

Clustering. Ahora mismo, en desarrollo de una búsqueda automática de los valores de épsilon y min_samples. Por el momento, almacena gráficas de distancias a primeros vecinos y de histogramas de tamaño de caras a partir de los datos del fichero producido por la extracción anterior.


