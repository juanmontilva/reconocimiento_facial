# Guía de Integración con Flask: Creando una Aplicación Web Completa

¡Hola! Esta guía te llevará paso a paso a través del proceso de transformar tu proyecto de un script a una aplicación web interactiva usando Flask. Crearemos una interfaz para ver el video en tiempo real y una sección de administración para añadir nuevas personas al sistema de reconocimiento facial.

## ¿Por qué esta arquitectura?

Usar Flask nos permite tener un **servidor central** que maneja tanto la lógica de visión por computadora como las peticiones web. Esto crea una aplicación robusta, escalable y fácil de gestionar.

**Lo que vamos a construir:**
1.  Una página principal que muestra el video de la cámara con el reconocimiento facial en tiempo real.
2.  Una página de administración (`/admin`) donde puedes escribir un nombre, subir una foto y añadir a esa persona a la base de datos de rostros conocidos.

---

## Paso 1: Estructura del Proyecto

Para mantener todo organizado, vamos a reestructurar ligeramente el proyecto. Crea las siguientes carpetas y archivos si no existen:

```
/Reconocimiento-facial-emociones/
|-- app.py                  # <-- NUESTRO SERVIDOR FLASK PRINCIPAL
|-- face_utils.py           # <-- Moveremos la lógica de reconocimiento aquí
|-- requirements.txt        # (Añadiremos Flask)
|-- known_faces/            # (Carpeta para guardar las imágenes de las personas)
|   |-- .gitkeep            # (Un archivo vacío para que git no ignore la carpeta)
|-- templates/              # <-- NUEVA CARPETA para los archivos HTML
|   |-- index.html          # (Página para ver el video)
|   |-- admin.html          # (Página para añadir personas)
|-- ADVANCED_CONCEPTS.md
|-- ... (tus otras carpetas)
```

---

## Paso 2: Actualizar Dependencias

Añade `Flask` a tu archivo `requirements.txt` y luego instálalo.

**requirements.txt:**
```
numpy
opencv-python
face_recognition
Flask
```

**Comando de instalación:**
```bash
pip install -r requirements.txt
```

---

## Paso 3: La Lógica de Reconocimiento (`face_utils.py`)

Crea el archivo `face_utils.py`. Cortaremos y pegaremos la lógica de reconocimiento facial aquí para mantener nuestro archivo `app.py` limpio.

```python
# face_utils.py
import face_recognition
import numpy as np
import os

# Variables globales para almacenar los rostros conocidos en memoria
KNOWN_FACE_ENCODINGS = []
KNOWN_FACE_NAMES = []
KNOWN_FACES_DIR = "known_faces"

def load_known_faces():
    """Carga los rostros desde el directorio y los almacena en las variables globales."""
    global KNOWN_FACE_ENCODINGS, KNOWN_FACE_NAMES
    
    # Limpiamos las listas antes de cargar
    KNOWN_FACE_ENCODINGS.clear()
    KNOWN_FACE_NAMES.clear()

    print("Cargando rostros conocidos...")
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)

    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith((".jpg", ".png")):
            try:
                path = os.path.join(KNOWN_FACES_DIR, filename)
                name = os.path.splitext(filename)[0]
                image = face_recognition.load_image_file(path)
                encoding = face_recognition.face_encodings(image)[0]
                KNOWN_FACE_ENCODINGS.append(encoding)
                KNOWN_FACE_NAMES.append(name)
                print(f"- Rostro de '{name}' cargado.")
            except IndexError:
                print(f"ADVERTENCIA: No se encontró rostro en {filename}. Saltando.")
            except Exception as e:
                print(f"Error cargando {filename}: {e}")

def process_frame_for_recognition(frame):
    """Procesa un solo frame para encontrar y nombrar rostros."""
    # Reducir el tamaño del frame para un procesamiento más rápido
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Encuentra todas las caras en el frame actual
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(KNOWN_FACE_ENCODINGS, face_encoding)
        name = "Desconocido"

        face_distances = face_recognition.face_distance(KNOWN_FACE_ENCODINGS, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = KNOWN_FACE_NAMES[best_match_index]
        
        face_names.append(name)

    # Dibuja los resultados en el frame original (no el reducido)
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Escalar las coordenadas de vuelta al tamaño original
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Dibuja el rectángulo y el nombre
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    return frame
```

---

## Paso 4: El Servidor Flask (`app.py`)

Este es el corazón de nuestra aplicación. Maneja las rutas web, el streaming de video y la lógica para añadir nuevas personas.

```python
# app.py
from flask import Flask, render_template, Response, request, redirect, url_for, flash
import cv2
import os
from werkzeug.utils import secure_filename

# Importamos nuestras funciones de utilidad
import face_utils

app = Flask(__name__)

# Configuración para subida de archivos
UPLOAD_FOLDER = face_utils.KNOWN_FACES_DIR
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'supersecretkey' # Necesario para `flash`

# --- Lógica de la Cámara ---
camera = cv2.VideoCapture(0)

def generate_frames():
    """Generador que captura frames de la cámara, los procesa y los devuelve como JPEGs."""
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Procesar el frame para reconocimiento facial
            frame = face_utils.process_frame_for_recognition(frame)
            
            # Codificar el frame en formato JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            # Devolver el frame como parte de una respuesta multipart
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# --- Rutas de la Aplicación Web ---

@app.route('/')
def index():
    """Página principal que muestra el streaming de video."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Ruta que proporciona el streaming de video."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/admin')
def admin():
    """Página de administración para añadir nuevas personas."""
    return render_template('admin.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/add_person', methods=['POST'])
def add_person():
    """Endpoint que maneja la subida de una nueva persona."""
    if 'photo' not in request.files or 'name' not in request.form:
        flash('Faltan partes del formulario')
        return redirect(url_for('admin'))
    
    file = request.files['photo']
    name = request.form['name']

    if file.filename == '' or name == '':
        flash('El nombre y el archivo no pueden estar vacíos')
        return redirect(url_for('admin'))

    if file and allowed_file(file.filename):
        # Usamos el nombre proporcionado para el archivo, asegurando que sea un nombre seguro
        filename = secure_filename(name + '.' + file.filename.rsplit('.', 1)[1].lower())
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Recargamos los rostros conocidos para incluir el nuevo
        face_utils.load_known_faces()
        
        flash(f'Persona "{name}" añadida correctamente.')
        return redirect(url_for('admin'))
    else:
        flash('Tipo de archivo no permitido')
        return redirect(url_for('admin'))

# --- Punto de Entrada ---
if __name__ == "__main__":
    # Cargar los rostros conocidos al iniciar la aplicación
    face_utils.load_known_faces()
    # Iniciar el servidor de Flask
    app.run(debug=True)

```

---

## Paso 5: Las Plantillas HTML (`templates/`)

### `index.html`
Esta es la página principal. Es muy simple: solo tiene un título y una etiqueta `<img>` cuyo `src` apunta a nuestra ruta de streaming.

```html
<!-- templates/index.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Reconocimiento Facial en Tiempo Real</title>
    <style>
        body { font-family: sans-serif; text-align: center; background-color: #f0f0f0; }
        h1 { color: #333; }
        #bg { border: 5px solid #333; border-radius: 10px; }
        nav a { margin: 0 15px; text-decoration: none; color: #007bff; font-size: 1.2em; }
    </style>
</head>
<body>
    <h1>Reconocimiento Facial en Tiempo Real</h1>
    <nav>
        <a href="{{ url_for('index') }}">Ver Stream</a>
        <a href="{{ url_for('admin') }}">Administrar Personas</a>
    </nav>
    <br>
    <img id="bg" src="{{ url_for('video_feed') }}">
</body>
</html>
```

### `admin.html`
Esta página contiene el formulario para añadir nuevas personas.

```html
<!-- templates/admin.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Administrar Personas</title>
    <style>
        body { font-family: sans-serif; text-align: center; background-color: #f0f0f0; }
        h1 { color: #333; }
        .container { background-color: white; padding: 20px; border-radius: 10px; display: inline-block; }
        nav a { margin: 0 15px; text-decoration: none; color: #007bff; font-size: 1.2em; }
        .messages { list-style: none; padding: 0; }
        .messages li { background-color: #d4edda; color: #155724; padding: 10px; margin-bottom: 10px; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>Administrar Personas</h1>
    <nav>
        <a href="{{ url_for('index') }}">Ver Stream</a>
        <a href="{{ url_for('admin') }}">Administrar Personas</a>
    </nav>

    <!-- Mostrar mensajes flash -->
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        <ul class=messages>
        {% for message in messages %}
          <li>{{ message }}</li>
        {% endfor %}
        </ul>
      {% endif %}
    {% endwith %}

    <div class="container">
        <h2>Añadir Nueva Persona</h2>
        <form action="{{ url_for('add_person') }}" method="post" enctype="multipart/form-data">
            <p>
                Nombre: <input type="text" name="name" required>
            </p>
            <p>
                Foto: <input type="file" name="photo" accept=".png,.jpg,.jpeg" required>
            </p>
            <p>
                <input type="submit" value="Añadir Persona">
            </p>
        </form>
    </div>

</body>
</html>
```

---

## Paso 6: ¡Ejecutar la Aplicación!

Una vez que todos los archivos estén en su lugar, simplemente ejecuta el siguiente comando en tu terminal desde el directorio raíz del proyecto:

```bash
python app.py
```

Luego, abre tu navegador web y ve a **`http://127.0.0.1:5000`**. Verás el video en tiempo real. Si vas a **`http://127.0.0.1:5000/admin`**, podrás añadir nuevas personas. ¡Cualquier persona que añadas será reconocida instantáneamente en la página principal!

¡Felicidades! Has convertido tu proyecto en una aplicación web de IA completa y funcional.

```