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
    app.run(debug=True, port=5001)