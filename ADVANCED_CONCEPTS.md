
# Guía de Conceptos Avanzados: Llevando tu Proyecto al Siguiente Nivel

¡Hola! Este documento es una guía práctica para implementar las ideas avanzadas que discutimos. Tu proyecto actual es una base excelente, y estos ejemplos te mostrarán cómo construir sobre ella para añadir funcionalidades de vanguardia.

## Estructura del Documento
1.  **Reconocimiento Facial:** Identificar *quién* es la persona.
2.  **Análisis de Microexpresiones:** Detectar emociones ocultas analizando el movimiento en el tiempo.
3.  **Predicción de Engagement:** Inferir estados complejos como la concentración o la confusión.
4.  **Animación Facial (Avatar 3D):** Animar un personaje digital con tus propias expresiones.

---

## 1. Reconocimiento Facial

Esta funcionalidad se integra perfectamente con tu bucle de `OpenCV` existente. Usaremos la librería `face_recognition`, que es muy fácil de usar.

**Paso 1: Instalar la librería**
```bash
pip install face_recognition
```

**Paso 2: Código de Ejemplo**

Este código muestra cómo reconocer caras en un video en tiempo real. Deberás tener imágenes de las personas que quieres reconocer en una carpeta.

```python
# facial_recognition_example.py
import cv2
import face_recognition
import numpy as np
import os

# --- 1. Preparación ---
# Carga las imágenes de las personas que quieres reconocer y codifica sus rostros.

def load_known_faces(known_faces_dir='known_faces'):
    """
    Carga imágenes y codifica los rostros desde un directorio.
    El nombre del archivo de imagen (sin extensión) se usará como el nombre de la persona.
    """
    known_face_encodings = []
    known_face_names = []

    if not os.path.exists(known_faces_dir):
        print(f"ADVERTENCIA: El directorio '{known_faces_dir}' no existe. No se cargarán caras conocidas.")
        return known_face_encodings, known_face_names

    for filename in os.listdir(known_faces_dir):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(known_faces_dir, filename)
            name = os.path.splitext(filename)[0]
            
            try:
                image = face_recognition.load_image_file(path)
                # Codifica el primer rostro que encuentre en la imagen
                encoding = face_recognition.face_encodings(image)[0]
                
                known_face_encodings.append(encoding)
                known_face_names.append(name)
                print(f"Rostro de '{name}' cargado.")
            except IndexError:
                print(f"ADVERTENCIA: No se encontró ningún rostro en {filename}. Saltando.")
            except Exception as e:
                print(f"Error cargando {filename}: {e}")

    return known_face_encodings, known_face_names

# --- 2. Ejecución en Tiempo Real ---

# Crea una carpeta llamada 'known_faces' y pon imágenes .jpg de personas allí.
# Por ejemplo: 'juan.jpg', 'ana.png'.
known_face_encodings, known_face_names = load_known_faces()

# Inicializa la cámara
video_capture = cv2.VideoCapture(0)

while True:
    # Captura un solo frame de video
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convierte la imagen de BGR (OpenCV) a RGB (face_recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Encuentra todas las caras y sus codificaciones en el frame actual
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Itera sobre cada cara encontrada en el frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compara la cara actual con todas las caras conocidas
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Desconocido"

        # Usa la distancia facial para encontrar la mejor coincidencia
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        # Dibuja un rectángulo alrededor de la cara
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Dibuja una etiqueta con el nombre debajo de la cara
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Muestra la imagen resultante
    cv2.imshow('Video', frame)

    # Presiona 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera la cámara y cierra las ventanas
video_capture.release()
cv2.destroyAllWindows()
```

---

## 2. Análisis de Microexpresiones (Conceptual)

Esto requiere un modelo que entienda secuencias, como un LSTM. El siguiente código es **conceptual** y se enfoca en la **preparación de los datos**. El entrenamiento del modelo es un proyecto de Machine Learning en sí mismo.

```python
# microexpression_concept.py
import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Suponemos que tienes un modelo LSTM pre-entrenado
# model = load_lstm_model('path/to/your/lstm_model.h5')

# --- 1. Configuración ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Usamos una 'deque' para guardar una secuencia de landmarks de longitud fija
SEQUENCE_LENGTH = 15  # Analizar los últimos 15 frames
landmarks_sequence = deque(maxlen=SEQUENCE_LENGTH)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Procesamiento con MediaPipe
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            
            # --- 2. Recopilación de Datos Secuenciales ---
            # Normaliza y aplana los landmarks del frame actual
            current_landmarks = []
            for lm in face_landmarks.landmark:
                current_landmarks.extend([lm.x, lm.y, lm.z])
            
            # Añade los landmarks del frame actual a nuestra secuencia
            landmarks_sequence.append(current_landmarks)

            # --- 3. Predicción (cuando la secuencia está completa) ---
            if len(landmarks_sequence) == SEQUENCE_LENGTH:
                # Prepara los datos para el modelo LSTM
                # El modelo espera una forma como (1, SEQUENCE_LENGTH, num_features)
                input_data = np.expand_dims(np.array(landmarks_sequence), axis=0)

                # --- Aquí es donde llamarías a tu modelo ---
                # prediction = model.predict(input_data)
                # emotion = np.argmax(prediction) 
                # emotion_label = ["Neutral", "Micro-Sorpresa", "Micro-Frustración"][emotion]
                
                # --- Simulación de la salida del modelo ---
                emotion_label = "Analizando..."
                if np.mean(input_data) > 0.5: # Lógica de ejemplo
                    emotion_label = "Micro-expresión detectada!"

                cv2.putText(image, emotion_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    cv2.imshow('Microexpression Analysis', image)
    if cv2.waitKey(5) & 0xFF == 27: # ESC para salir
        break

cap.release()
```

---

## 3. Predicción de Engagement (Conceptual)

Esta idea combina diferentes métricas faciales para predecir un estado complejo. Aquí, calcularemos el "Eye Aspect Ratio" (EAR) para detectar parpadeos, una señal clave del engagement.

```python
# engagement_concept.py
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist

# --- 1. Constantes y Funciones Auxiliares ---
# Índices de los landmarks para los ojos según MediaPipe
LEFT_EYE_IDXS = [362, 382, 381, 380, 373, 374, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_IDXS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

def calculate_ear(eye_landmarks):
    # Calcula la distancia euclidiana entre los puntos verticales del ojo
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    # Calcula la distancia euclidiana entre los puntos horizontales del ojo
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    # Calcula el Eye Aspect Ratio (EAR)
    ear = (A + B) / (2.0 * C)
    return ear

# --- 2. Bucle Principal ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

cap = cv2.VideoCapture(0)

# Variables para el análisis de parpadeo
EAR_THRESHOLD = 0.20
CONSECUTIVE_FRAMES_THRESHOLD = 3
frame_counter = 0
blink_counter = 0

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Extraer coordenadas de los ojos
        left_eye_coords = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in LEFT_EYE_IDXS])
        right_eye_coords = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in RIGHT_EYE_IDXS])

        # Calcular EAR para cada ojo
        left_ear = calculate_ear(left_eye_coords)
        right_ear = calculate_ear(right_eye_coords)
        avg_ear = (left_ear + right_ear) / 2.0

        # Detección de parpadeo
        if avg_ear < EAR_THRESHOLD:
            frame_counter += 1
        else:
            if frame_counter >= CONSECUTIVE_FRAMES_THRESHOLD:
                blink_counter += 1
            frame_counter = 0
        
        # --- 3. Inferencia del Modelo de Engagement ---
        # Aquí combinarías varias métricas (EAR, pose de la cabeza, etc.)
        # y las pasarías a un modelo pre-entrenado (ej. XGBoost, RandomForest).
        
        # feature_vector = [avg_ear, head_pose_x, head_pose_y, mouth_opening_ratio]
        # engagement_prediction = engagement_model.predict([feature_vector])
        # engagement_label = ["Engaged", "Not Engaged"][engagement_prediction[0]]

        # --- Simulación de la salida ---
        engagement_label = "Engaged" if avg_ear > 0.15 else "Not Engaged"
        
        cv2.putText(image, f"Blinks: {blink_counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(image, f"EAR: {avg_ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(image, f"Status: {engagement_label}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


    cv2.imshow('Engagement Analysis', image)
    if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
```

---

## 4. Animación Facial (Avatar 3D)

Esta es la idea más compleja y requiere dos partes: un backend en Python para procesar la cara y un frontend en JavaScript para renderizar el avatar.

### Parte A: Backend (Python con WebSockets)

Este script envía los datos de la malla facial a través de un WebSocket.

**Paso 1: Instalar la librería**
```bash
pip install websockets
```

**Paso 2: Código del Servidor**
```python
# animation_backend.py
import asyncio
import websockets
import cv2
import mediapipe as mp
import json

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

async def video_stream_handler(websocket, path):
    print("Cliente conectado.")
    cap = cv2.VideoCapture(0)
    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # Procesar la imagen para obtener los landmarks
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)

            if results.multi_face_landmarks:
                # Extraer los 478 landmarks 3D
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Preparar los datos para enviar como JSON
                # Enviamos solo una lista de listas [x, y, z]
                landmarks_data = [[lm.x, lm.y, lm.z] for lm in landmarks]
                
                # Enviar los datos al cliente a través del WebSocket
                await websocket.send(json.dumps(landmarks_data))
            
            # Pequeña pausa para no sobrecargar el bucle
            await asyncio.sleep(0.01)

    except websockets.exceptions.ConnectionClosed:
        print("Cliente desconectado.")
    finally:
        cap.release()

async def main():
    # Inicia el servidor de WebSockets en localhost, puerto 8765
    async with websockets.serve(video_stream_handler, "localhost", 8765):
        print("Servidor WebSocket iniciado en ws://localhost:8765")
        await asyncio.Future()  # Ejecutar indefinidamente

if __name__ == "__main__":
    asyncio.run(main())
```

### Parte B: Frontend (HTML y JavaScript con Three.js)

Crea un archivo `avatar.html` y pega este código. Necesitarás descargar la librería Three.js (`three.min.js`) y ponerla en la misma carpeta, o puedes usar un CDN.

```html
<!-- avatar.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Avatar 3D en Tiempo Real</title>
    <style>
        body { margin: 0; }
        canvas { display: block; }
    </style>
</head>
<body>
    <!-- Usando un CDN para Three.js -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        // --- 1. Configuración de la Escena 3D ---
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer();
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);

        // Añadir una luz simple
        const light = new THREE.PointLight(0xffffff, 1, 100);
        light.position.set(0, 0, 10);
        scene.add(light);
        scene.add(new THREE.AmbientLight(0x404040));

        camera.position.z = 5;

        // --- 2. Creación del "Avatar" (una esfera con puntos) ---
        // En un proyecto real, cargarías un modelo 3D (GLB, FBX).
        // Aquí, creamos una esfera y añadimos puntos para representar los landmarks.
        const avatarGroup = new THREE.Group();
        const sphereGeom = new THREE.SphereGeometry(2, 32, 16);
        const sphereMat = new THREE.MeshPhongMaterial({ color: 0x4a86e8, transparent: true, opacity: 0.3 });
        const sphere = new THREE.Mesh(sphereGeom, sphereMat);
        avatarGroup.add(sphere);

        // Creamos puntos visuales para los landmarks
        const pointsMaterial = new THREE.PointsMaterial({ color: 0xffffff, size: 0.05 });
        const pointsGeometry = new THREE.BufferGeometry();
        const points = new THREE.Points(pointsGeometry, pointsMaterial);
        avatarGroup.add(points);
        
        scene.add(avatarGroup);
        avatarGroup.rotation.y = Math.PI; // Girar para que nos mire

        // --- 3. Conexión con el Backend (WebSocket) ---
        const socket = new WebSocket('ws://localhost:8765');

        socket.onopen = function(event) {
            console.log("Conectado al servidor WebSocket.");
        };

        socket.onmessage = function(event) {
            const landmarks = JSON.parse(event.data);
            
            // --- 4. Actualizar la posición de los puntos del avatar ---
            const positions = [];
            for (let i = 0; i < landmarks.length; i++) {
                // MediaPipe da coordenadas normalizadas. Las escalamos y centramos.
                const x = (landmarks[i][0] - 0.5) * 5;
                const y = -(landmarks[i][1] - 0.5) * 5;
                const z = -landmarks[i][2] * 5;
                positions.push(x, y, z);
            }
            
            // Actualiza la geometría de los puntos con las nuevas posiciones
            pointsGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
            pointsGeometry.attributes.position.needsUpdate = true;
        };

        socket.onclose = function(event) {
            console.log("Desconectado del servidor WebSocket.");
        };

        // --- 5. Bucle de Animación ---
        function animate() {
            requestAnimationFrame(animate);
            // Puedes añadir rotaciones u otras animaciones aquí
            // avatarGroup.rotation.y += 0.005;
            renderer.render(scene, camera);
        }
        animate();

    </script>
</body>
</html>
```

### Cómo ejecutar el ejemplo de animación:
1.  Ejecuta el backend: `python animation_backend.py`
2.  Abre el archivo `avatar.html` en tu navegador web.
3.  Deberías ver una esfera azul semitransparente con puntos blancos que imitan los movimientos de tu cara en tiempo real.

Espero que esta guía te sea de gran utilidad. ¡Tienes un proyecto con un potencial enorme!
