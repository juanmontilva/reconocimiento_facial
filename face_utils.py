import face_recognition
import numpy as np
import os
import cv2

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
