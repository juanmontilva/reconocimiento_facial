import cv2


class Camera:
    def __init__(self, index: int, width: int, height: int):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise Exception("Error: Could not open camera.")
        self.cap.set(3, width)
        self.cap.set(4, height)

    def read(self):
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        self.cap.release()
