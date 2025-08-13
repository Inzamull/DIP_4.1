import cv2
import numpy as np
import matplotlib.pyplot as plt


def create_hist_image(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_img = np.full((200, 256), 255, dtype=np.uint8) 

    cv2.normalize(hist, hist, 0, 200, cv2.NORM_MINMAX)
    for x, y in enumerate(hist):
        cv2.line(hist_img, (x, 200), (x, 200 - int(y)), 0)
    return hist_img

def main():
    cam = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            cv2.imshow('Waiting for Face...', gray)
        else:
            (x, y, w, h) = faces[0] 
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (256, 256))

         
            hist_image = create_hist_image(face_resized)

         
            combined = np.vstack((face_resized, hist_image))

            cv2.imshow("Grayscale Face with Histogram", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
