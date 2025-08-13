import cv2
import numpy as np
from matplotlib import pyplot as plt

# -------- Linear Mapping (Contrast Stretching) --------
def contrast_stretching(img):
    a, b = 0, 255  # Output intensity range
    c, d = np.min(img), np.max(img)  # Input intensity range
    stretched = ((img - c) * ((b - a) / (d - c)) + a).astype(np.uint8)
    return stretched

# -------- Plot histogram --------
def plot_histogram(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_img = np.zeros((256, 256), dtype=np.uint8)
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    for x, y in enumerate(hist):
        cv2.line(hist_img, (x, 255), (x, 255 - int(y)), 255)
    return hist_img

# -------- Main --------
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect face
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        cv2.imshow("Waiting for face...", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]

        # Linear: Contrast Stretching
        linear_img = contrast_stretching(face_img)
        linear_hist = plot_histogram(linear_img)

        # Nonlinear: Histogram Equalization
        nonlinear_img = cv2.equalizeHist(face_img)
        nonlinear_hist = plot_histogram(nonlinear_img)

        # Resize for display
        linear_img = cv2.resize(linear_img, (200, 200))
        nonlinear_img = cv2.resize(nonlinear_img, (200, 200))
        linear_hist = cv2.resize(linear_hist, (200, 200))
        nonlinear_hist = cv2.resize(nonlinear_hist, (200, 200))

        # Stack for display
        top = np.hstack((linear_img, nonlinear_img))
        bottom = np.hstack((linear_hist, nonlinear_hist))
        combined = np.vstack((top, bottom))

        cv2.imshow("Face with Linear + Nonlinear Mapping and Histograms", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
