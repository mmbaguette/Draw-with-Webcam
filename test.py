import cv2

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
#cap.set(cv2.CAP_PROP_FPS, 60)

while cap.isOpened():
    success, image = cap.read()

    if not success:
        break
    
    cv2.waitKey(1)
    cv2.imshow("bruh", image)

cv2.destroyAllWindows()
cap.release()