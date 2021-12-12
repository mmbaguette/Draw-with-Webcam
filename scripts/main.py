print("Loading modules...")
import cv2
import mediapipe as mp
import numpy as np
import pyvirtualcam
from drawing_and_hands import fullscreen_image, process_hand_and_paint, draw_from_strokes, overlay_image

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic

CAP_FPS = 30
CAP_WIDTH, CAP_HEIGHT = 1280, 720 # 16:9 ratio
VIRTUAL_CAM = False

print("Opening webcam...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) # faster VideoCapture?
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)

def main():
    print("Loading MediaPipe holistic model...")
    current_colour = (0, 0, 255) # BGR
    current_thickness = 15
    left_strokes = [] # paint strokes for left hand
    right_strokes = []
    left_pinched_before = False # left hand was drawing before
    right_pinched_before = False

    with mp_holistic.Holistic(
        min_detection_confidence = 0.5,
        min_tracking_confidence=0.5) as holistic:
        
        if VIRTUAL_CAM:
            print("Starting virtual camera...")
            cam = pyvirtualcam.Camera(width=CAP_WIDTH, height=CAP_HEIGHT, fps=30)

        while cap.isOpened():
            success, image = cap.read()

            if not success:
                print("Ignoring empty camera frame.")
                continue
        
            image_to_display = np.zeros(image.shape, dtype=np.uint8) #image.copy()
            image_to_display[:] = (255,255,255)
            image_to_process = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # improve performance by optionally marking image as not writeable
            image_to_process.flags.writeable = False
            results = holistic.process(image_to_process)

            # draw pose
            mp_drawing.draw_landmarks(
                image_to_display,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles
                .get_default_pose_landmarks_style())

            # check the positions of the fingers of the left and right hand to see if they're drawing
            left_hand = results.left_hand_landmarks
            right_hand = results.right_hand_landmarks

            if left_hand:
                left_strokes, left_pinched_before = process_hand_and_paint(
                results.left_hand_landmarks, 
                image_to_display,
                left_strokes,
                left_pinched_before,
                current_colour,
                current_thickness)
            if right_hand:
                right_strokes, right_pinched_before = process_hand_and_paint(
                right_hand,
                image_to_display,
                right_strokes,
                right_pinched_before,
                current_colour,
                current_thickness)
        
            # using the strokes painted from each hand, draw on the image
            draw_from_strokes(image_to_display, left_strokes)
            draw_from_strokes(image_to_display, right_strokes)
            mini_cam = cv2.resize(
                image, 
                (round(image.shape[1] / 5), round(image.shape[0] / 5)), 
            )
            overlay_image(
                image_to_display, 
                mini_cam,
                0,
                image_to_display.shape[0] - mini_cam.shape[0]
            )
        
            # show image stream and handle windows
            if VIRTUAL_CAM:
                cam.send(image_to_display)
                cam.sleep_until_next_frame()
            else:
                image_to_display = fullscreen_image(image_to_display)
                cv2.namedWindow("MediaPipe Hands", cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty("MediaPipe Hands", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow('MediaPipe Hands', image_to_display)
            
            if cv2.waitKey(1) == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
  main()