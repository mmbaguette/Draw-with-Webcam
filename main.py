print("Loading modules...")
import cv2
import mediapipe as mp
import pyvirtualcam
from drawing_and_hands import *

mp_holistic = mp.solutions.holistic

CAP_FPS = 30
CAP_WIDTH, CAP_HEIGHT = 640, 480
VIRTUAL_CAM = False

print("Opening web cam...")
cap = cv2.VideoCapture(0)

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
      
      imageToDisplay = image.copy()
      imageToProcess = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      # improve performance by optionally marking image as not writeable
      imageToProcess.flags.writeable = False
      results = holistic.process(imageToProcess)

      # draw pose
      mp_drawing.draw_landmarks(
        imageToDisplay,
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
          imageToDisplay,
          left_strokes,
          left_pinched_before,
          current_colour,
          current_thickness)
      if right_hand:
        right_strokes, right_pinched_before = process_hand_and_paint(
          right_hand,
          imageToDisplay,
          right_strokes,
          right_pinched_before,
          current_colour,
          current_thickness)
      
      # using the strokes painted from each hand, draw on the image
      imageToDisplay = draw_from_strokes(imageToDisplay, left_strokes)
      imageToDisplay = draw_from_strokes(imageToDisplay, right_strokes)
      
      # show image stream and handle windows
      if VIRTUAL_CAM:
        cam.send(imageToDisplay)
        cam.sleep_until_next_frame()
      else:
        imageToDisplay = fullscreen_image(imageToDisplay)
        cv2.namedWindow("MediaPipe Hands", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("MediaPipe Hands", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('MediaPipe Hands', imageToDisplay)
      
      if cv2.waitKey(1) == 27:
        break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
  main()