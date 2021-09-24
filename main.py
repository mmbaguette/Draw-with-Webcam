print("Loading modules...")
import cv2
import numpy as np
import mediapipe as mp
import pyvirtualcam
import pyautogui
import statistics
from not_my_functions import *

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic

CAP_FPS = 30
CAP_WIDTH, CAP_HEIGHT = 640, 480
VIRTUAL_CAM = False

print("Opening web cam...")
cap = cv2.VideoCapture(0)

def draw_from_strokes(imageToDisplay, strokes):
  for stroke in strokes:
    previousPointDrawn = None

    for line in stroke:
      point2 = line[0]
      point1 = point2 if previousPointDrawn is None else previousPointDrawn
      color = line[1]
      thickness = line[2]
      cv2.line(imageToDisplay, point1, point2, color, thickness=thickness, lineType=cv2.LINE_AA)
      previousPointDrawn = point2
  return imageToDisplay

# resizes image, keeps aspect ratio
def fullscreen_image(imageToDisplay):
  imageToDisplay = cv2.flip(imageToDisplay, 1)
  imageToDisplay = image_resize(imageToDisplay, height=pyautogui.size()[1])
  
  # make sure the size of image_resize is perfect
  imageToDisplay = cv2.resize(
    imageToDisplay, 
    (imageToDisplay.shape[1], pyautogui.size()[1]), 
    interpolation=cv2.INTER_LINEAR)
    
  x_offset = int(pyautogui.size()[0] / 2 - imageToDisplay.shape[1] / 2)
  y_offset = 0
  fullscreenImgBack = np.zeros(
    (pyautogui.size()[1], pyautogui.size()[0], 3), 
    dtype=np.uint8)
  fullscreenImgBack[:] = (0,0,0)
  fullscreenImgBack[y_offset:y_offset+imageToDisplay.shape[0], 
    x_offset:x_offset+imageToDisplay.shape[1]] = imageToDisplay
  
  return fullscreenImgBack

def process_hand_and_paint(
  hand_landmarks, 
  imageToDisplay, 
  strokes, 
  pinched_before,
  current_colour,
  current_thickness):
  img_h, img_w = imageToDisplay.shape[:2]
  mp_drawing.draw_landmarks(
    imageToDisplay,
    hand_landmarks,
    mp_hands.HAND_CONNECTIONS,
    mp_drawing_styles.get_default_hand_landmarks_style(),
    mp_drawing_styles.get_default_hand_connections_style())

  index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
  index_finger_tip_coords = (index_finger_tip.x, index_finger_tip.y)
  index_finger_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]

  middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
  middle_finger_tip_coords = (middle_finger_tip.x, middle_finger_tip.y)
  middle_finger_dip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]

  pinch_fingers_distance = round(distance2D(index_finger_tip_coords, middle_finger_tip_coords),2)

  if pinch_fingers_distance < 0.06 and index_finger_tip.y < index_finger_dip.y and middle_finger_tip.y < middle_finger_dip.y:            
    if not pinched_before:
      strokes.append([]) # if you weren't pinching before, create a new stoke to draw in
    
    pinched_before = True
    processor_midpoint = midpoint(index_finger_tip_coords, middle_finger_tip_coords)
    real_midpoint = (int(processor_midpoint[0] * img_w), int(processor_midpoint[1] * img_h))
    strokes[len(strokes) - 1].append([
      real_midpoint,
      current_colour,
      current_thickness
    ])
  else:
    pinched_before = False
  return strokes, pinched_before

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