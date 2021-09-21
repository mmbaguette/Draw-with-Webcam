import cv2
import mediapipe as mp
import math
import statistics
import numpy as np
import pyvirtualcam
import pyautogui
import time
#import sys

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def distance2D(point1: tuple, point2: tuple):
  return math.hypot(point2[0] - point1[0], point2[1] - point1[1])

def midpoint(point1: tuple, point2: tuple):
  x1, y1 = point1
  x2, y2 = point2
  return ((x1 + x2)/2, (y1 + y2)/2)

def main():
  currentColour = (0, 0, 255)
  currentThickness = 15
  strokes = []
  pinchingBefore = False
  cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

  #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
  #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
  #cap.set(cv2.CAP_PROP_FPS, 60)

  with mp_hands.Hands(
      min_detection_confidence=0.5,
      min_tracking_confidence=0.6,
      max_num_hands=1) as hands:

    with pyvirtualcam.Camera(width=1920, height=1080, fps=30) as cam:
      while cap.isOpened():
        last_time = time.time()
        success, image = cap.read()
        imageToDisplay = cv2.flip(image, 1)
        img_h, img_w = imageToDisplay.shape[:2]

        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        imageToProcess = cv2.cvtColor(imageToDisplay, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        imageToProcess.flags.writeable = False
        results = hands.process(imageToProcess)
        
        for stroke in strokes:
          previousPointDrawn = None

          for line in stroke:
            point2 = line[0]
            point1 = point2 if previousPointDrawn is None else previousPointDrawn
            color = line[1]
            thickness = line[2]
            cv2.line(imageToDisplay, point1, point2, color, thickness=thickness, lineType=cv2.LINE_AA)
            previousPointDrawn = point2

        if results.multi_hand_landmarks:
          for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
              imageToDisplay,
              hand_landmarks,
              mp_hands.HAND_CONNECTIONS,
              mp_drawing_styles.get_default_hand_landmarks_style(),
              mp_drawing_styles.get_default_hand_connections_style())

            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_finger_tip_coords = (index_finger_tip.x, index_finger_tip.y)
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_tip_coords = (thumb_tip.x, thumb_tip.y)

            index_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            index_finger_mcp_coords = (index_finger_mcp.x, index_finger_mcp.y)
            middle_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            middle_finger_mcp_coords = (middle_finger_mcp.x, middle_finger_mcp.y)
            ring_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
            ring_finger_mcp_coords = (ring_finger_mcp.x, ring_finger_mcp.y)
            pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
            pinky_mcp_coords = (pinky_mcp.x, pinky_mcp.y)
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            wrist_coords = (wrist.x, wrist.y)

            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            middle_finger_tip_coords = (middle_finger_tip.x, middle_finger_tip.y)
            ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            ring_finger_tip_coords = (ring_finger_tip.x, ring_finger_tip.y)
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            pinky_tip_coords = (pinky_tip.x, pinky_tip.y)

            pinch_fingers_distance = round(distance2D(index_finger_tip_coords, thumb_tip_coords),2)
            average_palm_fingers_to_wrist_distance = round(statistics.mean([
              distance2D(index_finger_mcp_coords, wrist_coords),
              distance2D(middle_finger_mcp_coords, wrist_coords),
              distance2D(ring_finger_mcp_coords, wrist_coords),
              distance2D(pinky_mcp_coords, wrist_coords),
            ]),2)

            average_finger_tips_to_mpc = round(statistics.mean([
              distance2D(middle_finger_mcp_coords, middle_finger_tip_coords),
              distance2D(ring_finger_mcp_coords, ring_finger_tip_coords),
              distance2D(pinky_mcp_coords, pinky_tip_coords),
            ]),2)
            
            '''
            if pinch_fingers_distance * (average_palm_fingers_to_wrist_distance * 2) < 0.05:
              sys.stdout.write(f"\rWrist distance: {average_palm_fingers_to_wrist_distance} | Pinch distance: {pinch_fingers_distance}")
              sys.stdout.flush()
            else:
              sys.stdout.write(f"Not pinching!")
              sys.stdout.flush()
            '''
            if pinch_fingers_distance < 0.07:            
              if not pinchingBefore:
                strokes.append([]) # if you weren't pinching before, create a new stoke to draw in
              pinchingBefore = True

              if strokes != []: # make the list is not empty (weird bug)
                processor_midpoint = midpoint(index_finger_tip_coords, thumb_tip_coords)
                real_midpoint = (int(processor_midpoint[0] * img_w), int(processor_midpoint[1] * img_h))
                strokes[len(strokes) - 1].append([
                  real_midpoint,
                  currentColour,
                  currentThickness
                ])
            else:
              if average_finger_tips_to_mpc < 0.08:
                strokes = []
                pass
              pinchingBefore = False

        '''
        cam.send(cv2.cvtColor(imageToDisplay, cv2.COLOR_RGB2BGR))
        cam.sleep_until_next_frame()
        '''
        fullscreenImgBack = np.zeros((pyautogui.size()[1], pyautogui.size()[0], 3), dtype=np.uint8)
        fullscreenImgBack[:] = (0,0,0)

        imageToDisplay = image_resize(imageToDisplay, height=pyautogui.size()[1])
        #make sure the size is perfect
        imageToDisplay = cv2.resize(imageToDisplay, (imageToDisplay.shape[1], pyautogui.size()[1]), interpolation=cv2.INTER_LINEAR)
        x_offset = int(pyautogui.size()[0] / 2 - imageToDisplay.shape[1] / 2)
        y_offset = 0
        fullscreenImgBack[y_offset:y_offset+imageToDisplay.shape[0], x_offset:x_offset+imageToDisplay.shape[1]] = imageToDisplay
        
        cv2.namedWindow("MediaPipe Hands", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("MediaPipe Hands",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        cv2.imshow('MediaPipe Hands', fullscreenImgBack)

        k = cv2.waitKey(5)
        if k == 27:
          break
        elif k == 99:
          strokes = []

        print(f"Image height: {img_h} Image width: {img_w}")
        print(f"FPS: {1 / (time.time() - last_time)}")
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
  main()