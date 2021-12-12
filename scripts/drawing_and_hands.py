import mediapipe as mp
import numpy as np
import pyautogui
from not_my_functions import distance_2D, image_resize, midpoint
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def overlay_image(l_img, s_img, x_offset, y_offset):
	l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img

def draw_from_strokes(image_to_display, strokes, erasing=[]):
	for stroke in strokes:
		previous_point_drawn = None
	
		for line in stroke:
			point2 = line[0]
			point1 = point2 if previous_point_drawn is None else previous_point_drawn
			color = line[1]
			thickness = line[2]

			cv2.line(image_to_display, point1, point2, color, thickness=thickness, lineType=cv2.LINE_AA)
			previous_point_drawn = point2
	
	if erasing:
		previous_erase_drawn = None

		for erase in erasing:
			point2 = erase[0]
			point1 = point2 if previous_erase_drawn is None else previous_erase_drawn
			color = (0,0,0,0)
			thickness = erase[1]

			cv2.line(image_to_display, point1, point2, color, thickness=thickness, lineType=cv2.LINE_AA)
			previous_erase_drawn = point2

# resizes image, keeps aspect ratio
def fullscreen_image(image_to_display):
	image_to_display = cv2.flip(image_to_display, 1)
	image_to_display = image_resize(image_to_display, height=pyautogui.size()[1])
	print(image_to_display.shape)
	# make sure the size of image_resize is perfect
	cv2.resize(
		image_to_display, 
		(image_to_display.shape[1], pyautogui.size()[1]), 
		interpolation=cv2.INTER_LINEAR)
	print(image_to_display.shape)
	x_offset = int(pyautogui.size()[0] / 2 - image_to_display.shape[1] / 2)
	y_offset = 0
	
	fullscreen_img_back = np.zeros(
    	(pyautogui.size()[1], pyautogui.size()[0], 3), 
		dtype=np.uint8)
	fullscreen_img_back[:] = (255,0,0)
	fullscreen_img_back[y_offset:y_offset+image_to_display.shape[0], 
    	x_offset:x_offset+image_to_display.shape[1]] = image_to_display
	return fullscreen_img_back
  
def process_hand_and_paint(
	hand_landmarks, 
	image_to_display, 
	strokes, 
	pinched_before,
	current_colour,
	current_thickness):
	img_h, img_w = image_to_display.shape[:2]
	mp_drawing.draw_landmarks(
		image_to_display,
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

	pinch_fingers_distance = round(distance_2D(index_finger_tip_coords, middle_finger_tip_coords),2)

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