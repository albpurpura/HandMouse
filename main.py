import cv2
import mediapipe as mp

from gesture import recognize_gesture, mouse_move, scroll, right_click, left_click

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

prev_res = None
background = None
cap = cv2.VideoCapture(0)

while cap.isOpened():
    if prev_res is not None and len(prev_res) >= 100:
        prev_res = prev_res[-10:]
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # if background is None:
    #     background = image
    # image = image - background
    image_height, image_width, _ = image.shape
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False

    results = hands.process(image)
    if prev_res is None:
        prev_res = [results]
    elif results.multi_handedness is not None:
        action = recognize_gesture(results)
        if action is not None and len(prev_res) > 5:
            # print('ACTION={}'.format(action))
            prev_action = recognize_gesture(prev_res[-1])
            prev_prev_action = recognize_gesture(prev_res[-2])  # to prevent triple clicking
            if action == 'left_click' and prev_prev_action != 'left_click':
                left_click()
            elif action == 'right_click' and prev_action != 'right_click':
                right_click()
            elif action == 'mouse_move' and prev_action is not None:
                mouse_move(results, prev_res, image_height, image_width)
            elif action == 'scroll':
                scroll(results, prev_res[-5])
        prev_res.append(results)
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    resized_img = cv2.resize(image, (400, 200))

    cv2.imshow('MediaPipe Hands (ESC or q to quit)', resized_img)
    cv2.moveWindow('image', 400, 200)
    if cv2.waitKey(5) & 0xFF == 27:
        break
hands.close()
cap.release()
