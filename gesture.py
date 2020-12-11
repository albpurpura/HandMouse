import numpy as np
import pyautogui

THR = 0.08
WRIST_THR = 0.2


def get_last_valid_reading(prev_results):
    for i in range(1, len(prev_results)):
        prev_landmarks = get_hand_landmarks(prev_results[-i])
        if prev_landmarks is not None:
            return prev_landmarks


def left_click():
    pyautogui.click(button='left')
    print('left clicked')


def right_click():
    pyautogui.click(button='right')
    print('right clicked')


def mouse_move(results, prev_results, image_height, image_width):
    curr_landmarks = get_hand_landmarks(results)
    prev_landmarks = get_last_valid_reading(prev_results)

    if curr_landmarks is None or prev_landmarks is None:
        return
    curr_x_coords = np.array([curr_landmarks.landmark[i].x for i in range(len(curr_landmarks.landmark))])[8]
    curr_y_coords = np.array([curr_landmarks.landmark[i].y for i in range(len(curr_landmarks.landmark))])[8]

    prev_x_coords = np.array([prev_landmarks.landmark[i].x for i in range(len(prev_landmarks.landmark))])[8]
    prev_y_coords = np.array([prev_landmarks.landmark[i].y for i in range(len(prev_landmarks.landmark))])[8]
    diff = (curr_x_coords - prev_x_coords, curr_y_coords - prev_y_coords)
    # print(diff)
    pyautogui.moveRel(int(diff[0] * image_width * 3), int(diff[1] * image_height * 3), duration=0.1)
    print('mouse moved')


def scroll(results, prev_results):
    curr_landmarks = get_hand_landmarks(results)
    prev_landmarks = get_hand_landmarks(prev_results)

    if curr_landmarks is None or prev_landmarks is None:
        return

    curr_y_coords = np.array([curr_landmarks.landmark[i].y for i in range(len(curr_landmarks.landmark))])[8]
    prev_y_coords = np.array([prev_landmarks.landmark[i].y for i in range(len(prev_landmarks.landmark))])[8]
    diff = curr_y_coords - prev_y_coords
    # print(diff)
    if np.absolute(diff) >= 0.01:
        pyautogui.scroll(2 * -np.sign(diff))
        print('scrolled')


def compute_distance_matrix(hand_landmarks):
    x_coords = np.array([hand_landmarks.landmark[i].x for i in range(len(hand_landmarks.landmark))])
    y_coords = np.array([hand_landmarks.landmark[i].y for i in range(len(hand_landmarks.landmark))])
    z_coords = np.array([hand_landmarks.landmark[i].z for i in range(len(hand_landmarks.landmark))])
    dm = np.zeros(shape=(len(z_coords), len(z_coords)))
    for i in range(len(z_coords)):
        for j in range(len(z_coords)):
            dm[i, j] = np.sqrt(np.square(x_coords[i] - x_coords[j])
                               + np.square(y_coords[i] - y_coords[j])
                               + np.square(z_coords[i] - z_coords[j]))

    return dm


def get_hand_landmarks(results):
    if results.multi_handedness is None:
        return None
    multi_hand_landmarks = results.multi_hand_landmarks
    multi_handedness = results.multi_handedness

    hands_labels = [item.classification[0].label for item in multi_handedness if item.classification[0].label]

    if len(hands_labels) == 1:
        hands_index = 0
    else:
        if 'Right' in hands_labels and len(set(hands_labels)) == 2:
            hands_index = hands_labels.index('Right')
        else:
            return None
    hand_landmarks = multi_hand_landmarks[hands_index]
    return hand_landmarks


def recognize_gesture(results):
    hand_landmarks = get_hand_landmarks(results)
    if hand_landmarks is None:
        return None

    dist_m = compute_distance_matrix(hand_landmarks)

    action = None

    thumb_idx_dist = dist_m[4, 8]
    mid_thumb_dist = dist_m[4, 12]
    mid_idx_dist = dist_m[8, 12]
    base_idx_thumb_dist = dist_m[4, 5]
    wrist_idx_dist = dist_m[0, 8]
    writst_mid_dist = dist_m[0, 12]
    wrist_ring_dist = dist_m[0, 16]
    wrist_pinky_dist = dist_m[0, 20]

    if thumb_idx_dist > THR and mid_idx_dist > THR and base_idx_thumb_dist <= THR \
            and wrist_ring_dist <= WRIST_THR and wrist_pinky_dist <= WRIST_THR and wrist_idx_dist > WRIST_THR \
            and writst_mid_dist <= WRIST_THR:
        # move cursor
        action = 'mouse_move'
    elif thumb_idx_dist <= THR and mid_idx_dist > THR and wrist_ring_dist > WRIST_THR and writst_mid_dist > WRIST_THR and wrist_pinky_dist > WRIST_THR:
        # left click
        action = 'left_click'
    elif thumb_idx_dist > THR and base_idx_thumb_dist > THR and mid_thumb_dist <= THR and wrist_ring_dist > WRIST_THR \
            and wrist_pinky_dist > WRIST_THR:
        # right click
        action = 'right_click'
    elif thumb_idx_dist > THR and mid_thumb_dist > THR and mid_idx_dist <= THR and base_idx_thumb_dist <= THR \
            and wrist_pinky_dist <= WRIST_THR and wrist_idx_dist > WRIST_THR \
            and writst_mid_dist > WRIST_THR: # and wrist_ring_dist <= WRIST_THR
        # move cursor
        action = 'scroll'
    else:
        # print(str(dist_m))
        action = None

    return action


def compute_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
