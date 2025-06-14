import cv2
import mediapipe as mp
import pygame.midi
import time

# === Initialize MediaPipe Hands ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# === Initialize Pygame MIDI ===
pygame.midi.init()
player = pygame.midi.Output(0)  # Use 0 or appropriate device ID
player.set_instrument(0)        # 0 = Acoustic Grand Piano

# === Define Chords ===
chord_notes_map = {
    "C": [60, 64, 67],   # C major
    "D": [62, 66, 69],   # D major
    "G": [67, 71, 74],   # G major
    "A": [69, 73, 76],   # A major
}

# === Define Finger Pattern to Chord Mapping ===
# Pattern: (thumb, index, middle, ring, pinky)
chords = {
    (0, 1, 0, 0, 0): "C",
    (0, 1, 1, 0, 0): "D",
    (0, 1, 1, 1, 0): "G",
    (0, 1, 0, 0, 1): "A",
}

# === Define Play and Stop Functions ===
def play_chord(notes):
    for note in notes:
        player.note_on(note, 127)

def stop_chord(notes):
    for note in notes:
        player.note_off(note, 127)

# === Detect Which Fingers Are Up ===
def fingers_up(hand_landmarks):
    fingers = []

    # Thumb: Use x-axis
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers: Use y-axis
    tips = [8, 12, 16, 20]
    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return tuple(fingers)

# === Main Program ===
cap = cv2.VideoCapture(0)
last_chord = None
last_time = 0
cooldown = 2  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            current_fingers = fingers_up(hand_landmarks)

            if current_fingers in chords:
                chord_name = chords[current_fingers]
                chord_notes = chord_notes_map[chord_name]

                if chord_name != last_chord and (time.time() - last_time) > cooldown:
                    if last_chord:
                        stop_chord(chord_notes_map[last_chord])

                    play_chord(chord_notes)
                    last_chord = chord_name
                    last_time = time.time()

                cv2.putText(frame, f"Chord: {chord_name}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                # No matching gesture, stop previous chord
                if last_chord and (time.time() - last_time) > cooldown:
                    stop_chord(chord_notes_map[last_chord])
                    last_chord = None

    cv2.imshow("Finger Chord Player", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
player.close()
pygame.midi.quit()
