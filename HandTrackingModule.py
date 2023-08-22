import cv2
import mediapipe as mp
import time


class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # used for hands tracking
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.detectionCon)
        self.mpDraw = mp.solutions.drawing_utils


def FindHands(self, frame, draw=True):
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    self.results = self.hands.process(imgRGB)
    print(self.results.multi_hand_landmarks)  # it will print the position of hand in output

    # it shows connections, dots on all the fingures
    if self.results.multi_hand_landmarks:
        for handLM in self.results.multi_hand_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(frame, handLM, self.mpHands.HAND_CONNECTIONS)  # it draws the connection

    return frame


def positionFinder(self, image, handNo=0, draw=True):
    lmlist = []
    if self.results.multi_hand_landmarks:
        Hand = self.results.multi_hand_landmarks[handNo]
        for id, lm in enumerate(Hand.landmark):
            h, w, c = image.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmlist.append([id, cx, cy])
            if draw:
                cv2.circle(image, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

    return lmlist


def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while (True):
        _, frame = cap.read()
        frame = detector.FindHands(frame)
        lmList = detector.positionFinder(frame)

        # used to see FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(frame, str(int(fps)), (30, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Video", frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()  # releases the output when everything is done
    cv2.destroyAllWindows()  # it destroys all the windows output


if __name__ == "__main__":
    main()
