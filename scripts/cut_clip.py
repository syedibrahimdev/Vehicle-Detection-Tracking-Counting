import sys

import cv2

src, dst, start, duration = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])

cap = cv2.VideoCapture(src)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))

out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
for _ in range(int(duration * fps)):
    ok, frame = cap.read()
    if not ok:
        break
    out.write(frame)

out.release()
cap.release()
print("saved", dst)