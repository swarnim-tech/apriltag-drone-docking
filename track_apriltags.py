import math
import cv2
import numpy as np

TAG_FAMILY = cv2.aruco.DICT_APRILTAG_36h11
ALIGNMENT_THRESHOLD_PX = 35  # Tolerance radius for "ON TARGET" lock


def get_camera_index(max_test=4):
  """Finds available camera indices."""
  available = []
  for i in range(max_test):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
      ret, _ = cap.read()
      if ret:
        available.append(i)
      cap.release()
  return available


def draw_drone_hud(frame, target_center, tag_corners, tag_id):
  """Draws a drone targeting reticle, direction vector, and telemetry readouts."""
  h, w = frame.shape[:2]
  cam_cx, cam_cy = w // 2, h // 2

  # Colors (BGR)
  COLOR_NEUTRAL = (200, 200, 200)
  COLOR_CYAN = (255, 255, 0)
  COLOR_GREEN = (0, 255, 0)
  COLOR_RED = (0, 0, 255)

  # Draw static drone center crosshair
  cross_size = 20
  cv2.line(
      frame,
      (cam_cx - cross_size, cam_cy),
      (cam_cx + cross_size, cam_cy),
      COLOR_NEUTRAL,
      1,
  )
  cv2.line(
      frame,
      (cam_cx, cam_cy - cross_size),
      (cam_cx, cam_cy + cross_size),
      COLOR_NEUTRAL,
      1,
  )
  cv2.circle(frame, (cam_cx, cam_cy), ALIGNMENT_THRESHOLD_PX, COLOR_NEUTRAL, 1)

  if target_center is None:
    # Search mode overlay
    cv2.putText(
        frame,
        "STATUS: SEARCHING FOR TARGET...",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        COLOR_RED,
        2,
    )
    return

  cX, cY = target_center
  dx = cX - cam_cx
  dy = cY - cam_cy
  dist_px = math.hypot(dx, dy)

  # Calculate bearing angle (0° = straight ahead/up, +90° = right, -90° = left)
  bearing_deg = math.degrees(math.atan2(dx, -dy))

  # Calculate tag in-plane roll orientation from its top edge (corner 0 to corner 1)
  p0, p1 = tag_corners[0], tag_corners[1]
  tag_orient_deg = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))

  is_aligned = dist_px <= ALIGNMENT_THRESHOLD_PX
  status_color = COLOR_GREEN if is_aligned else COLOR_CYAN
  status_text = "STATUS: TARGET LOCKED" if is_aligned else "STATUS: ALIGNING..."

  # 1. Draw Direction Vector Arrow
  if dist_px > 5:
    # Main vector line from camera center to tag center
    cv2.arrowedLine(
        frame,
        (cam_cx, cam_cy),
        (cX, cY),
        status_color,
        2,
        tipLength=min(0.2, 25.0 / max(dist_px, 1)),
    )

  # 2. Draw Target Lock Ring
  cv2.circle(frame, (cX, cY), 8, status_color, -1)
  cv2.circle(frame, (cX, cY), 18, status_color, 2)

  # 3. Semi-transparent Telemetry Overlay Box
  overlay = frame.copy()
  cv2.rectangle(overlay, (20, 15), (340, 155), (0, 0, 0), -1)
  cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

  # 4. Telemetry Text
  cv2.putText(
      frame,
      status_text,
      (30, 42),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.65,
      status_color,
      2,
  )
  cv2.putText(
      frame,
      f"TARGET ID    : {tag_id}",
      (30, 70),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.5,
      (255, 255, 255),
      1,
  )
  cv2.putText(
      frame,
      f"OFFSET (X,Y) : ({dx:+4d}, {dy:+4d}) px",
      (30, 92),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.5,
      (255, 255, 255),
      1,
  )
  cv2.putText(
      frame,
      f"BEARING      : {bearing_deg:+6.1f} deg",
      (30, 114),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.5,
      (255, 255, 255),
      1,
  )
  cv2.putText(
      frame,
      f"TAG ROLL     : {tag_orient_deg:+6.1f} deg",
      (30, 136),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.5,
      (255, 255, 255),
      1,
  )


def main():
  indices = get_camera_index()
  if not indices:
    print("Error: No cameras detected.")
    return

  cam_idx = 1 if 1 in indices else indices[0]
  print(
      f"Available cameras: {indices}. Connecting to index {cam_idx} (iPhone)..."
  )

  cap = cv2.VideoCapture(cam_idx, cv2.CAP_AVFOUNDATION)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

  dictionary = cv2.aruco.getPredefinedDictionary(TAG_FAMILY)
  parameters = cv2.aruco.DetectorParameters()
  parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

  if hasattr(cv2.aruco, "ArucoDetector"):
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    detect_fn = detector.detectMarkers
  else:
    detect_fn = lambda img: cv2.aruco.detectMarkers(
        img, dictionary, parameters=parameters
    )

  print("Tracking running. Move iPhone to steer the vector. Press 'q' to quit.")

  while True:
    ret, frame = cap.read()
    if not ret:
      break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detect_fn(gray)

    target_center = None
    target_corners = None
    target_id = None

    if ids is not None and len(ids) > 0:
      cv2.aruco.drawDetectedMarkers(frame, corners, ids)

      # Target the first detected tag
      flat_ids = ids.flatten()
      target_id = int(flat_ids[0])
      target_corners = corners[0].reshape((4, 2))

      cX = int(target_corners[:, 0].mean())
      cY = int(target_corners[:, 1].mean())
      target_center = (cX, cY)

    # Render HUD over camera stream
    draw_drone_hud(frame, target_center, target_corners, target_id)

    cv2.imshow("Drone Guidance Overlay (Press 'q' to quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
      break

  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()