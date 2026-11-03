import math
import cv2
import numpy as np

# =====================================================================
# SIMULATION & OPTICAL CONFIGURATION
# =====================================================================
PAD_SIZE_M = 0.65  # 65 cm x 65 cm physical landing pad
TEX_RES = 650  # 1 pixel = 1 mm texture resolution

CAM_RES = 340  # 340 x 340 resolution per downward camera
FOCAL_WIDE = 240.0  # Wide-FOV camera focal length (~71 deg FOV)
FOCAL_NARROW = 460.0  # Narrow-FOV camera focal length (~40 deg FOV)

# Multi-scale AprilTag layout on the pad:
# { tag_id: (center_east_m, center_north_m, side_length_m) }
PAD_LAYOUT = {
    0: (0.000, 0.000, 0.050),  # Center Bullseye (5 cm)
    1: (-0.175, +0.175, 0.200),  # North-West (20 cm)
    2: (+0.175, +0.175, 0.200),  # North-East (20 cm)
    3: (+0.175, -0.175, 0.200),  # South-East (20 cm)
    4: (-0.175, -0.175, 0.200),  # South-West (20 cm)
}


class DualCameraDroneSim:

  def __init__(self):
    # 1. Initialize AprilTag dictionary & detector
    dict_id = getattr(cv2.aruco, "DICT_APRILTAG_36h11", None) or getattr(
        cv2.aruco, "DICT_APRILTAG_36H11"
    )
    self.dict_tag = cv2.aruco.getPredefinedDictionary(dict_id)

    self.params = cv2.aruco.DetectorParameters()
    self.params.minMarkerPerimeterRate = 0.02
    self.params.adaptiveThreshWinSizeMin = 3
    self.params.adaptiveThreshWinSizeMax = 23
    self.params.adaptiveThreshWinSizeStep = 4
    self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    self.params.cornerRefinementWinSize = 3

    if hasattr(cv2.aruco, "ArucoDetector"):
      self.detector = cv2.aruco.ArucoDetector(self.dict_tag, self.params)
      self.detect_fn = self.detector.detectMarkers
    else:
      self.detect_fn = lambda gray: cv2.aruco.detectMarkers(
          gray, self.dict_tag, parameters=self.params
      )

    # 2. Precompute 3D object corners for every tag in Unified Pad Frame
    self.tag_obj_corners = {}
    for tid, (cx_w, cy_w, side_m) in PAD_LAYOUT.items():
      h = side_m / 2.0
      self.tag_obj_corners[tid] = np.array(
          [
              [cx_w - h, -(cy_w + h), 0.0],
              [cx_w + h, -(cy_w + h), 0.0],
              [cx_w + h, -(cy_w - h), 0.0],
              [cx_w - h, -(cy_w - h), 0.0],
          ],
          dtype=np.float32,
      )

    # 3. Build pad texture & initialize drone state
    self._build_pad_texture()
    self.reset()

  def _generate_marker(self, tag_id, size_px):
    if hasattr(cv2.aruco, "generateImageMarker"):
      return cv2.aruco.generateImageMarker(
          self.dict_tag, tag_id, size_px, borderBits=1
      )
    return cv2.aruco.drawMarker(self.dict_tag, tag_id, size_px, borderBits=1)

  def _build_pad_texture(self):
    """Generates high-contrast 650x650 landing pad with 5 nested AprilTags."""
    self.pad_tex = np.full((TEX_RES, TEX_RES, 3), 245, dtype=np.uint8)
    c = TEX_RES // 2

    cv2.rectangle(
        self.pad_tex, (8, 8), (TEX_RES - 8, TEX_RES - 8), (40, 40, 200), 4
    )
    cv2.circle(self.pad_tex, (c, c), 42, (0, 180, 0), 2)
    cv2.line(self.pad_tex, (c - 48, c), (c + 48, c), (0, 180, 0), 1)
    cv2.line(self.pad_tex, (c, c - 48), (c, c + 48), (0, 180, 0), 1)

    scale = TEX_RES / PAD_SIZE_M
    for tid, (cx_w, cy_w, side_m) in PAD_LAYOUT.items():
      tag_px = int(round(side_m * scale))
      raw = self._generate_marker(tid, tag_px)
      tag_bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

      u_center = int(round(c + cx_w * scale))
      v_center = int(round(c - cy_w * scale))
      u0 = u_center - tag_px // 2
      v0 = v_center - tag_px // 2
      self.pad_tex[v0 : v0 + tag_px, u0 : u0 + tag_px] = tag_bgr

  def reset(self, random_offset=False):
    """Resets drone to initial approach altitude and lateral/heading offset."""
    if random_offset:
      self.z = float(np.random.uniform(1.00, 3.20))  # Random altitude (1.0m - 3.2m)
      max_rad = min(0.95, self.z * 0.42)
      rad = float(np.random.uniform(0.15, max_rad))
      ang = float(np.random.uniform(0, 2 * math.pi))
      self.x = rad * math.cos(ang)
      self.y = rad * math.sin(ang)
      self.yaw = float(np.random.uniform(-math.radians(45), math.radians(45)))
    else:
      self.x = 0.60  # East offset (m)
      self.y = 0.45  # North offset (m)
      self.z = 2.20  # Default altitude (m)
      self.yaw = math.radians(35.0)  # Heading offset (rad)

    self.vx, self.vy, self.vz = 0.0, 0.0, 0.0
    self.yaw_rate = 0.0
    self.auto_dock = True
    self.state = "ALIGN_HIGH"

    self.profile_sign = 1.0 if (self.x + self.y) >= 0 else -1.0
    self.trail = []
    self._record_trail_point()

  def _record_trail_point(self):
    r = math.hypot(self.x, self.y)
    if r < 0.03:
      self.profile_sign = 1.0 if (self.x + self.y) >= 0 else -1.0
    r_signed = self.profile_sign * r

    if not self.trail:
      self.trail.append((self.x, self.y, self.z, r_signed))
    else:
      lx, ly, lz, _ = self.trail[-1]
      if math.hypot(self.x - lx, self.y - ly) > 0.015 or abs(self.z - lz) > 0.015:
        self.trail.append((self.x, self.y, self.z, r_signed))
        if len(self.trail) > 180:
          self.trail.pop(0)

  def get_body_velocities(self):
    """Projects world velocities (vx, vy) into Drone Body Frame (Forward, Right)."""
    cos_y, sin_y = math.cos(self.yaw), math.sin(self.yaw)
    v_fwd = self.vx * sin_y + self.vy * cos_y
    v_right = self.vx * cos_y - self.vy * sin_y
    return v_fwd, v_right

  def step_physics(self, cmd_vx_b, cmd_vy_b, cmd_vz_b, cmd_yaw_rate, dt=0.033):
    """Updates 3D drone kinematics with motor lag and touchdown clamp."""
    if self.state == "DOCKED" and self.auto_dock:
      self.vx = self.vy = self.vz = self.yaw_rate = 0.0
      return

    if self.auto_dock:
      cos_y, sin_y = math.cos(self.yaw), math.sin(self.yaw)
      world_vx = cmd_vx_b * sin_y + cmd_vy_b * cos_y
      world_vy = cmd_vx_b * cos_y - cmd_vy_b * sin_y

      lag = 0.22
      self.vx += (world_vx - self.vx) * lag
      self.vy += (world_vy - self.vy) * lag
      self.vz += (cmd_vz_b - self.vz) * lag
      self.yaw_rate += (cmd_yaw_rate - self.yaw_rate) * lag

      self.x += self.vx * dt
      self.y += self.vy * dt
      self.z -= self.vz * dt
      self.yaw = (self.yaw + self.yaw_rate * dt + math.pi) % (
          2 * math.pi
      ) - math.pi

    # Ground contact clamp
    if self.z <= 0.04:
      self.z = 0.04
      self.vx = self.vy = self.vz = self.yaw_rate = 0.0
      if self.auto_dock:
        self.state = "DOCKED"

    self._record_trail_point()

  def render_camera(self, focal_px):
    """Synthesizes downward camera image using sub-pixel float32 perspective warp."""
    h_pad = PAD_SIZE_M / 2.0
    pad_corners_w = [
        (-h_pad, +h_pad),  # NW
        (+h_pad, +h_pad),  # NE
        (+h_pad, -h_pad),  # SE
        (-h_pad, -h_pad),  # SW
    ]

    cos_y, sin_y = math.cos(self.yaw), math.sin(self.yaw)
    zc = max(self.z, 0.02)
    half_c = CAM_RES / 2.0

    dst_pts = []
    for wx, wy in pad_corners_w:
      dx = wx - self.x
      dy = wy - self.y
      xc = cos_y * dx - sin_y * dy
      yc = -(sin_y * dx + cos_y * dy)

      u = (xc * focal_px / zc) + half_c
      v = (yc * focal_px / zc) + half_c
      dst_pts.append([u, v])

    src_pts = np.array(
        [[0, 0], [TEX_RES, 0], [TEX_RES, TEX_RES], [0, TEX_RES]],
        dtype=np.float32,
    )
    dst_pts = np.array(dst_pts, dtype=np.float32)

    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    frame = cv2.warpPerspective(
        self.pad_tex,
        H,
        (CAM_RES, CAM_RES),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(42, 42, 45),
    )
    return frame

  def estimate_pose_from_camera(self, frame, focal_px):
    """Detects all visible AprilTags in frame and runs unified multi-tag solvePnP."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = self.detect_fn(gray)

    if ids is None or len(ids) == 0:
      return None

    all_obj_pts = []
    all_img_pts = []
    detected_ids = []
    edge_angles = []

    for corner, raw_id in zip(corners, ids.flatten()):
      tid = int(raw_id)
      if tid in self.tag_obj_corners:
        pts_2d = corner.reshape((4, 2)).astype(np.float32)
        all_obj_pts.append(self.tag_obj_corners[tid])
        all_img_pts.append(pts_2d)
        detected_ids.append(tid)

        dx_edge = float(pts_2d[1][0] - pts_2d[0][0])
        dy_edge = float(pts_2d[1][1] - pts_2d[0][1])
        edge_angles.append(math.atan2(dy_edge, dx_edge))

    if not detected_ids:
      return None

    cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    obj_pts_flat = np.vstack(all_obj_pts)
    img_pts_flat = np.vstack(all_img_pts)

    K = np.array(
        [[focal_px, 0, CAM_RES / 2.0], [0, focal_px, CAM_RES / 2.0], [0, 0, 1]],
        dtype=np.float32,
    )
    dist = np.zeros((4, 1), dtype=np.float32)

    success, rvec, tvec = cv2.solvePnP(
        obj_pts_flat, img_pts_flat, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success or tvec is None:
      return None

    tx, ty, tz = tvec.flatten()
    est_fwd = float(-ty)
    est_right = float(tx)
    est_alt = float(tz)

    mean_sin = sum(math.sin(a) for a in edge_angles) / len(edge_angles)
    mean_cos = sum(math.cos(a) for a in edge_angles) / len(edge_angles)
    yaw_err = math.atan2(mean_sin, mean_cos)

    u_bull = int(round((tx * focal_px / max(tz, 0.02)) + CAM_RES / 2.0))
    v_bull = int(round((ty * focal_px / max(tz, 0.02)) + CAM_RES / 2.0))

    return {
        "fwd": est_fwd,
        "right": est_right,
        "alt": est_alt,
        "yaw_err": yaw_err,
        "ids": detected_ids,
        "bullseye_px": (u_bull, v_bull),
    }


def draw_camera_hud(frame, pose_data, title):
  """Draws targeting crosshairs and steering vector onto a camera feed."""
  c = CAM_RES // 2
  cv2.line(frame, (c - 15, c), (c + 15, c), (180, 180, 180), 1)
  cv2.line(frame, (c, c - 15), (c, c + 15), (180, 180, 180), 1)
  cv2.circle(frame, (c, c), 22, (120, 120, 120), 1)

  if pose_data is not None:
    bx, by = pose_data["bullseye_px"]
    dist_px = math.hypot(bx - c, by - c)
    col = (0, 255, 0) if dist_px < 18 else (255, 255, 0)

    bx_c = max(5, min(CAM_RES - 5, bx))
    by_c = max(5, min(CAM_RES - 5, by))
    if dist_px > 3:
      cv2.arrowedLine(frame, (c, c), (bx_c, by_c), col, 2, tipLength=0.2)
    cv2.circle(frame, (bx_c, by_c), 5, col, -1)

    tag_str = ",".join(str(i) for i in sorted(pose_data["ids"]))
    cv2.putText(
        frame,
        f"Tags: [{tag_str}]",
        (10, CAM_RES - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 120),
        1,
    )
  else:
    cv2.putText(
        frame,
        "NO TAGS IN VIEW",
        (10, CAM_RES - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 80, 255),
        1,
    )

  cv2.rectangle(frame, (0, 0), (CAM_RES, 26), (20, 20, 22), -1)
  cv2.putText(
      frame, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1
  )


def render_radar(sim, size=210):
  """Draws top-down world position radar with trajectory trail."""
  radar = np.full((size, size, 3), 32, dtype=np.uint8)
  c = size // 2
  scale = 95  # pixels per meter

  cv2.line(radar, (c, 24), (c, size - 6), (48, 48, 52), 1)
  cv2.line(radar, (6, c), (size - 6, c), (48, 48, 52), 1)

  p_half = int(PAD_SIZE_M * scale / 2)
  cv2.rectangle(
      radar,
      (c - p_half, c - p_half),
      (c + p_half, c + p_half),
      (225, 225, 225),
      -1,
  )
  cv2.circle(radar, (c, c), int(round(0.10 * scale)), (0, 180, 80), 1)
  cv2.drawMarker(radar, (c, c), (0, 150, 0), cv2.MARKER_CROSS, 12, 2)

  if len(sim.trail) > 1:
    pts = []
    for tx, ty, _, _ in sim.trail:
      px = max(6, min(size - 6, int(round(c + tx * scale))))
      py = max(24, min(size - 6, int(round(c - ty * scale))))
      pts.append([px, py])
    cv2.polylines(
        radar, [np.array(pts, dtype=np.int32)], False, (180, 160, 0), 1, cv2.LINE_AA
    )

  dx = int(round(c + sim.x * scale))
  dy = int(round(c - sim.y * scale))
  dx = max(10, min(size - 10, dx))
  dy = max(26, min(size - 10, dy))

  tip_x = int(round(dx + 16 * math.sin(sim.yaw)))
  tip_y = int(round(dy - 16 * math.cos(sim.yaw)))
  cv2.line(radar, (dx, dy), (tip_x, tip_y), (0, 255, 255), 2, cv2.LINE_AA)
  cv2.circle(radar, (dx, dy), 6, (0, 140, 255), -1)

  cv2.rectangle(radar, (0, 0), (size, 22), (20, 20, 22), -1)
  cv2.putText(
      radar,
      "TOP-DOWN RADAR (XY)",
      (8, 15),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.40,
      (210, 210, 210),
      1,
  )
  return radar


def render_side_profile(sim, cmd_vz_b, size=210):
  """Draws vertical side-profile altitude & glide-slope funnel visualizer."""
  prof = np.full((size, size, 3), 32, dtype=np.uint8)
  c_x = size // 2
  ground_y = size - 22
  top_y = 28
  max_z = 3.50  # Expanded max altitude to 3.5 meters

  scale_z = (ground_y - top_y) / max_z
  scale_r = 95.0

  def z_to_y(z_m):
    return int(round(ground_y - np.clip(z_m, 0.0, max_z) * scale_z))

  def r_to_x(r_m):
    return int(round(np.clip(c_x + r_m * scale_r, 8, size - 8)))

  # 1. Shaded Glide-Slope Funnel Polygon
  funnel_pts = np.array(
      [
          [r_to_x(-0.35), z_to_y(max_z)],
          [r_to_x(+0.35), z_to_y(max_z)],
          [r_to_x(+0.10), z_to_y(1.10)],
          [r_to_x(+0.10), ground_y],
          [r_to_x(-0.10), ground_y],
          [r_to_x(-0.10), z_to_y(1.10)],
      ],
      dtype=np.int32,
  )
  overlay = prof.copy()
  cv2.fillPoly(overlay, [funnel_pts], (38, 72, 45))
  cv2.addWeighted(overlay, 0.45, prof, 0.55, 0, prof)

  # 2. Horizontal Altitude Grid Lines & Labels (0.5m to 3.0m)
  for z_tick in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    gy = z_to_y(z_tick)
    cv2.line(prof, (28, gy), (size - 8, gy), (48, 48, 52), 1)
    cv2.putText(
        prof,
        f"{z_tick:.1f}m",
        (4, gy + 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.30,
        (130, 130, 135),
        1,
    )

  cv2.line(prof, (c_x, top_y), (c_x, ground_y), (60, 90, 60), 1)
  cv2.polylines(prof, [funnel_pts], False, (60, 175, 85), 1, cv2.LINE_AA)

  y_commit = z_to_y(0.14)
  cv2.line(
      prof, (r_to_x(-0.25), y_commit), (r_to_x(+0.25), y_commit), (120, 120, 60), 1
  )

  # 3. Ground Tarmac & Landing Pad Cross-Section
  cv2.line(prof, (0, ground_y), (size, ground_y), (90, 90, 95), 2)
  p_half = int(round((PAD_SIZE_M / 2.0) * scale_r))
  cv2.rectangle(
      prof,
      (c_x - p_half, ground_y - 3),
      (c_x + p_half, ground_y + 3),
      (230, 230, 230),
      -1,
  )
  cv2.rectangle(
      prof, (c_x - 4, ground_y - 4), (c_x + 4, ground_y + 4), (0, 220, 0), -1
  )

  # 4. Drone Current Signed Radial Coordinate & Altitude
  r_curr = math.hypot(sim.x, sim.y) * sim.profile_sign
  drone_x = r_to_x(r_curr)
  drone_y = z_to_y(sim.z)

  # 5. Downward Camera Optical FOV Cones (Wide & Narrow)
  if sim.z > 0.06:
    half_w_m = sim.z * (CAM_RES / (2.0 * FOCAL_WIDE))
    half_n_m = sim.z * (CAM_RES / (2.0 * FOCAL_NARROW))

    fov_overlay = prof.copy()
    wide_cone = np.array(
        [
            [drone_x, drone_y],
            [r_to_x(r_curr - half_w_m), ground_y],
            [r_to_x(r_curr + half_w_m), ground_y],
        ],
        dtype=np.int32,
    )
    narrow_cone = np.array(
        [
            [drone_x, drone_y],
            [r_to_x(r_curr - half_n_m), ground_y],
            [r_to_x(r_curr + half_n_m), ground_y],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(fov_overlay, [wide_cone], (90, 70, 35))
    cv2.fillPoly(fov_overlay, [narrow_cone], (110, 110, 45))
    cv2.addWeighted(fov_overlay, 0.25, prof, 0.75, 0, prof)

  # 6. Historical Trajectory Trail in Side Profile
  if len(sim.trail) > 1:
    t_pts = [[r_to_x(pr), z_to_y(pz)] for _, _, pz, pr in sim.trail]
    cv2.polylines(
        prof, [np.array(t_pts, dtype=np.int32)], False, (0, 210, 255), 1, cv2.LINE_AA
    )

  # 7. Descent Velocity Vector Arrow & Side-View Quadcopter Icon
  if cmd_vz_b > 0.02 and sim.state != "DOCKED":
    arr_len = int(round(min(28, cmd_vz_b * 55)))
    cv2.arrowedLine(
        prof,
        (drone_x, drone_y + 4),
        (drone_x, min(ground_y - 2, drone_y + 4 + arr_len)),
        (0, 255, 160),
        2,
        tipLength=0.35,
    )

  cv2.line(
      prof, (drone_x - 11, drone_y), (drone_x + 11, drone_y), (0, 165, 255), 2
  )
  cv2.line(
      prof, (drone_x - 9, drone_y - 4), (drone_x - 9, drone_y), (200, 200, 200), 1
  )
  cv2.line(
      prof, (drone_x + 9, drone_y - 4), (drone_x + 9, drone_y), (200, 200, 200), 1
  )
  cv2.line(
      prof, (drone_x - 14, drone_y - 4), (drone_x - 4, drone_y - 4), (0, 255, 255), 2
  )
  cv2.line(
      prof, (drone_x + 4, drone_y - 4), (drone_x + 14, drone_y - 4), (0, 255, 255), 2
  )
  cv2.circle(prof, (drone_x, drone_y), 3, (255, 255, 255), -1)

  cv2.putText(
      prof,
      f"Z: {sim.z:.2f}m | R: {math.hypot(sim.x, sim.y)*100:.1f}cm",
      (36, size - 6),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.35,
      (200, 220, 200),
      1,
  )
  cv2.rectangle(prof, (0, 0), (size, 22), (20, 20, 22), -1)
  cv2.putText(
      prof,
      "SIDE GLIDE-SLOPE (RZ)",
      (8, 15),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.40,
      (210, 210, 210),
      1,
  )
  return prof


def main():
  sim = DualCameraDroneSim()

  while True:
    wide_frame = sim.render_camera(FOCAL_WIDE)
    narrow_frame = sim.render_camera(FOCAL_NARROW)

    pose_wide = sim.estimate_pose_from_camera(wide_frame, FOCAL_WIDE)
    pose_narrow = sim.estimate_pose_from_camera(narrow_frame, FOCAL_NARROW)

    fused = None
    active_source = "NONE"
    if pose_wide and pose_narrow:
      w_n, w_w = 0.70, 0.30
      fused = {
          "fwd": w_n * pose_narrow["fwd"] + w_w * pose_wide["fwd"],
          "right": w_n * pose_narrow["right"] + w_w * pose_wide["right"],
          "alt": w_n * pose_narrow["alt"] + w_w * pose_wide["alt"],
          "yaw_err": (
              w_n * pose_narrow["yaw_err"] + w_w * pose_wide["yaw_err"]
          ),
      }
      active_source = "FUSED (WIDE + NARROW)"
    elif pose_narrow:
      fused = pose_narrow
      active_source = "NARROW ONLY"
    elif pose_wide:
      fused = pose_wide
      active_source = "WIDE ONLY"

    cmd_vx_b, cmd_vy_b, cmd_vz_b, cmd_yaw_rate = 0.0, 0.0, 0.0, 0.0
    est_fwd, est_right, err_xy, yaw_err_deg = 0.0, 0.0, 0.0, 0.0

    if fused is not None:
      est_fwd = fused["fwd"]
      est_right = fused["right"]
      err_xy = math.hypot(est_fwd, est_right)
      yaw_err_deg = math.degrees(fused["yaw_err"])

      curr_v_fwd, curr_v_right = sim.get_body_velocities()

      KP_XY = 1.65
      KD_XY = 0.70
      KP_YAW = 1.50

      cmd_vx_b = float(
          np.clip(KP_XY * est_fwd - KD_XY * curr_v_fwd, -0.60, 0.60)
      )
      cmd_vy_b = float(
          np.clip(KP_XY * est_right - KD_XY * curr_v_right, -0.60, 0.60)
      )
      cmd_yaw_rate = float(
          np.clip(KP_YAW * fused["yaw_err"], -0.60, 0.60)
      )

      if sim.state != "DOCKED":
        if err_xy > 0.10 or abs(yaw_err_deg) > 8.0:
          sim.state = "ALIGN_HIGH"
          cmd_vz_b = 0.12 if err_xy < 0.30 else 0.0
        elif sim.z > 0.14:
          sim.state = "DESCEND_LOCKED"
          cmd_vz_b = float(np.clip(0.38 * sim.z + 0.06, 0.12, 0.42))
        else:
          sim.state = "FINAL_DOCKING"
          cmd_vz_b = 0.14
    else:
      if sim.state != "DOCKED":
        sim.state = "SEARCHING"

    sim.step_physics(cmd_vx_b, cmd_vy_b, cmd_vz_b, cmd_yaw_rate)

    draw_camera_hud(wide_frame, pose_wide, "CAM 1: WIDE-FOV (APPROACH)")
    draw_camera_hud(narrow_frame, pose_narrow, "CAM 2: NARROW-FOV (PRECISION)")

    canvas = np.full((470, 1200, 3), 24, dtype=np.uint8)
    canvas[20 : 20 + CAM_RES, 20 : 20 + CAM_RES] = wide_frame
    canvas[20 : 20 + CAM_RES, 380 : 380 + CAM_RES] = narrow_frame

    radar = render_radar(sim, size=210)
    side_prof = render_side_profile(sim, cmd_vz_b, size=210)
    canvas[20 : 20 + 210, 740 : 740 + 210] = radar
    canvas[20 : 20 + 210, 968 : 968 + 210] = side_prof

    true_err_xy = math.hypot(sim.x, sim.y)
    status_col = (
        (0, 255, 120)
        if fused is not None or sim.state == "DOCKED"
        else (0, 80, 255)
    )

    hud_lines = [
        (f"FSM STATE   : {sim.state}", status_col, 0.55, 2),
        (f"SENSOR MODE : {active_source}", (0, 220, 255), 0.46, 1),
        (
            f"FLIGHT MODE : {'AUTONOMOUS' if sim.auto_dock else 'MANUAL'}",
            (220, 220, 220),
            0.46,
            1,
        ),
        (
            f"TRUE POS XY : ({sim.x:+.3f}, {sim.y:+.3f}) m  [Err:"
            f" {true_err_xy*100:.1f} cm]",
            (240, 240, 240),
            0.46,
            1,
        ),
        (
            f"ALTITUDE Z  : {sim.z:.3f} m   |  YAW :"
            f" {math.degrees(sim.yaw):+5.1f} deg",
            (240, 240, 240),
            0.46,
            1,
        ),
        (
            f"PNP EST XY  : ({est_fwd:+.3f}, {est_right:+.3f}) m",
            (200, 255, 200),
            0.46,
            1,
        ),
        (
            f"CMD VEL     : Vx={cmd_vx_b:+.2f}  Vy={cmd_vy_b:+.2f} "
            f" Vz={cmd_vz_b:.2f} m/s",
            (0, 220, 255),
            0.46,
            1,
        ),
    ]

    hy = 252
    for text, col, scale, thick in hud_lines:
      cv2.putText(
          canvas, text, (740, hy), cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick
      )
      hy += 27

    cv2.rectangle(canvas, (20, 375), (720, 450), (35, 35, 40), -1)
    cv2.putText(
        canvas,
        "CONTROLS:  [R] Default (2.2m) | [T] Random 3D Spawn (1.0-3.2m) |"
        " [Space] Auto/Manual | [Q] Quit",
        (32, 402),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.41,
        (220, 220, 220),
        1,
    )
    cv2.putText(
        canvas,
        "NUDGE / FLY:  [W/A/S/D] Horizontal XY   |   [I / K] Climb / Descend"
        " Altitude Z",
        (32, 432),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (0, 220, 255),
        1,
    )

    cv2.imshow("Dual-Camera Autonomous Drone Docking Simulator", canvas)

    key = cv2.waitKey(30) & 0xFF
    if key == ord("q") or key == 27:
      break
    elif key == ord("r"):
      sim.reset(random_offset=False)
    elif key == ord("t"):
      sim.reset(random_offset=True)
    elif key == ord(" "):
      sim.auto_dock = not sim.auto_dock
      if sim.auto_dock and sim.z > 0.05:
        sim.state = "ALIGN_HIGH"
      else:
        sim.vx = sim.vy = sim.vz = 0.0
    elif key == ord("w"):
      sim.y += 0.12
      if sim.state == "DOCKED":
        sim.z = 0.60
        sim.state = "ALIGN_HIGH"
    elif key == ord("s"):
      sim.y -= 0.12
      if sim.state == "DOCKED":
        sim.z = 0.60
        sim.state = "ALIGN_HIGH"
    elif key == ord("a"):
      sim.x -= 0.12
      if sim.state == "DOCKED":
        sim.z = 0.60
        sim.state = "ALIGN_HIGH"
    elif key == ord("d"):
      sim.x += 0.12
      if sim.state == "DOCKED":
        sim.z = 0.60
        sim.state = "ALIGN_HIGH"
    elif key == ord("i"):
      sim.z = min(3.40, sim.z + 0.25)
      sim.vz = 0.0  # Stop descent velocity so it doesn't immediately pull down
      if sim.state == "DOCKED":
        sim.state = "ALIGN_HIGH"
    elif key == ord("k"):
      sim.z = max(0.04, sim.z - 0.25)
      sim.vz = 0.0

  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()