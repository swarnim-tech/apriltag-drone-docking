import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Supported AprilTag families in OpenCV
FAMILIES = {
    "tag36h11": getattr(cv2.aruco, "DICT_APRILTAG_36h11", None)
    or getattr(cv2.aruco, "DICT_APRILTAG_36H11", None),
    "tag25h9": getattr(cv2.aruco, "DICT_APRILTAG_25h9", None)
    or getattr(cv2.aruco, "DICT_APRILTAG_25H9", None),
    "tag16h5": getattr(cv2.aruco, "DICT_APRILTAG_16h5", None)
    or getattr(cv2.aruco, "DICT_APRILTAG_16H5", None),
}


def create_apriltag(
    tag_id: int,
    family: str = "tag36h11",
    tag_size_px: int = 400,
    margin_px: int = 60,
    add_label: bool = True,
) -> Image.Image:
  """Generates an individual AprilTag with a white quiet zone and optional label."""
  dict_id = FAMILIES.get(family.lower())
  if dict_id is None:
    raise ValueError(
        f"Family '{family}' not found. Supported: {list(FAMILIES.keys())}"
    )

  dictionary = cv2.aruco.getPredefinedDictionary(dict_id)

  # Support both OpenCV 4.7+ (generateImageMarker) and legacy versions (drawMarker)
  if hasattr(cv2.aruco, "generateImageMarker"):
    marker_raw = cv2.aruco.generateImageMarker(
        dictionary, tag_id, tag_size_px, borderBits=1
    )
  else:
    marker_raw = cv2.aruco.drawMarker(
        dictionary, tag_id, tag_size_px, borderBits=1
    )

  tag_img = Image.fromarray(marker_raw).convert("RGB")

  # Add white quiet zone (required for accurate camera detection)
  label_height = 45 if add_label else 0
  canvas_w = tag_size_px + (2 * margin_px)
  canvas_h = tag_size_px + (2 * margin_px) + label_height

  canvas = Image.new("RGB", (canvas_w, canvas_h), color="white")
  canvas.paste(tag_img, (margin_px, margin_px))

  if add_label:
    draw = ImageDraw.Draw(canvas)
    label_text = f"{family}  |  ID: {tag_id}"

    # Use standard macOS system font, fall back to default if unavailable
    try:
      font = ImageFont.truetype(
          "/System/Library/Fonts/Supplemental/Arial.ttf", size=20
      )
    except OSError:
      try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Helvetica.ttc", size=20
        )
      except OSError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), label_text, font=font)
    text_w = bbox[2] - bbox[0]
    text_x = (canvas_w - text_w) // 2
    text_y = tag_size_px + margin_px + 12
    draw.text((text_x, text_y), label_text, fill="black", font=font)

  return canvas


def main():
  parser = argparse.ArgumentParser(
      description="Generate AprilTag markers with quiet margins."
  )
  parser.add_argument(
      "--start-id", type=int, default=0, help="Starting tag ID (default: 0)"
  )
  parser.add_argument(
      "--count",
      type=int,
      default=5,
      help="Number of sequential tags to export (default: 5)",
  )
  parser.add_argument(
      "--family",
      type=str,
      default="tag36h11",
      choices=list(FAMILIES.keys()),
      help="AprilTag dictionary family (default: tag36h11)",
  )
  parser.add_argument(
      "--size",
      type=int,
      default=400,
      help="Black tag dimension in pixels (default: 400)",
  )
  parser.add_argument(
      "--margin",
      type=int,
      default=60,
      help="White border padding in pixels (default: 60)",
  )
  parser.add_argument(
      "--no-label",
      action="store_true",
      help="Exclude text identification below the tag",
  )
  parser.add_argument(
      "--output-dir",
      type=str,
      default="output_tags",
      help="Destination folder for exported PNG files",
  )

  args = parser.parse_args()

  dest = Path(args.output_dir)
  dest.mkdir(parents=True, exist_ok=True)

  print(
      f"Generating {args.count} tags ({args.family}) into: {dest.resolve()}\n"
  )

  for tag_id in range(args.start_id, args.start_id + args.count):
    image = create_apriltag(
        tag_id=tag_id,
        family=args.family,
        tag_size_px=args.size,
        margin_px=args.margin,
        add_label=not args.no_label,
    )
    target_file = dest / f"{args.family}_id_{tag_id:03d}.png"
    image.save(target_file, dpi=(300, 300))
    print(f"  ✓ Saved: {target_file.name}")

  print(f"\nCompleted. Files ready in '{args.output_dir}'.")


if __name__ == "__main__":
  main()