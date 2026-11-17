import cv2
import numpy as np
from typing import Tuple, Optional

class BoundingBoxCrop:
    """
    Handles cropping of the region of interest strictly based on provided coordinates.
    Supports both absolute pixel coordinates ('x1 y1 x2 y2', e.g., BivTatt) and
    normalized YOLOv5 coordinates ('class_id x_center y_center width height' in [0, 1]).
    """
    @staticmethod
    def parse_box_file(box_path: str, image_shape: Optional[Tuple[int, int]] = None) -> Tuple[int, int, int, int]:
        """
        Reads a text file containing either:
        - 4 values: 'x1 y1 x2 y2' (absolute pixels, or normalized if all <= 1.0)
        - 5+ values per line (YOLOv5 format): 'class_id cx cy w h' (normalized in [0, 1]).
          When multiple lines exist, selects the bounding box with the largest area.
        Returns (x1, y1, x2, y2) in pixel coordinates, or (0, 0, 0, 0) if reading fails.
        """
        try:
            with open(box_path, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            if not lines:
                return (0, 0, 0, 0)

            first_parts = [float(x) for x in lines[0].split()]

            # Case 1: YOLOv5 format (5 values per line: class_id cx cy w h)
            if len(first_parts) >= 5:
                if image_shape is None:
                    return (0, 0, 0, 0)
                img_h, img_w = image_shape[:2]
                best_box = (0, 0, 0, 0)
                max_area = -1.0

                for line in lines:
                    parts = [float(x) for x in line.split()]
                    if len(parts) < 5:
                        continue
                    _, cx, cy, bw, bh = parts[:5]
                    area = bw * bh
                    if area > max_area:
                        max_area = area
                        if cx <= 1.05 and cy <= 1.05 and bw <= 1.05 and bh <= 1.05:
                            x1 = int(round((cx - bw / 2.0) * img_w))
                            y1 = int(round((cy - bh / 2.0) * img_h))
                            x2 = int(round((cx + bw / 2.0) * img_w))
                            y2 = int(round((cy + bh / 2.0) * img_h))
                        else:
                            x1 = int(round(cx - bw / 2.0))
                            y1 = int(round(cy - bh / 2.0))
                            x2 = int(round(cx + bw / 2.0))
                            y2 = int(round(cy + bh / 2.0))
                        best_box = (x1, y1, x2, y2)
                return best_box

            # Case 2: 4 values ('x1 y1 x2 y2')
            if len(first_parts) == 4:
                vals = first_parts
                if image_shape is not None and all(0.0 <= v <= 1.0 for v in vals) and any(v > 0.0 for v in vals):
                    img_h, img_w = image_shape[:2]
                    return (
                        int(round(vals[0] * img_w)),
                        int(round(vals[1] * img_h)),
                        int(round(vals[2] * img_w)),
                        int(round(vals[3] * img_h)),
                    )
                return int(vals[0]), int(vals[1]), int(vals[2]), int(vals[3])

        except Exception as e:
            print(f"Failed to read box file {box_path}: {e}")
            
        return (0, 0, 0, 0)

    @staticmethod
    def crop_image(image: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Crops the numpy array image using the box coordinates (x1, y1, x2, y2).
        If the box is invalid or (0,0,0,0), returns the original image.
        """
        x1, y1, x2, y2 = box
        
        # Validate coordinates against image bounds
        h, w = image.shape[:2]
        
        # Ensure coordinates are within valid range and form a valid rectangle
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))
        
        if x1 >= x2 or y1 >= y2:
            # Invalid box, return original
            return image
            
        return image[y1:y2, x1:x2]
