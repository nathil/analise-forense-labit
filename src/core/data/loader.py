import os
import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from PIL import Image

class BivTattDatasetLoader:
    """
    Loads images and their corresponding labels from the BivTatt dataset.
    The dataset follows the nomenclature:
    - Original: label_number.jpg
    - Transformed: label_number_transformation.jpg
    
    It also finds the corresponding bounding box text files (x1 y1 x2 y2).
    """
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / "images"
        self.boxes_dir = self.root_dir / "bounding_boxes"
        
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found at {self.images_dir}")

    def load_dataset(self, start: int = 0, length: int = None) -> List[Dict]:
        """
        Scans the directory and builds a list of dictionaries containing
        metadata for each image.
        """
        dataset = []
        files = [
            p for p in self.images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.IMAGE_EXTENSIONS
        ]
        
        # Sort files to ensure deterministic partitioning
        files.sort(key=lambda x: x.name)
        
        for file_path in files:
            filename = file_path.stem
            parts = filename.split("_")
            
            # Extract label (subject/tattoo family ID)
            label = parts[0]
            
            # Check if it's an original image (gallery) or transformed (probe)
            is_original = len(parts) == 2
            transformation = parts[2] if not is_original else None
            
            # Check for bounding box file
            box_path = self.boxes_dir / f"{filename}.txt"
            has_box = box_path.exists()
            
            dataset.append({
                "image_path": str(file_path),
                "box_path": str(box_path) if has_box else None,
                "label": label,
                "is_original": is_original,
                "transformation": transformation,
                "filename": file_path.name
            })
            
        if length is not None:
            return dataset[start:start+length]
        return dataset[start:]


class YOLOv5WeaponDatasetLoader:
    """
    Loads images and YOLOv5 annotations from the Weapon Dataset for YOLOv5
    (raghavnanjappan/weapon-dataset-for-yolov5 or any compatible YOLOv5 dataset).

    Labeling & Splitting Strategy to Maximize Retrieval Metrics:
    1. Semantic Class Labeling from YOLOv5 Annotations:
       Extracts the weapon class ('knife' or 'handgun') of the primary (largest area)
       bounding box in each image's .txt label file (with fallback to filename/folder).
    2. Bounding Box Association:
       Pairs each image with its YOLOv5 normalized .txt file so BoundingBoxCrop isolates
       the weapon region from background clutter prior to feature extraction.
    3. Stratified Interleaved Gallery/Probe Split (80% Gallery / 20% Probe):
       Interleaves classes deterministically and assigns 4 Gallery (is_original=True)
       per 1 Probe (is_original=False) within each class so any slice [start:start+length]
       contains balanced classes and zero leakage.
    """
    DEFAULT_CLASS_MAP = {
        0: "knife",
        1: "handgun"
    }
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root_dir: str, probe_ratio: float = 0.2):
        self.root_dir = Path(root_dir)
        self.probe_ratio = max(0.05, min(0.5, probe_ratio))
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root directory not found at {self.root_dir}")
        self.class_map = self._load_class_map()

    def _load_class_map(self) -> Dict[int, str]:
        """Attempts to read class names from data.yaml or classes.txt; falls back to {0: knife, 1: handgun}."""
        class_map = dict(self.DEFAULT_CLASS_MAP)

        # Check classes.txt
        for classes_file in self.root_dir.rglob("classes.txt"):
            try:
                lines = [l.strip() for l in classes_file.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]
                if lines:
                    return {idx: name.lower() for idx, name in enumerate(lines)}
            except Exception:
                pass

        # Check data.yaml / dataset.yaml
        for yaml_file in list(self.root_dir.rglob("*.yaml")) + list(self.root_dir.rglob("*.yml")):
            try:
                content = yaml_file.read_text(encoding="utf-8", errors="ignore")
                # Match list syntax: names: ['knife', 'handgun']
                list_match = re.search(r"names\s*:\s*\[([^\]]+)\]", content)
                if list_match:
                    raw_names = [n.strip().strip("'\"") for n in list_match.group(1).split(",") if n.strip()]
                    if raw_names:
                        return {idx: name.lower() for idx, name in enumerate(raw_names)}
            except Exception:
                pass

        return class_map

    def _find_box_path(self, img_path: Path) -> Optional[Path]:
        """Finds the corresponding YOLOv5 .txt annotation file for a given image path."""
        stem = img_path.stem
        candidates = [
            img_path.with_suffix(".txt"),
            img_path.parent.parent / "labels" / f"{stem}.txt",
            img_path.parent.parent / "bounding_boxes" / f"{stem}.txt",
        ]
        # Also support images/train/xxx.jpg -> labels/train/xxx.txt
        parts = list(img_path.parts)
        if "images" in parts:
            idx = len(parts) - 1 - parts[::-1].index("images")
            label_parts = parts[:idx] + ["labels"] + parts[idx + 1:]
            candidates.append(Path(*label_parts).with_suffix(".txt"))

        for cand in candidates:
            if cand.exists() and cand.name.lower() != "classes.txt":
                return cand
        return None

    def _resolve_label(self, img_path: Path, box_path: Optional[Path]) -> Optional[str]:
        """Determines the semantic class label from the largest YOLOv5 box or filename/folder."""
        # 1. Try reading primary (largest area) object class from YOLOv5 .txt file
        if box_path is not None and box_path.exists():
            try:
                lines = [l.strip() for l in box_path.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]
                best_cls_id = None
                max_area = -1.0
                for line in lines:
                    tokens = line.split()
                    if len(tokens) >= 5:
                        cls_id = int(float(tokens[0]))
                        bw = float(tokens[3])
                        bh = float(tokens[4])
                        area = bw * bh
                        if area > max_area:
                            max_area = area
                            best_cls_id = cls_id
                    elif len(tokens) >= 1:
                        best_cls_id = int(float(tokens[0]))
                if best_cls_id is not None:
                    return self.class_map.get(best_cls_id, f"class_{best_cls_id}")
            except Exception:
                pass

        # 2. Fallback: infer from filename or parent directory name
        search_str = f"{img_path.parent.name}_{img_path.stem}".lower()
        if any(k in search_str for k in ("knife", "knives", "blade", "dagger", "faca")):
            return "knife"
        if any(k in search_str for k in ("handgun", "pistol", "gun", "revolver", "firearm", "arma", "weapons")):
            return "handgun"

        # Generic prefix fallback
        prefix = re.split(r"[_\-\s\d]+", img_path.stem.lower())[0]
        return prefix if prefix else None

    def load_dataset(self, start: int = 0, length: int = None) -> List[Dict]:
        """
        Scans root_dir recursively for images, resolves YOLOv5 bounding boxes and
        semantic class labels, and builds a stratified interleaved Gallery/Probe list.
        """
        all_images = [
            p for p in self.root_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in self.IMAGE_EXTENSIONS
        ]
        all_images.sort(key=lambda p: (p.name, str(p)))

        items_by_label: Dict[str, List[Dict]] = {}
        seen_filenames = set()

        for img_path in all_images:
            # Avoid duplicate filenames if Kaggle archive has redundant nested copies
            unique_key = (img_path.name, img_path.stat().st_size)
            if unique_key in seen_filenames:
                continue
            seen_filenames.add(unique_key)

            box_path = self._find_box_path(img_path)
            label = self._resolve_label(img_path, box_path)
            if not label:
                continue

            # Ensure unique filename identifier across splits for runner metrics
            rel_parent = img_path.parent.name.lower()
            filename_id = img_path.name if rel_parent in ("images", "") else f"{rel_parent}_{img_path.name}"

            items_by_label.setdefault(label, []).append({
                "image_path": str(img_path),
                "box_path": str(box_path) if box_path is not None else None,
                "label": label,
                "filename": filename_id,
                "_split_hint": rel_parent
            })

        # Assign Gallery (is_original=True) vs Probe (is_original=False) per class
        # Every N-th item (e.g. 1 in 5 for probe_ratio=0.2) is assigned as Probe
        step = max(2, int(round(1.0 / self.probe_ratio)))
        processed_by_label: Dict[str, List[Dict]] = {}

        for label, items in sorted(items_by_label.items()):
            class_list = []
            for idx, item in enumerate(items):
                is_probe = (idx % step == (step - 1))
                # Guarantee at least 1 gallery item when class has >= 2 items
                if len(items) >= 2 and idx == 0:
                    is_probe = False
                elif len(items) >= 2 and idx == 1 and step > len(items):
                    is_probe = True

                class_list.append({
                    "image_path": item["image_path"],
                    "box_path": item["box_path"],
                    "label": item["label"],
                    "is_original": not is_probe,
                    "transformation": "yolo_probe" if is_probe else None,
                    "filename": item["filename"]
                })
            processed_by_label[label] = class_list

        # Interleave classes deterministically (round-robin) so any slice [start:start+length]
        # has a balanced representation of all weapon classes and both Gallery & Probe items
        interleaved_dataset: List[Dict] = []
        sorted_labels = sorted(processed_by_label.keys())
        max_len = max((len(v) for v in processed_by_label.values()), default=0)

        for i in range(max_len):
            for lbl in sorted_labels:
                if i < len(processed_by_label[lbl]):
                    interleaved_dataset.append(processed_by_label[lbl][i])

        if length is not None:
            return interleaved_dataset[start:start + length]
        return interleaved_dataset[start:]

class DatasetSplitter:
    """
    Splits the dataset into Gallery and Probe sets based on the forensic protocol.
    Gallery: Original images (clean).
    Probe: Transformed images.
    """
    @staticmethod
    def split(dataset: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Returns a tuple of (Gallery, Probe).
        Ensures zero leakage between the sets based on the 'is_original' flag.
        """
        gallery = []
        probe = []
        
        for item in dataset:
            if item["is_original"]:
                gallery.append(item)
            else:
                probe.append(item)
                
        return gallery, probe


# ===========================================================================
# Weapon Dataset for YOLOv5 (Kaggle: raghavnanjappan/weapon-dataset-for-yolov5)
# ~4156 imagens (train 4000 + val 156) de 2 classes -> {0: Knife, 1: Handgun}.
# Layout real:
#   <root>/images/train/*.jpg   <root>/images/val/*.jpg
#   <root>/labels/train/*.txt   <root>/labels/val/*.txt   ('class cx cy w h' YOLO)
#   <root>/dataset.yaml         (names em bloco: '0: Knife' / '1: Handgun')
# Nao ha pares 'original/transformado', mas ha muitas imagens por CLASSE, entao o
# retrieval e por TIPO de arma (class split): label = nome da classe; dentro de
# cada classe metade vira galeria e metade vira probe. Saida = mesma List[Dict]
# do BivTattDatasetLoader.
#
# Regras respeitadas (sem elas a acuracia e 0 / o run aborta):
#   - galeria e probe compartilham 'label' (o acerto e por igualdade de label);
#   - a lista sai INTERCALADA G,P,G,P (a TUI fatia dataset_full antes do split);
#   - box_path, se gerado, e 'x1 y1 x2 y2' em PIXELS (formato BivTatt); o box YOLO
#     normalizado e convertido e gravado num sidecar sob data/_derived/.
# ===========================================================================
_WY_IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
_WY_SPLIT_DIRS = ("train", "valid", "validation", "val", "test")


def _wy_image_size(path: str) -> Tuple[int, int]:
    try:
        with Image.open(path) as im:
            return im.size  # (W, H); .size nao decodifica os pixels
    except Exception:
        return (0, 0)


def _wy_parse_label(label_path):
    """Le um .txt YOLO -> [(class_id, cx, cy, w, h)] em floats. Ausente -> []."""
    rows = []
    if not label_path or not os.path.exists(label_path):
        return rows
    try:
        with open(label_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    cid = int(float(parts[0]))
                    cx, cy, w, h = (float(x) for x in parts[1:5])
                except ValueError:
                    continue
                rows.append((cid, cx, cy, w, h))
    except Exception:
        pass
    return rows


def _wy_to_abs(cx, cy, w, h, W, H) -> Tuple[int, int, int, int]:
    bw, bh = w * W, h * H
    x1 = max(0, min(int(round(cx * W - bw / 2)), W))
    y1 = max(0, min(int(round(cy * H - bh / 2)), H))
    x2 = max(0, min(int(round(cx * W + bw / 2)), W))
    y2 = max(0, min(int(round(cy * H + bh / 2)), H))
    return x1, y1, x2, y2


def _wy_class_names(root: Path) -> Dict[int, str]:
    """Le o yaml do dataset (names: [..] ou 'names:\\n  0: Knife'). Sem PyYAML."""
    for name in ("dataset.yaml", "dataset.yml", "data.yaml", "data.yml"):
        p = root / name
        if not p.exists():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        m = re.search(r"names\s*:\s*\[(.*?)\]", text, re.S)
        if m:
            items = [i.strip() for i in
                     re.findall(r"['\"]?([^,'\"\[\]]+?)['\"]?\s*(?:,|$)", m.group(1))
                     if i.strip()]
            return {i: n for i, n in enumerate(items)}
        mapping = {}
        for mm in re.finditer(r"^\s*(\d+)\s*:\s*['\"]?([^'\"\n]+?)['\"]?\s*$",
                              text, re.M):
            mapping[int(mm.group(1))] = mm.group(2).strip()
        if mapping:
            return mapping
    return {}


def _wy_find_pairs(root: Path):
    """
    [(image_path, label_path|None, rel_id)] casando imagem<->label pelo subcaminho.
    Cobre os layouts YOLO comuns:
      images/<split>/*.jpg + labels/<split>/*.txt   (o desta base),
      images/*.jpg         + labels/*.txt,
      <split>/images       + <split>/labels.
    """
    pairs = []

    def from_roots(images_root: Path, labels_root):
        for img in sorted(images_root.rglob("*")):
            if img.suffix.lower() not in _WY_IMG_EXT:
                continue
            rel = img.relative_to(images_root)          # ex: train/00001.jpg
            lbl = None
            if labels_root is not None:
                cand = labels_root / rel.with_suffix(".txt")  # labels/train/00001.txt
                if cand.exists():
                    lbl = cand
            rel_id = str(rel.with_suffix("")).replace(os.sep, "_")
            pairs.append((img, lbl, rel_id))

    # 1) images/ e labels/ no topo (com ou sem subpastas train/val) -- caso real
    if (root / "images").is_dir():
        from_roots(root / "images",
                   root / "labels" if (root / "labels").is_dir() else None)
        return pairs
    # 2) <split>/images + <split>/labels
    for d in _WY_SPLIT_DIRS:
        sp = root / d
        if sp.is_dir() and (sp / "images").is_dir():
            from_roots(sp / "images",
                       sp / "labels" if (sp / "labels").is_dir() else None)
    if pairs:
        return pairs
    # 3) fallback: imagens e .txt soltos na raiz
    from_roots(root, root)
    return pairs


class WeaponYolov5DatasetLoader:
    def __init__(self, root_dir: str, dataset_key: str = "weapon_yolov5",
                 use_boxes: bool = True, max_per_class: int = None):
        self.root = Path(root_dir)
        if not self.root.exists():
            raise FileNotFoundError(f"Directory not found at {self.root}")
        self.dataset_key = dataset_key
        self.use_boxes = use_boxes
        self.max_per_class = max_per_class
        self.class_names = _wy_class_names(self.root)

    def _box_dir(self) -> Path:
        p = Path("data") / "_derived" / self.dataset_key / "boxes"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _box_for(self, img: Path, lbl, stem: str, cid: int):
        if not self.use_boxes or lbl is None:
            return None
        rows = [r for r in _wy_parse_label(str(lbl)) if r[0] == cid] \
            or _wy_parse_label(str(lbl))
        if not rows:
            return None
        _, cx, cy, w, h = max(rows, key=lambda r: r[3] * r[4])
        W, H = _wy_image_size(str(img))
        if W == 0 or H == 0:
            return None
        x1, y1, x2, y2 = _wy_to_abs(cx, cy, w, h, W, H)
        if x2 <= x1 or y2 <= y1:
            return None
        out = self._box_dir() / f"{stem}.txt"
        out.write_text(f"{x1} {y1} {x2} {y2}\n")
        return str(out)

    def load_dataset(self, start: int = 0, length: int = None) -> List[Dict]:
        pairs = _wy_find_pairs(self.root)
        if not pairs:
            raise FileNotFoundError(
                f"Nenhuma imagem em {self.root} (esperado images/labels em YOLO).")

        # agrupa por classe dominante (maior box) da anotacao YOLO
        by_class: Dict[str, list] = {}
        for img, lbl, rel_id in pairs:
            rows = _wy_parse_label(str(lbl)) if lbl else []
            if not rows:
                continue
            cid = max(rows, key=lambda r: r[3] * r[4])[0]
            cname = self.class_names.get(cid, f"class_{cid}")
            by_class.setdefault(cname, []).append((img, lbl, rel_id, cid))

        gallery, probe = [], []
        for cname in sorted(by_class):
            items = sorted(by_class[cname], key=lambda t: t[2])
            if self.max_per_class:
                items = items[: self.max_per_class]
            if len(items) < 2:
                continue  # nao da para formar galeria + probe
            half = len(items) // 2
            for k, (img, lbl, rel_id, cid) in enumerate(items):
                is_orig = k < half
                (gallery if is_orig else probe).append({
                    "image_path": str(img),
                    "box_path": self._box_for(img, lbl, rel_id, cid),
                    "label": cname,
                    "is_original": is_orig,
                    "transformation": None if is_orig else "class_probe",
                    "filename": img.name,
                })

        # intercala G,P,G,P para que fatias por indice contenham ambos
        dataset, i, j = [], 0, 0
        while i < len(gallery) or j < len(probe):
            if i < len(gallery):
                dataset.append(gallery[i]); i += 1
            if j < len(probe):
                dataset.append(probe[j]); j += 1

        if length is not None:
            return dataset[start:start + length]
        return dataset[start:]
