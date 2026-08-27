import os
import re
from typing import List, Dict, Tuple
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
        files = list(self.images_dir.glob("*.jpg"))
        
        # Sort files to ensure deterministic partitioning
        files.sort(key=lambda x: x.name)
        
        for file_path in files:
            filename = file_path.stem
            parts = filename.split("_")
            
            # Extract label
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
