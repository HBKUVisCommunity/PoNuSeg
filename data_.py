import numpy as np
from typing import Tuple, Dict
from skimage import io, color, filters
from skimage.filters import sobel
from skimage.graph import MCP_Geometric
from skimage.segmentation import watershed, find_boundaries
from skimage.morphology import remove_small_objects, binary_opening, binary_closing, disk
from scipy.ndimage import binary_fill_holes, distance_transform_edt, label as cc_label
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree
import os
from tqdm import tqdm
from dists.dists import getMBD as getDist
import torch
import FastGeodis
from PIL import Image
import xml.etree.ElementTree as ET
import os
import numpy as np
import cv2
import json
from PIL import Image

keys_Mo = {'dataset':{'name':'MoNuSeg',
                      'dir':'./data/MoNuSeg/',
                      'tissue_images_dir':'images/',
                      'annotations_dir':'annotations/',
                      'full_labels_dir':'labels/',
                      'images_dir':'images_/',
                      'labels_dir':'labels_/',
                      'grid_size':4,
                      'test':{'1':'1'}}, 
           'info':{'verbos':1}           
       }

class Config:
    def __init__(self, file_path):
        self.file_path = file_path
        self.config = self._load()
        self.set_defaults(self.config,keys_Mo)
        self.save()
    def set_defaults(self,config,defaults,override=True):
        for key, value in defaults.items():
            if key not in config or override == True:
                config[key] = value
            elif isinstance(value, dict):
                self.set_defaults(config[key], value)
            
    def _load(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Config file '{self.file_path}' does not exist.")
        with open(self.file_path, 'r') as f:
            return json.load(f)

    def get(self, key, default=None):
        """Get a parameter using dotted key notation."""
        keys = key.split('.')
        val = self.config
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def set(self, key, value):
        """Set a parameter using dotted key notation."""
        keys = key.split('.')
        d = self.config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    def validate(self, required_keys):
        """Check that all required keys exist in the config."""
        missing = [k for k in required_keys if self.get(k) is None]
        if missing:
            raise KeyError(f"Missing required config keys: {missing}")


    def save(self, out_path=None):
        """Save the config to a file (overwrite or to a new path)."""
        path = out_path if out_path else self.file_path
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=4)

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        self.set(key, value)

    def __repr__(self):
        return json.dumps(self.config, indent=4)

class MoNuSegData:
    def __init__(self,cfg):
        self.cfg = cfg

    def read_xml_file(self,file_path):
        tree = ET.parse(file_path)
        return tree.getroot()
    
    def read_contour(self,vertices):
        points = []
        for child in vertices:
            x = float(child.attrib.get('X'))
            y = float(child.attrib.get('Y'))
            points.append((x, y))
        return points
    
    def extract_contours_from_xml(self,xml_root):
        contour_list = []
        def traverse(element):
            if element.tag == 'Vertices':
                contour = self.read_contour(element)
                contour_list.append(contour)
            else:
                for child in element:
                    traverse(child)
        traverse(xml_root)
        return contour_list
    
    def annotation_to_instance_mask(self,xml_file, image_shape):
        root = self.read_xml_file(xml_file)
        contours = self.extract_contours_from_xml(root)
    
        height, width = image_shape[:2]
        instance_mask = np.zeros((height, width), dtype=np.uint16)
    
        for idx, contour in enumerate(contours, start=1):
            polygon = np.array(contour, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(instance_mask, [polygon], color=idx)

        return instance_mask
        
    def generate_numpy_instance_labels(self):
        xml_folder = self.cfg['dataset.dir']+self.cfg['dataset.annotations_dir']
        image_folder = self.cfg['dataset.dir']+self.cfg['dataset.tissue_images_dir']
        output_folder = self.cfg['dataset.dir']+self.cfg['dataset.full_labels_dir']
        
        # Make sure output folder exists
        os.makedirs(output_folder, exist_ok=True)
        if cfg['info.verbos']==1:
            print('Generating numpy masks from annotation xml files')
        # Process files
        for file in os.listdir(xml_folder):
            if not file.endswith('.xml'):
                continue
        
            xml_path = os.path.join(xml_folder, file)
            image_path = os.path.join(image_folder, file.replace('.xml', '.tif'))
            output_path = os.path.join(output_folder, file.replace('.xml', '.npy'))
        
            # Load image to get shape
            image = cv2.imread(image_path)
            mask = self.annotation_to_instance_mask(xml_path, image.shape)
        
            # Save the mask as .npy
            np.save(output_path, mask)
            if cfg['info.verbos']==1:
                print(f"Saved numpy mask to: {output_path}")
            
    def split_into_grid(self):
        image_folder = self.cfg['dataset.dir'] + self.cfg['dataset.tissue_images_dir']
        label_folder = self.cfg['dataset.dir'] + self.cfg['dataset.full_labels_dir']

        output_image_folder = self.cfg['dataset.dir'] + self.cfg['dataset.images_dir']
        output_label_folder = self.cfg['dataset.dir'] + self.cfg['dataset.labels_dir']

        os.makedirs(output_image_folder, exist_ok=True)
        os.makedirs(output_label_folder, exist_ok=True)
        grid_size = self.cfg['dataset.grid_size']
        if cfg['info.verbos']==1:
            print(f'Partitioning images and labels into a grid of size {grid_size}')

        for file in os.listdir(image_folder):
            if not file.endswith('.tif'):
                continue

            img_path = os.path.join(image_folder, file)
            label_path = os.path.join(label_folder, file.replace('.tif', '.npy'))

            img = np.array(Image.open(img_path))
            label = np.load(label_path)

            h, w = img.shape[:2]
            tile_h, tile_w = h // grid_size, w // grid_size

            base_name = os.path.splitext(file)[0]

            for i in range(grid_size):
                for j in range(grid_size):
                    y1, y2 = i * tile_h, (i + 1) * tile_h
                    x1, x2 = j * tile_w, (j + 1) * tile_w

                    img_tile = img[y1:y2, x1:x2]
                    label_tile = label[y1:y2, x1:x2]

                    img_tile_path = os.path.join(output_image_folder, f"{base_name}_{i}_{j}.png")
                    label_tile_path = os.path.join(output_label_folder, f"{base_name}_{i}_{j}.npy")

                    Image.fromarray(img_tile).save(img_tile_path)
                    np.save(label_tile_path, label_tile)
                    if cfg['info.verbos']==1:
                        print(f"Saved: {img_tile_path}, {label_tile_path}")


    
    
    

def save_array_as_png(arr: np.ndarray, path: str):
    """
    Min–max-normalize a 2-D NumPy array to 0–255 and save as an 8-bit PNG.

    Parameters
    ----------
    arr : np.ndarray        # shape (H, W) or (1, H, W)
    path : str              # filename, e.g. "output.png"
    """
    img = np.squeeze(arr).astype(float)

    vmin, vmax = img.min(), img.max()
    if vmax > vmin:
        img = (img - vmin) / (vmax - vmin)
    img = (img * 255).round().astype(np.uint8)

    Image.fromarray(img, mode="L").save(path)


# =========================================================
# 0) Load seeds from point-mask PNG (centers as nonzero pixels)
# =========================================================
def load_seeds_from_point_mask(pm: np.ndarray,
                               patch_half: int = 25,
                               image_shape: Tuple[int, int] = None) -> np.ndarray:
    """
    Returns seeds as (N,2) int array of (y, x).
    Optionally remove points too close to edges (if you later use local ops).
    """
    if pm.ndim == 3:
        pm = pm[..., 0]
    pm_bin = pm > 0
    seeds_yx = np.argwhere(pm_bin)

    if image_shape is None:
        H, W = pm_bin.shape
    else:
        H, W = image_shape
        assert (H, W) == pm_bin.shape, "Point mask and image must have same spatial size."

    # Optional: keep points not too close to border (useful in some postproc steps)
    keep = (
        (seeds_yx[:, 0] >= 0) & (seeds_yx[:, 0] < H) &
        (seeds_yx[:, 1] >= 0) & (seeds_yx[:, 1] < W)
    )
    seeds_yx = seeds_yx[keep]
    return seeds_yx.astype(int)


# =========================================================
# 1) Feature construction for the FULL image
# =========================================================
def build_features(image_rgb: np.ndarray,
                               seeds_yx: np.ndarray,
                               image_lbl: np.ndarray,
                               geodesic_beta: float = 25.0) -> Dict[str, np.ndarray]:
    """
    image_rgb: HxW×3 (uint8 or float in [0,1])
    seeds_yx : (N,2) int, (y,x)
    Returns:
      dict with:
        'X'     : (H*W, D) feature matrix
        'lab'   : (H,W,3)
        'ed2s'  : (H,W) Euclidean distance to nearest seed (normalized)
        'gd2s'  : (H,W) Geodesic distance to nearest seed on edge-cost map (normalized)
    """
    # normalize image
    if image_rgb.dtype.kind in "ui":
        img = (image_rgb.astype(np.float32) / 255.0).clip(0, 1)
    else:
        img = np.clip(image_rgb.astype(np.float32), 0.0, 1.0)

    H, W = img.shape[:2]
    # Color features in Lab
    lab = color.rgb2lab(img).astype(np.float32)

    # Euclidean distance to nearest seed (in pixels) -> normalized [0,1]
    # edt computes distance to nearest zero; so build a map with zeros at seed positions
    ones = np.ones((H, W), dtype=bool)
    seed_zero = ones.copy()
    seed_zero[:] = True
    seed_zero[tuple(seeds_yx.T)] = False  # zeros at seeds
    ed2s = distance_transform_edt(seed_zero).astype(np.float32)
    if ed2s.max() > 0:
        ed2s /= ed2s.max()

    # Geodesic distance to nearest seed using an edge-based cost (graph shortest paths)
    # Edge strength on luminance -> cost = 1 + beta * edges
    gray = color.rgb2gray(img)
    edges = sobel(gray)
    edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-8)
    cost = 1.0 + geodesic_beta * edges

    # Multi-source MCP: distances to the CLOSEST seed by path cost
    starts = [tuple(yx) for yx in seeds_yx]
    mcp = MCP_Geometric(cost)
    gd2s, _ = mcp.find_costs(starts)
    gd2s = np.asarray(gd2s, dtype=np.float32)
    gmin, gmax = np.nanmin(gd2s), np.nanmax(gd2s)
    gd2s = (gd2s - gmin) / (gmax - gmin + 1e-8)
    gd2s = np.where(np.isfinite(gd2s), gd2s, 1.0).astype(np.float32)
    
    dist_embeddings = distance_transform_edt(255 - image_lbl).reshape(-1, 1)
    dist_embeddings = np.clip(dist_embeddings, a_min=0, a_max=20)
    #color_embeddings = np.array(ori_image, dtype=float).reshape(-1, 3) / 10
    #embeddings = np.concatenate((color_embeddings, clip_dist_embeddings), axis=1)

    # Stack features (L,a,b, ed2s, gd2s) per pixel
    X = np.column_stack([
        lab[..., 0].reshape(-1),
        lab[..., 1].reshape(-1),
        lab[..., 2].reshape(-1),
        dist_embeddings.reshape(-1),
        ed2s.reshape(-1),
        gd2s.reshape(-1),
    ]).astype(np.float32)

    return {"X": X, "lab": lab, "ed2s": ed2s, "gd2s": gd2s}


# =========================================================
# 2) KMeans clustering over the FULL image
# =========================================================
def kmeans_segment(features: Dict[str, np.ndarray],
                              n_clusters: int = 2,
                              random_state: int = 42) -> Tuple[np.ndarray, int]:
    """
    Returns:
      labels_img : (H,W) int labels
      nuc_k      : index of the cluster chosen as nuclei (0..K-1)
    """
    X = features["X"]
    H, W = features["lab"].shape[:2]
    lab = features["lab"]; ed2s = features["ed2s"]; gd2s = features["gd2s"]
    
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = km.fit_predict(Xs).reshape(H, W)
    lbls = np.unique(labels)
    return labels


# =========================================================
# 3) Voronoi labeling (nearest seed) for the FULL grid
# =========================================================
def voronoi_labels(H: int, W: int, seeds_yx: np.ndarray) -> np.ndarray:
    """
    Returns a label image (H,W) with values in {1..N} indicating the nearest seed by Euclidean distance.
    """
    yy, xx = np.mgrid[0:H, 0:W]
    coords = np.stack([yy.ravel(), xx.ravel()], axis=1)  # (H*W, 2)
    tree = cKDTree(seeds_yx)  # (N,2) with (y,x)
    _, nn_idx = tree.query(coords, k=1, workers=-1)
    return (nn_idx.reshape(H, W) + 1).astype(np.int32)  # 1..N


# =========================================================
# 4) Post-processing helpers
# =========================================================
def keep_seed_component(mask: np.ndarray, seed_yx: Tuple[int, int]) -> np.ndarray:
    """Keep only the connected component that contains the given seed."""
    lab, n = cc_label(mask)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    sy, sx = seed_yx
    if not mask[sy, sx]:
        return np.zeros_like(mask, dtype=bool)
    return lab == lab[sy, sx]

def clean_binary(mask: np.ndarray, min_area: int = 30, close_r: int = 2, open_r: int = 1) -> np.ndarray:
    if min_area > 0:
        mask = remove_small_objects(mask, min_size=min_area)
    if close_r > 0:
        mask = binary_closing(mask, footprint=disk(close_r))
    if open_r > 0:
        mask = binary_opening(mask, footprint=disk(open_r))
    mask = binary_fill_holes(mask)
    return mask


# =========================================================
# 5) WHOLE-IMAGE nuclei segmentation (no partitioning)
#     - KMeans → FG mask
#     - Voronoi + (graph-based) watershed to separate touching nuclei
# =========================================================
def segment_nuclei_Kmeans(image_rgb: np.ndarray,
                               seeds_yx: np.ndarray,
                               image_lbl: np.ndarray,
                               geodesic_beta: float = 25.0,
                               min_area: int = 30,
                               n_clusters: int = 2,
                               random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    image_rgb: (H,W,3)
    seeds_yx : (N,2) int (y,x)

    Returns:
      instance_mask: (H,W) int32; 0=bg, 1..N = per-seed instance labels
      binary_mask  : (H,W) bool union of all instances
      debug dict
    """
    H, W = image_rgb.shape[:2]

    # ---- Features & whole-image KMeans ----
    feats = build_features(image_rgb, seeds_yx,image_lbl, geodesic_beta=geodesic_beta)
    labels_img = kmeans_segment(feats, n_clusters=n_clusters, random_state=random_state)
    overlap_nums = [np.sum((labels_img == i) * image_lbl) for i in range(n_clusters)]        
    nuclei_idx = np.argmax(overlap_nums)
    
    nuclei_cluster = labels_img == nuclei_idx
    background_cluster = labels_img != nuclei_idx


    fg_mask = (labels_img == nuclei_idx)
    fg_mask = clean_binary(fg_mask, min_area=min_area, close_r=2, open_r=1)

    # ---- Voronoi partition (Euclidean nearest seed) ----
    vlabels = voronoi_labels(H, W, seeds_yx)  # 1..N per pixel
    vor_bound = find_boundaries(vlabels, mode="thick")  # Voronoi ridges (graph cut lines)
    # OPTIONAL: carve Voronoi ridges from foreground to help split joints
    # (only where fg is present; prevents a single blob spanning multiple seeds)
    fg_mask = fg_mask & (~vor_bound)
    # ---- Graph-based watershed (markers = seeds) restricted to foreground ----
    # Elevation inside FG: -distance transform to split touching objects
    dist_inside = distance_transform_edt(fg_mask).astype(np.float32)
    elevation = -dist_inside  # higher (less negative) on boundaries
    # Markers: label image 0=bg, 1..N at seed locations (only those that fall inside fg)
    markers = np.zeros((H, W), dtype=np.int32)
    for i, (y, x) in enumerate(seeds_yx, start=1):
        markers[y, x] = i
    # If some seeds lie slightly off FG (due to clustering errors), you may dilate/relax here if needed.

    inst_ws = watershed(elevation, markers=markers, mask=fg_mask)  # graph-based (minimum spanning forest)

    # ---- Enforce Voronoi consistency (never let a seed "own" pixels outside its Voronoi cell)
    inst_final = np.where(inst_ws == 0, 0, inst_ws)
    mismatch = (inst_final > 0) & (inst_final != vlabels)
    inst_final[mismatch] = 0  # remove inconsistent pixels
    # Fill any small gaps for each instance and keep the seed-connected component
    binary_mask = inst_final > 0
    if binary_mask.any():
        for i, (y, x) in enumerate(seeds_yx, start=1):
            comp = (inst_final == i)
            if comp.any():
                comp = clean_binary(comp, min_area=max(5, min_area // 2), close_r=1, open_r=1)
                comp = keep_seed_component(comp, (y, x))
                inst_final[inst_final == i] = 0
                inst_final[comp] = i

    # Remove tiny stray pixels globally
    binary_mask = inst_final > 0
    if binary_mask.any():
        cleaned_union = remove_small_objects(binary_mask, min_size=max(5, min_area))
        removed = binary_mask & (~cleaned_union)
        inst_final[removed] = 0
        binary_mask = cleaned_union

    return inst_final.astype(np.int32), binary_mask.astype(bool)
    
def prepare_kmeans_lbls(imgs_path,points_path,lbls_path,no_layers = 5):
    os.makedirs(lbls_path, exist_ok=True)
    fs = [x for x in os.listdir(imgs_path) if x.endswith('png')]
    #fs = ['TCGA-18-5592-01Z-00-DX1_0_0.png']
    for img_name in tqdm(fs):
        image_rgb = io.imread(imgs_path+img_name)
        image_lbl = io.imread(points_path+img_name)
        H, W = image_rgb.shape[:2]
        seeds_yx = load_seeds_from_point_mask(io.imread(points_path+img_name), image_shape=(H, W))
        final = np.zeros(image_rgb.shape[:2],np.uint8)
        for j in range(no_layers):
            inst_mask, bin_mask = segment_nuclei_Kmeans(
                image_rgb,
                seeds_yx,
                image_lbl=image_lbl,
                geodesic_beta=25.0,
                min_area=30,
                n_clusters=3+j,
                random_state=42
            )        
            final = final + bin_mask*(255//no_layers)
        io.imsave(lbls_path+img_name, final.astype(np.uint8))

def create_circle_mask(center, radius, shape=(251, 250), dtype=np.uint8):
    """
    Create a binary mask with a filled circle.

    Parameters
    ----------
    center : tuple (x, y)
        Pixel coordinates of the circle centre (x = column, y = row).
    radius : int or float
        Circle radius in pixels.
    shape : tuple, optional
        (height, width) of the output mask. Default is (256, 256).
    dtype : NumPy dtype, optional
        Data type of the mask. Use np.bool_ for True/False, or an
        integer type (e.g., np.uint8) for 0/1. Default is np.uint8.

    Returns
    -------
    mask : ndarray
        Binary mask of the requested shape with ones inside the circle
        and zeros elsewhere.
    """
    h, w = shape
    cx, cy = center

    # Coordinate grids (row indices = Y, column indices = X)
    Y, X = np.ogrid[:h, :w]

    # Squared distance from (cx, cy)
    dist2 = (X - cx)**2 + (Y - cy)**2

    # Create mask: inside circle = 1, outside = 0
    mask = (dist2 <= radius**2).astype(dtype)
    return mask

def np_class2one_hot(seg: np.ndarray, K: int) -> np.ndarray:
    # print("Np enters")
    """
    Seems to be blocking here when using multi-processing.
    Don't know why, so for now I'll re-implement the same function in numpy
    which should be faster anyhow, but can introduce inconsistencies in the code
    so need to be careful.
    """
    
    b, w, h = seg.shape
    res = np.zeros((b, K, w, h), dtype=np.int64)
    np.put_along_axis(res, seg[:, None, :, :], 1, axis=1)

    return res
    
def to_dmap(im,m):
    
    #labels, img = sources
    K: int = 2

    lab_arr = m.astype(np.uint8) #np.ndarray = np.asarray(Image.open(labels).convert(mode="L"))
    img_arr = np.array(Image.fromarray(im).convert("L")).astype(np.float32)    

    assert lab_arr.shape == img_arr.shape
    assert lab_arr.dtype == np.uint8

    lab_oh: np.ndarray = np_class2one_hot(lab_arr[None, ...], K)[0]
    assert lab_oh.shape == (K, *img_arr.shape), lab_oh.shape

    res: np.ndarray = np.zeros(lab_oh.shape, dtype=np.float32)

    neg_oh: np.ndarray = np.logical_not(lab_oh)

    for k in range(K):
        if lab_oh[k].any():
            pos_dist = getDist(img_arr.squeeze(), lab_oh[[k], :, :].squeeze())
            neg_dist = getDist(img_arr.squeeze(), neg_oh[[k], :, :].squeeze())
            res[k, :, :] = pos_dist - neg_dist
            save_array_as_png(pos_dist, f"output_pos_dmap{k}.png")
            save_array_as_png(neg_dist, f"output_neg_dmap{k}.png")

    return res

def to_distmap(im,m):
    lamb = 0.5 #{"intensity": 1, "geodesic": 0.5, "euclidean": 0}

    K = 2 


    lab_arr = m.astype(np.uint8) #np.ndarray = np.asarray(Image.open(labels).convert(mode="L"))
    img_arr = im.astype(np.float32)    
    assert lab_arr.shape == img_arr.shape
    assert lab_arr.dtype == np.uint8
    lab_oh =  np_class2one_hot(lab_arr[None, ...], K)[0]
    assert lab_oh.shape == (K, *img_arr.shape), lab_oh.shape

    res: np.ndarray = np.zeros(lab_oh.shape, dtype=np.float32)

    neg_oh: np.ndarray = np.logical_not(lab_oh)

    device = "cpu" # cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu" if (not torch.cuda.is_available() or args.distmap_mode == "mbd") else "cuda"

    img_torch = torch.from_numpy(img_arr)[None, None, :, :].to(device)
    for k in range(K):
        if lab_oh[k].any():
            lab_torch = torch.from_numpy(lab_oh[None, [k], :, :]).to(device)
            neg_torch = torch.from_numpy(neg_oh[None, [k], :, :]).to(device)
            
            assert img_torch.shape == lab_torch.shape, (
                img_torch.shape,
                lab_torch.shape,
            )

            pos_dist = FastGeodis.generalised_geodesic2d(img_torch, lab_torch, 1e10, lamb, 2)
            neg_dist = FastGeodis.generalised_geodesic2d(img_torch, neg_torch, 1e10, lamb, 2)
            res[k, :, :] = neg_dist[0, 0].cpu().numpy()

    return res

def clean_binary(mask: np.ndarray, min_area: int = 30, close_r: int = 2, open_r: int = 1) -> np.ndarray:
    if min_area > 0:
        mask = remove_small_objects(mask, min_size=min_area)
    if close_r > 0:
        mask = binary_closing(mask, footprint=disk(close_r))
    if open_r > 0:
        mask = binary_opening(mask, footprint=disk(open_r))
    mask = binary_fill_holes(mask)
    return mask

def voronoi_labels(H: int, W: int, seeds_yx: np.ndarray) -> np.ndarray:
    """
    Returns a label image (H,W) with values in {1..N} indicating the nearest seed by Euclidean distance.
    """
    yy, xx = np.mgrid[0:H, 0:W]
    coords = np.stack([yy.ravel(), xx.ravel()], axis=1)  # (H*W, 2)
    tree = cKDTree(seeds_yx)  # (N,2) with (y,x)
    _, nn_idx = tree.query(coords, k=1, workers=-1)
    return (nn_idx.reshape(H, W) + 1).astype(np.int32)  # 1..N
    
def keep_seed_component(mask: np.ndarray, seed_yx) -> np.ndarray:
    """Keep only the connected component that contains the given seed."""
    lab, n = cc_label(mask)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    sy, sx = seed_yx
    if not mask[sy, sx]:
        return np.zeros_like(mask, dtype=bool)
    return lab == lab[sy, sx]
def nuclei_geodist_mask(im,m):
    H, W = im.shape
    seeds_yx = load_seeds_from_point_mask(m, image_shape=(H, W))
    mask = to_distmap(im,m).copy()
    mask_ = mask[1]
    mask_ = (1-(mask_-mask_.min())/(mask_.max()-mask_.min()))*255
    print(mask_.shape)
    fg_mask = (gaussian_filter((mask_), sigma = 1.5)>220).astype(np.uint8)
    #H,W = fg_mask.shape[0],fg_mask.shape[1]    
    
    min_area: int = 30
    vlabels = voronoi_labels(H, W, seeds_yx)
    vor_bound = find_boundaries(vlabels, mode="thick")  # Voronoi ridges (graph cut lines)
    
    markers = np.zeros((H, W), dtype=np.int32)
    for idx, (y, x) in enumerate(seeds_yx, start=1):
            markers[y, x] = idx
    # If some seeds lie slightly off FG (due to clustering errors), you may dilate/relax here if needed.
    fg_mask_sep = fg_mask & (~vor_bound)          
    fg_mask_sep = clean_binary(fg_mask_sep, 0)    # close tiny holes
    fg_mask_sep = fg_mask_sep * (1-vor_bound)
    elevation = -distance_transform_edt(fg_mask_sep)
    inst_ws   = watershed(elevation, markers, mask=fg_mask_sep, compactness=0)

    

    # ---- Enforce Voronoi consistency (never let a seed "own" pixels outside its Voronoi cell)
    inst_final = np.where(inst_ws == 0, 0, inst_ws)##

    mismatch = (inst_ws > 0) & (inst_ws != vlabels) ####
    inst_final[mismatch] = 0  # remove inconsistent pixels

    # Fill any small gaps for each instance and keep the seed-connected component
    binary_mask = inst_final > 0
    if binary_mask.any():
        for i, (y, x) in enumerate(seeds_yx, start=1):
            comp = (inst_final == i)
            if comp.any():
                comp = clean_binary(comp, min_area=max(5, min_area // 2), close_r=1, open_r=1)
                comp = keep_seed_component(comp, (y, x))
                inst_final[inst_final == i] = 0
                inst_final[comp] = i
        
    # Remove tiny stray pixels globally
    binary_mask = inst_final > 0
    if binary_mask.any():
        cleaned_union = remove_small_objects(binary_mask, min_size=max(5, min_area))
        removed = binary_mask & (~cleaned_union)
        inst_final[removed] = 0
        binary_mask = cleaned_union
    final_mask = binary_mask*mask_
    final_mask = (final_mask-final_mask.min())/(final_mask.max()-final_mask.min())
    return final_mask*255
    
def prepare_geodist_lbls(imgs_path,points_path,lbls_path):
    img_names = [f for f in os.listdir(imgs_path) if f.endswith('.png')]
    os.makedirs(lbls_path,exist_ok=True)
    for j, img_name in enumerate(tqdm(img_names)):            
        seeds_path = f'{points_path}{img_name}'
        path_img = f'{imgs_path}{img_name}'# JPEG input images
        im = Image.open(path_img).convert("L")
        im = np.array(im)        
        m = np.array(Image.open(seeds_path).convert("L"))
        m = (m > 0).astype(np.uint8)                       # Binary: 0 or 1
        mask  = nuclei_geodist_mask(im,m)
        io.imsave(lbls_path+img_name, mask.astype(np.uint8))    
import inspect

# --- compatibility wrapper for connected-component labeling ---
def label_cc(mask_bool, connectivity=2):
    """
    Returns a labeled image of connected components.
    Works with both scikit-image (new/old) and scipy.ndimage.
    """
    mask_bool = np.asarray(mask_bool, dtype=bool)

    # Try scikit-image (preferred)
    try:
        from skimage.measure import label as sk_label
        sig = inspect.signature(sk_label)
        if "connectivity" in sig.parameters:
            return sk_label(mask_bool, connectivity=connectivity)
        else:
            # very old scikit-image used neighbors via 'neighbors' or no kw
            # fall back to scipy for clarity
            raise TypeError("skimage.label without 'connectivity'")
    except Exception:
        # Fallback: scipy.ndimage
        from scipy.ndimage import label as ndi_label, generate_binary_structure
        # connectivity=1 -> 4-neigh, connectivity=2 -> 8-neigh in 2D
        struct = generate_binary_structure(rank=2, connectivity=1 if connectivity == 1 else 2)
        comps, _ = ndi_label(mask_bool, structure=struct)
        return comps
    
def keep_one_component_per_region(fg_mask, vlabels, seeds_yx=None, connectivity=2, remove_boundaries=None):
    fg = np.asarray(fg_mask).astype(bool)
    vl = np.asarray(vlabels)
    H, W = fg.shape
    assert vl.shape == fg.shape, "vlabels shape must match fg_mask"

    if remove_boundaries is not None:
        rb = np.asarray(remove_boundaries).astype(bool)
        fg &= ~rb
    else:
        rb = None

    out = np.zeros_like(fg, dtype=bool)

    seed_for_label = {}
    if seeds_yx is not None and len(seeds_yx) > 0:
        seeds_yx = np.asarray(seeds_yx)
        valid = (0 <= seeds_yx[:, 0]) & (seeds_yx[:, 0] < H) & (0 <= seeds_yx[:, 1]) & (seeds_yx[:, 1] < W)
        for y, x in seeds_yx[valid]:
            seed_for_label[vl[y, x]] = (y, x)

    ulabels = np.unique(vl)
    if -1 in ulabels:
        ulabels = ulabels[ulabels != -1]

    for lab in ulabels:
        region = (vl == lab)
        if not np.any(region):
            continue

        # 🔧 replaced cc_label(..., connectivity=...) with version-safe helper:
        comps = label_cc(fg & region, connectivity=connectivity)

        n = int(comps.max())
        if n <= 1:
            out |= (comps > 0)
            continue

        chosen_id = None
        if lab in seed_for_label:
            sy, sx = seed_for_label[lab]
            sid = int(comps[sy, sx])
            if sid > 0:
                chosen_id = sid

        if chosen_id is None:
            counts = np.bincount(comps.ravel(), minlength=n + 1)
            chosen_id = int(np.argmax(counts[1:]) + 1)

        out |= (comps == chosen_id)

    if rb is not None:
        out &= ~rb

    return out
def normalizeStaining(img, saveFile=None, Io=240, alpha=1, beta=0.15):
    ''' Normalize staining appearence of H&E stained images
    
    Example use:
        see test.py
        
    Input:
        I: RGB input image
        Io: (optional) transmitted light intensity
        
    Output:
        Inorm: normalized image
        H: hematoxylin image
        E: eosin image
    
    Reference: 
        A method for normalizing histology slides for quantitative analysis. M.
        Macenko et al., ISBI 2009
    '''
             
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
        
    maxCRef = np.array([1.9705, 1.0308])
    
    # define height and width of image
    h, w, c = img.shape
    
    # reshape image
    img = img.reshape((-1,3))

    # calculate optical density
    OD = -np.log((img.astype(float)+1)/Io)
    
    # remove transparent pixels
    ODhat = OD[~np.any(OD<beta, axis=1)]
        
    # compute eigenvectors
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    
    #eigvecs *= -1
    
    #project on the plane spanned by the eigenvectors corresponding to the two 
    # largest eigenvalues    
    That = ODhat.dot(eigvecs[:,1:3])
    
    phi = np.arctan2(That[:,1],That[:,0])
    
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100-alpha)
    
    vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
    
    # a heuristic to make the vector corresponding to hematoxylin first and the 
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:,0], vMax[:,0])).T
    else:
        HE = np.array((vMax[:,0], vMin[:,0])).T
    
    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T
    
    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE,Y, rcond=None)[0]
    
    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
    tmp = np.divide(maxC,maxCRef)
    C2 = np.divide(C,tmp[:, np.newaxis])
    
    # recreate the image using reference mixing matrix
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm>255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)  
    
    # unmix hematoxylin and eosin
    H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,0], axis=1).dot(np.expand_dims(C2[0,:], axis=0))))
    H[H>255] = 254
    H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)
    
    E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,1], axis=1).dot(np.expand_dims(C2[1,:], axis=0))))
    E[E>255] = 254
    E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)
    
    if saveFile is not None:
        Image.fromarray(Inorm).save(saveFile+'.png')
        Image.fromarray(H).save(saveFile+'_H.png')
        Image.fromarray(E).save(saveFile+'_E.png')

    return Inorm, H, E
