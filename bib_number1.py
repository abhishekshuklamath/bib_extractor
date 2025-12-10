import sys
import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import models
from tqdm import tqdm

# Custom preprocess: convert PIL image to normalized tensor without torchvision.transforms
def pil_to_tensor(img_pil):
    # Ensure RGB
    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')
    w, h = img_pil.size
    # raw bytes in RGBRGB... order
    data = img_pil.tobytes()
    # create uint8 tensor from bytes and reshape to (H, W, 3)
    byte_tensor = torch.frombuffer(data, dtype=torch.uint8)
    byte_tensor = byte_tensor.view(h, w, 3).permute(2, 0, 1).to(dtype=torch.float32) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3,1,1).to(byte_tensor.device)
    std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3,1,1).to(byte_tensor.device)
    return (byte_tensor - mean) / std


# Feature extractor: ResNet50 up to layer3 (output stride 16)
class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        rn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # take layers up to layer3
        self.features = torch.nn.Sequential(
            rn.conv1, rn.bn1, rn.relu, rn.maxpool,
            rn.layer1, rn.layer2, rn.layer3
        )
    def forward(self, x):
        return self.features(x)

@torch.no_grad()
def compute_feature_map(img_pil, extractor, device):
    # manual preprocess
    x = pil_to_tensor(img_pil).unsqueeze(0).to(device)
    fmap = extractor(x)  # [1,C,H,W]
    return fmap.squeeze(0)  # [C,H,W]


def find_max_match_cnn(image_fmap, template_fmap):
    """
    Cross-correlate template_fmap over image_fmap.
    Returns (x, y, w, h, max_val) in feature-map coords.
    """
    tpl = template_fmap.view(1, -1)
    tpl_norm = tpl / tpl.norm(dim=1, keepdim=True)
    C, Ht, Wt = template_fmap.shape
    weight = tpl_norm.view(1, C, Ht, Wt)
    resp = F.conv2d(image_fmap.unsqueeze(0), weight)
    resp = resp.squeeze(0).squeeze(0)
    max_val = resp.max().item()
    idx = resp.argmax().item()
    y, x = divmod(idx, resp.shape[1])
    return x, y, Wt, Ht, max_val


def load_grayscale(path):
    img = Image.open(path).convert('L')
    return np.array(img, dtype=np.float32) / 255.0


def main(templates_dir, images_dir, threshold):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractor = FeatureExtractor().to(device).eval()

    # load template feature maps
    templates = []
    for fn in os.listdir(templates_dir):
        if fn.lower().endswith(('.png','.jpg','.jpeg','.bmp')):
            tpl_img = Image.open(os.path.join(templates_dir, fn)).convert('RGB')
            tpl_fmap = compute_feature_map(tpl_img, extractor, device)
            templates.append((os.path.splitext(fn)[0], tpl_fmap))
    if not templates:
        print(f"No templates found in {templates_dir}")
        sys.exit(1)

    # prepare output folder
    out_dir = os.path.join(images_dir, 'max_match_regions')
    os.makedirs(out_dir, exist_ok=True)

    # process images
    for img_fn in tqdm(os.listdir(images_dir), desc="Images"):
        if not img_fn.lower().endswith(('.png','.jpg','.jpeg','.bmp')): continue
        img_path = os.path.join(images_dir, img_fn)
        try:
            img_pil = Image.open(img_path).convert('RGB')
        except:
            continue
        img_fmap = compute_feature_map(img_pil, extractor, device)
        W_img, H_img = img_pil.size
        _, Hf, Wf = img_fmap.shape
        sx = W_img / Wf
        sy = H_img / Hf

        for tpl_name, tpl_fmap in templates:
            x_f, y_f, w_f, h_f, score = find_max_match_cnn(img_fmap, tpl_fmap)
            if score < threshold:
                continue
            x = int(x_f * sx)
            y = int(y_f * sy)
            w = int(w_f * sx)
            h = int(h_f * sy)
            crop = img_pil.crop((x, y, x + w, y + h))
            base, ext = os.path.splitext(img_fn)
            out_fn = f"{base}__{tpl_name}_region{ext}"
            crop.save(os.path.join(out_dir, out_fn))
            print(f"{img_fn}+{tpl_name}: score={score:.3f}, saved {out_fn}")

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python bib_extract.py <templates_folder> <images_folder> <threshold>")
        sys.exit(1)
    tpl_dir, img_dir, thresh = sys.argv[1], sys.argv[2], sys.argv[3]
    try:
        thresh = float(thresh)
    except ValueError:
        print("Threshold must be a float between 0.0 and 1.0")
        sys.exit(1)
    main(tpl_dir, img_dir, thresh)
