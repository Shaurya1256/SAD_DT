
# sad_dt.py
SAD-DT: Saliency-aware Anomaly Detection for Digital Twins (starter)
Requirements: torch, torchvision, timm (optional), opencv-python, numpy, matplotlib
Usage (example):
 python src/sad_dt.py --data-dir /path/to/mvtec --category bottle --epochs 30 --batch-size 16

import os
import argparse
from glob import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
import cv2

# ---------------------------
# Simple MVTec dataset loader
# ---------------------------
class MVTecDataset(Dataset):
    def __init__(self, root, category='bottle', split='train', transform=None):
        self.root = root
        self.cat = category
        self.split = split  # 'train' or 'test'
        self.transform = transform
        # train: root/category/train/good/*.png
        # test:  root/category/test/*/*.png
        patterns = [
            os.path.join(root, category, split, '*', '*.png'),
            os.path.join(root, category, split, '*', '*.jpg')
        ]
        self.imgs = []
        for p in patterns:
            self.imgs.extend(glob(p))
        self.imgs = sorted(self.imgs)
        if len(self.imgs) == 0:
            raise FileNotFoundError(f"No images found for {category}/{split} in {root}.")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        p = self.imgs[idx]
        img = Image.open(p).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(p)

# ---------------------------
# Saliency utility (simple, fast)
# Replace with a stronger model (e.g., U2Net) if available
# ---------------------------
def compute_simple_saliency(img_tensor):
    # input: CxHxW torch tensor (0..1)
    img = (img_tensor.clamp(0,1) * 255).byte().cpu().numpy().transpose(1,2,0)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sal = cv2.GaussianBlur(gray, (11,11), 0)
    sal = cv2.Laplacian(sal, cv2.CV_32F)
    sal = np.abs(sal)
    sal = (sal - sal.min())/(sal.max()-sal.min()+1e-9)
    sal = cv2.resize(sal, (img_tensor.shape[2], img_tensor.shape[1]))
    return torch.from_numpy(sal).float().unsqueeze(0)  # 1xHxW

# ---------------------------
# Lightweight attention AE
# ---------------------------
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn = nn.GroupNorm(8, channels)
        self.act = nn.SiLU()
        self.att = nn.Sequential(
            nn.Conv2d(channels, max(1, channels//4), 1),
            nn.ReLU(),
            nn.Conv2d(max(1, channels//4), 1, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        h = self.act(self.gn(self.conv(x)))
        a = self.att(h)
        return x * a + x

class Encoder(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(base, base*2, 4, stride=2, padding=1),
            nn.ReLU(),
            AttentionBlock(base*2),
            nn.Conv2d(base*2, base*4, 4, stride=2, padding=1),
            nn.ReLU(),
            AttentionBlock(base*4),
        )
    def forward(self, x):
        return self.enc(x)

class Decoder(nn.Module):
    def __init__(self, out_ch=3, base=32):
        super().__init__()
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(base*4, base*2, 4, stride=2, padding=1),
            nn.ReLU(),
            AttentionBlock(base*2),
            nn.ConvTranspose2d(base*2, base, 4, stride=2, padding=1),
            nn.ReLU(),
            AttentionBlock(base),
            nn.Conv2d(base, out_ch, 3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, z):
        return self.dec(z)

class SAD_AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.dec = Decoder()
    def forward(self, x):
        z = self.enc(x)
        xrec = self.dec(z)
        return xrec

# ---------------------------
# Feature extractor (pretrained)
# ---------------------------
def get_feature_extractor(device):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    modules = list(model.children())[:-2]  # keep down to CxHxW feature map
    feat = nn.Sequential(*modules)
    feat.eval()
    for p in feat.parameters():
        p.requires_grad = False
    return feat.to(device)

# ---------------------------
# Training & loss
# ---------------------------
def saliency_weighted_mse(x, xrec, sal):
    # x, xrec: BxCxHxW ; sal: Bx1xHxW
    diff = (x - xrec) ** 2
    w = 1.0 + 4.0 * sal  # baseline weight + extra
    w = w.expand_as(diff)
    return (diff * w).mean()

def feature_loss(featnet, x, xrec, sal):
    fx = featnet(x)
    frec = featnet(xrec)
    b,c,h,w = fx.shape
    sal_resized = nn.functional.interpolate(sal, size=(h,w), mode='bilinear', align_corners=False)
    diff = (fx - frec).pow(2).sum(1, keepdim=True)
    w = 1.0 + 4.0 * sal_resized
    w = w.expand_as(diff)
    return (diff * w).mean()

def train_epoch(model, loader, device, optimizer, featnet):
    model.train()
    running = 0.0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        # saliency per image
        sals = []
        for i in range(imgs.size(0)):
            s = compute_simple_saliency(imgs[i])
            sals.append(s)
        sal = torch.stack(sals).to(device)  # Bx1xHxW
        optimizer.zero_grad()
        rec = model(imgs)
        lrec = saliency_weighted_mse(imgs, rec, sal)
        lfeat = feature_loss(featnet, imgs, rec, sal)
        loss = lrec + 0.5 * lfeat
        loss.backward()
        optimizer.step()
        running += loss.item() * imgs.size(0)
    return running / len(loader.dataset)

# ---------------------------
# Evaluate & visualize
# ---------------------------
def compute_anomaly_map(model, featnet, img, device):
    model.eval()
    with torch.no_grad():
        x = img.to(device).unsqueeze(0)
        rec = model(x)
        sal = compute_simple_saliency(x[0]).to(device).unsqueeze(0)  # 1x1xHxW
        # pixel residual
        res = (x - rec).abs().mean(1, keepdim=True)  # 1x1xHxW
        # feature residual
        fx = featnet(x)
        frec = featnet(rec)
        fdiff = (fx - frec).pow(2).sum(1, keepdim=True)
        # resize to image size
        fdiff_resized = nn.functional.interpolate(fdiff, size=(res.shape[2], res.shape[3]), mode='bilinear', align_corners=False)
        amap = res + 0.5 * fdiff_resized
        # saliency weighting
        amap = amap * (1.0 + 4.0 * sal)
        amap = amap.squeeze().cpu().numpy()
        amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-9)
    return amap, rec.squeeze().cpu().numpy()

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = T.Compose([T.Resize((256,256)), T.ToTensor()])
    train_ds = MVTecDataset(args.data_dir, category=args.category, split='train', transform=transform)
    test_ds = MVTecDataset(args.data_dir, category=args.category, split='test', transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=1)

    model = SAD_AE().to(device)
    featnet = get_feature_extractor(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, device, optimizer, featnet)
        print(f"Epoch {epoch+1}/{args.epochs}  loss {loss:.4f}")

        if (epoch+1) % args.log_interval == 0:
            # quick viz on the first test image
            img, name = test_ds[0]
            amap, rec = compute_anomaly_map(model, featnet, img, device)
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10,4))
            plt.subplot(1,3,1); plt.title('Input'); plt.imshow(img.permute(1,2,0).cpu().numpy()); plt.axis('off')
            plt.subplot(1,3,2); plt.title('Reconstruction'); plt.imshow(np.transpose(rec, (1,2,0))); plt.axis('off')
            plt.subplot(1,3,3); plt.title('Anomaly Map'); plt.imshow(amap, cmap='jet'); plt.axis('off')
            out = f'viz_epoch_{epoch+1}.png'
            plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
            print(f"Saved {out}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True, help='Root of MVTec AD')
    parser.add_argument('--category', type=str, default='bottle')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--log-interval', type=int, default=5)
    args = parser.parse_args()
    main(args)
