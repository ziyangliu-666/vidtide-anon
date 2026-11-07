"""E1: frozen detectors on the new slices. DeMamba-Pika (GenVideo-trained, zero-shot) and the
VidTide-M0 DeMamba linear probe (lp_fc1.pt, trained on M0 only) score, with one forward pass:
  m0ref : 800 M0-test clips (16 s window), re-encoded to short-side 480 H.264
  early : M1-M2 (May-Jun 2026) clips, same window + encode
  late  : M5-M7 (Sep-Nov 2026) clips, same window + encode
Frames use the original extract_frames (8 x 224) so the probe sees its training preprocessing."""
import json, sys, copy
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch, torch.nn as nn
R = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / 'scripts'))
from server.detection.dataset import extract_frames
import ft_extract_features as fe
from sklearn.metrics import roc_auc_score
dev = 'cuda'
m = fe.build_model.__wrapped__(dev) if hasattr(fe.build_model, '__wrapped__') else None
# build with original head kept
sys.path.insert(0, str(fe.NSGVD_DIR))
from models.demamba import XCLIP_DeMamba
model = XCLIP_DeMamba()
st = torch.load(str(fe.CKPT), map_location='cpu', weights_only=False)
st = {k[7:] if k.startswith('module.') else k: v for k, v in st.items()}
model.load_state_dict(st)
zs_head = copy.deepcopy(model.fc1).to(dev).eval()
model.fc1 = nn.Identity(); model.dropout = nn.Identity(); model.to(dev).eval()
sd = torch.load(R / 'data/ft_features/lp_fc1.pt', map_location='cpu')
lp = nn.Linear(sd['weight'].shape[1], 1); lp.weight.data, lp.bias.data = sd['weight'].float(), sd['bias'].float(); lp.to(dev).eval()
mean, std = fe.IMAGENET_MEAN.to(dev), fe.IMAGENET_STD.to(dev)
items = []
for lab in ['real', 'fake']:
    for p in sorted((R / 'data/newslice_clips/m0ref' / lab).glob('*.mp4')):
        items.append(('m0ref', p, int(lab == 'fake')))
grp = {}
for l in open(R / 'data/newslice_clips/manifest.jsonl'):
    r = json.loads(l)
    if r['status'] == 'ok': grp[r['video']] = r['group']
for lab in ['real', 'fake']:
    for p in sorted((R / 'data/newslice_clips/std' / lab).glob('*.mp4')):
        if p.stem in grp: items.append((grp[p.stem], p, int(lab == 'fake')))
def load(it):
    try: return it, extract_frames(it[1], num_frames=8, resolution=224)
    except Exception as e: return it, None
rows = []
with ThreadPoolExecutor(8) as ex:
    for (g, p, y), fr in ex.map(load, items):
        if fr is None: continue
        x = torch.from_numpy(fr).float().to(dev).div(255).permute(0, 3, 1, 2).unsqueeze(0)
        x = (x - mean) / std
        with torch.no_grad():
            f = model(x); zs = torch.sigmoid(zs_head(f)).item(); s = torch.sigmoid(lp(f.float())).item()
        rows.append(dict(group=g, video=p.stem, label=y, demamba_pika_zs=zs, vidtide_lp=s))
json.dump(rows, open(R / 'results/iclr_e1_newslice_scores.json', 'w'))
rng = np.random.default_rng(0); out = {}
for g in ['m0ref', 'early', 'late']:
    rs = [r for r in rows if r['group'] == g]
    if not rs or len({r['label'] for r in rs}) < 2: continue
    y = np.array([r['label'] for r in rs]); out[g] = {'n_fake': int(y.sum()), 'n_real': int((1 - y).sum())}
    for k in ['demamba_pika_zs', 'vidtide_lp']:
        s = np.array([r[k] for r in rs]); a = roc_auc_score(y, s); bs = []
        for _ in range(2000):
            i = rng.integers(0, len(y), len(y))
            if 0 < y[i].sum() < len(i): bs.append(roc_auc_score(y[i], s[i]))
        out[g][k] = dict(auroc=a, ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                         mean_fake=float(s[y == 1].mean()), mean_real=float(s[y == 0].mean()))
    print(g, out[g])
json.dump(out, open(R / 'results/iclr_e1_newslice_eval.json', 'w'), indent=1)
