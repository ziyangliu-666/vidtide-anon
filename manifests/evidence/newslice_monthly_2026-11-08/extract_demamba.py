"""DeMamba features + two frozen heads for arbitrary clip lists.
Usage: python extract_demamba.py NAME LISTFILE   (LISTFILE: jsonl with path,label[,group,generator])
Writes data/xfeat/NAME_X.npy (float16), NAME_meta.jsonl (path,label,group,generator,zs,lp)."""
import json, sys, copy
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch, torch.nn as nn
R = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / 'scripts'))
from server.detection.dataset import extract_frames
import ft_extract_features as fe
sys.path.insert(0, str(fe.NSGVD_DIR))
from models.demamba import XCLIP_DeMamba
name, lst = sys.argv[1], sys.argv[2]
items = [json.loads(l) for l in open(lst)]
dev = 'cuda'
model = XCLIP_DeMamba()
st = torch.load(str(fe.CKPT), map_location='cpu', weights_only=False)
model.load_state_dict({k[7:] if k.startswith('module.') else k: v for k, v in st.items()})
zs = copy.deepcopy(model.fc1).to(dev).eval(); model.fc1 = nn.Identity(); model.dropout = nn.Identity(); model.to(dev).eval()
sd = torch.load(R / 'data/ft_features/lp_fc1.pt', map_location='cpu')
lp = nn.Linear(sd['weight'].shape[1], 1); lp.weight.data, lp.bias.data = sd['weight'].float(), sd['bias'].float(); lp.to(dev).eval()
mean, std = fe.IMAGENET_MEAN.to(dev), fe.IMAGENET_STD.to(dev)
out = R / 'data/xfeat'; out.mkdir(exist_ok=True)
def load(it):
    try: return it, extract_frames(Path(it['path']), num_frames=8, resolution=224)
    except Exception: return it, None
feats, meta = [], []
with ThreadPoolExecutor(8) as ex:
    for it, fr in ex.map(load, items):
        if fr is None: continue
        x = torch.from_numpy(fr).float().to(dev).div(255).permute(0, 3, 1, 2).unsqueeze(0)
        with torch.no_grad():
            f = model((x - mean) / std)
            it = dict(it, zs=torch.sigmoid(zs(f)).item(), lp=torch.sigmoid(lp(f.float())).item())
        feats.append(f.cpu().numpy().astype(np.float16).squeeze()); meta.append(it)
np.save(out / f'{name}_X.npy', np.stack(feats))
with open(out / f'{name}_meta.jsonl', 'w') as fh:
    for m in meta: fh.write(json.dumps(m, ensure_ascii=False) + '\n')
print(name, len(meta), 'of', len(items))
