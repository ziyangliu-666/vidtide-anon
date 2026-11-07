"""Download a small balanced sample from the new metadata slices and standardize it like M0:
short-side-480 H.264 (non-login Bilibili cap; the M0 reference is re-encoded identically), then the same fixed 16 s window (start = min(10% of duration, dur-16)) used by the
transcode-ladder 'orig' reference. Groups: early = 2026-05..06 (M1-M2), late = 2026-09..11 (M5-M7)."""
import json, random, subprocess, time, hashlib, sys, os
from pathlib import Path
R = Path(__file__).resolve().parents[3]; OUT = R / 'data/newslice_clips'
SCALE = "scale='if(lt(iw,ih),min(480,iw),-2)':'if(lt(iw,ih),-2,min(480,ih))'"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 75
env = dict(l.strip().split('=', 1) for l in open(R / 'data/bilibili_cookies.env') if '=' in l)
ck = OUT / 'cookies.txt'
with open(ck, 'w') as f:
    f.write('# Netscape HTTP Cookie File\n')
    for k, n in [('BILIBILI_SESSDATA', 'SESSDATA'), ('BILIBILI_BILI_JCT', 'bili_jct'),
                 ('BILIBILI_DEDEUSERID', 'DedeUserID'), ('BILIBILI_BUVID3', 'buvid3')]:
        if env.get(k): f.write(f".bilibili.com\tTRUE\t/\tFALSE\t1893456000\t{n}\t{env[k].strip()}\n")
os.chmod(ck, 0o600)
rows = []
for f in [str(R / 'data/meta/ai_slices.jsonl'), str(R / 'data/meta/generic_slices.jsonl')]:
    for l in open(f):
        r = json.loads(l)
        if r.get('_type') or r['label'] not in ('fake', 'real'): continue
        if not r.get('duration_sec') or r['duration_sec'] < 5: continue
        rows.append(r)
grp = {'2026-05': 'early', '2026-06': 'early', '2026-09': 'late', '2026-10': 'late', '2026-11': 'late'}
rng = random.Random(20261107)
pick = []
for g in ['late', 'early']:
    for lab in ['fake', 'real']:
        c = [r for r in rows if grp.get((r['published_at'] or '')[:7]) == g and r['label'] == lab]
        rng.shuffle(c); pick += [(g, r) for r in c[:N]]
man = OUT / 'manifest.jsonl'
done = {json.loads(l)['source_id'] for l in open(man)} if man.exists() else set()
K, NS = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
for j, (g, r) in enumerate(pick):
    sid = r['source_id']
    if sid in done or j % NS != K: continue
    vid = hashlib.sha1(f'bilibili:{sid}'.encode()).hexdigest()[:32]
    raw = OUT / 'raw' / f'{vid}.mp4'; std = OUT / 'std' / r['label'] / f'{vid}.mp4'
    raw.parent.mkdir(parents=True, exist_ok=True); std.parent.mkdir(parents=True, exist_ok=True)
    status = 'ok'
    if not raw.exists():
        p = subprocess.run(['yt-dlp', '-q', '--no-warnings', '--cookies', str(ck), '-f', 'bv*/b',
                            '-S', 'res:480,vcodec:h264',
                            '--no-playlist', '-o', str(raw), r['source_url']],
                           capture_output=True, text=True, timeout=300)
        if p.returncode != 0 or not raw.exists():
            status = 'download_failed:' + (p.stderr.strip()[-120:])
    if status == 'ok':
        dur = float(r['duration_sec']); start = 0.0 if dur <= 16 else min(dur * 0.10, dur - 16)
        p = subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', f'{start:.3f}', '-i', str(raw),
                            '-t', '16', '-vf', SCALE,
                            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-an',
                            '-movflags', '+faststart', str(std)], capture_output=True, text=True, timeout=300)
        if p.returncode != 0 or not std.exists(): status = 'ffmpeg_failed'
        else: raw.unlink(missing_ok=True)  # keep only the standardized window
    with open(man, 'a') as f:
        f.write(json.dumps({'video': vid, 'source_id': sid, 'label': 1 if r['label'] == 'fake' else 0,
                            'group': g, 'published_at': r['published_at'], 'status': status}) + '\n')
    print(g, r['label'], sid, status, flush=True)
    time.sleep(6)
