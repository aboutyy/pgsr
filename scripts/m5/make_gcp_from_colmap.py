"""Smoke-test helper ONLY: pick COLMAP sparse points with long tracks and write them as pseudo-GCPs
(3D from points3D.bin, image observations from images.bin). Not real surveyed control."""
import argparse, os, sys, csv, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scene.colmap_loader import read_points3D_binary, read_extrinsics_binary, read_intrinsics_binary
ap = argparse.ArgumentParser()
ap.add_argument("-s", required=True); ap.add_argument("-o", required=True)
ap.add_argument("--n", type=int, default=20); ap.add_argument("--min_track", type=int, default=8)
ap.add_argument("--max_err", type=float, default=0.5); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--sigma_xyz", type=float, default=0.0); ap.add_argument("--sigma_px", type=float, default=1.0)
a = ap.parse_args()
sp = os.path.join(a.s, "sparse/0")
xyz, rgb, err = read_points3D_binary(os.path.join(sp, "points3D.bin")) if False else (None, None, None)
ims = read_extrinsics_binary(os.path.join(sp, "images.bin"))
cams = read_intrinsics_binary(os.path.join(sp, "cameras.bin"))
# build tracks from images.bin
tracks = {}
for im in ims.values():
    for (u, v), pid in zip(im.xys, im.point3D_ids):
        if pid >= 0: tracks.setdefault(int(pid), []).append((im.name, u, v, cams[im.camera_id].width))
# 3D coords: parse points3D.bin directly (loader returns arrays without ids)
import struct, numpy as np
P = {}
with open(os.path.join(sp, "points3D.bin"), "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    for _ in range(n):
        pid, x, y, z = struct.unpack("<Qddd", f.read(32)); f.read(3); e = struct.unpack("<d", f.read(8))[0]
        tl = struct.unpack("<Q", f.read(8))[0]; f.read(8 * tl)
        P[pid] = (x, y, z, e)
cand = [p for p, t in tracks.items() if len(t) >= a.min_track and P[p][3] <= a.max_err]
random.Random(a.seed).shuffle(cand); sel = cand[:a.n]
with open(a.o, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["point_id","X","Y","Z","sigma_xyz","image_name","u","v","sigma_px","image_width"])
    for p in sel:
        for name, u, v, W in tracks[p]:
            w.writerow([p, *P[p][:3], a.sigma_xyz, name, u, v, a.sigma_px, W])
print(f"{len(cand)} candidates, wrote {len(sel)} pseudo-GCPs, {sum(len(tracks[p]) for p in sel)} observations -> {a.o}")
