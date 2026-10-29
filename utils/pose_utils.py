# M5: per-camera se(3) pose refinement and GCP reprojection term for PGSR.
# Convention: W2C is a 4x4 world->camera matrix in column-vector form (x_c = W2C @ x_w).
# PGSR stores camera.world_view_transform = W2C^T (row-vector form).
import math, csv
import numpy as np
import torch
import torch.nn.functional as F


def skew(w):
    wx, wy, wz = w.unbind(-1)
    z = torch.zeros_like(wx)
    return torch.stack([torch.stack([z, -wz, wy], -1),
                        torch.stack([wz, z, -wx], -1),
                        torch.stack([-wy, wx, z], -1)], -2)


def se3_exp(xi):
    """xi = (rho[3], omega[3]) -> 4x4 SE(3). Taylor-safe near zero."""
    rho, omega = xi[:3], xi[3:]
    th2 = (omega * omega).sum()
    th = torch.sqrt(th2 + 1e-12)
    K = skew(omega)
    I = torch.eye(3, device=xi.device, dtype=xi.dtype)
    if th2 < 1e-10:
        A, B, C = 1.0 - th2 / 6.0, 0.5 - th2 / 24.0, 1.0 / 6.0 - th2 / 120.0
    else:
        A, B, C = torch.sin(th) / th, (1 - torch.cos(th)) / th2, (th - torch.sin(th)) / (th2 * th)
    R = I + A * K + B * (K @ K)
    V = I + B * K + C * (K @ K)
    T = torch.eye(4, device=xi.device, dtype=xi.dtype)
    T = torch.cat([torch.cat([R, (V @ rho)[:, None]], 1), T[3:]], 0)
    return T


def quat_from_R_near_identity(R):
    """Quaternion (w,x,y,z) of a rotation that is numerically ~I (value exact at I, first-order grads)."""
    q = torch.stack([1.0 + R[0, 0] + R[1, 1] + R[2, 2],
                     R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return q / q.norm()


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw], -1)


def rot_err_deg(Ra, Rb):
    c = ((Ra.T @ Rb).trace() - 1) / 2
    return math.degrees(math.acos(float(max(-1.0, min(1.0, c)))))


def perturb_cameras(cams, rot_deg, trans, seed):
    """Left-multiply each camera's W2C with a random SE(3) (rotation angle rot_deg, translation norm trans)."""
    g = np.random.default_rng(seed)
    for cam in cams:
        ax = g.normal(size=3); ax /= np.linalg.norm(ax)
        tv = g.normal(size=3); tv /= np.linalg.norm(tv)
        xi = torch.tensor(np.concatenate([tv * trans, ax * math.radians(rot_deg)]), dtype=torch.float32, device="cuda")
        cam.set_base_w2c(se3_exp(xi) @ cam.base_w2c)


def pose_errors(cams):
    """Per-camera error of current effective pose vs. stored ground truth: (rot deg, camera-center distance)."""
    re, te = [], []
    with torch.no_grad():
        for cam in cams:
            W = cam.effective_w2c(); G = cam.gt_w2c
            re.append(rot_err_deg(W[:3, :3].cpu().double(), G[:3, :3].cpu().double()))
            cw = -W[:3, :3].T @ W[:3, 3]; cg = -G[:3, :3].T @ G[:3, 3]
            te.append(float((cw - cg).norm()))
    re, te = np.array(re), np.array(te)
    return dict(rot_mean_deg=re.mean(), rot_median_deg=float(np.median(re)), rot_max_deg=re.max(),
                center_mean=te.mean(), center_median=float(np.median(te)), center_max=te.max(), n=len(re))


class GCPTerm:
    """GCP reprojection term.
    CSV columns: point_id,X,Y,Z,sigma_xyz,image_name,u,v,sigma_px,image_width
      X,Y,Z / sigma_xyz in the scene world frame & units (same frame as the camera poses);
      u,v / sigma_px in pixels of an image `image_width` px wide (rescaled to the training resolution).
    Each residual is whitened by sigma_tot^2 = sigma_px^2 + (f * sigma_xyz / z)^2, then a Huber loss is applied.
    """
    def __init__(self, path, cams, huber_delta=3.0):
        by_name = {c.image_name: c for c in cams}
        self.obs, skipped = [], 0
        with open(path) as f:
            for r in csv.DictReader(f):
                name = r["image_name"].rsplit(".", 1)[0]
                cam = by_name.get(name) or by_name.get(r["image_name"])
                if cam is None:
                    skipped += 1; continue
                s = cam.image_width / float(r["image_width"])
                self.obs.append((cam, [float(r["X"]), float(r["Y"]), float(r["Z"])], float(r["sigma_xyz"]),
                                 [float(r["u"]) * s, float(r["v"]) * s], float(r["sigma_px"]) * s, r["point_id"]))
        self.points = sorted({o[5] for o in self.obs})
        self.delta = huber_delta
        print(f"[GCP] {len(self.obs)} observations of {len(self.points)} points loaded, {skipped} skipped (image not in train set)")

    def residuals(self):
        out = []
        for cam, X, sx, uv, spx, _ in self.obs:
            W = cam.effective_w2c()
            Xc = W[:3, :3] @ torch.tensor(X, device="cuda") + W[:3, 3]
            z = Xc[2].clamp(min=1e-6)
            proj = torch.stack([cam.Fx * Xc[0] / z + cam.Cx, cam.Fy * Xc[1] / z + cam.Cy])
            sig = torch.sqrt(spx ** 2 + (cam.Fx * sx / z) ** 2)
            out.append((proj - torch.tensor(uv, device="cuda")) / sig)
        return torch.stack(out)

    def loss(self):
        r = self.residuals()
        return F.huber_loss(r, torch.zeros_like(r), delta=self.delta)

    def rms_px(self):
        with torch.no_grad():
            r = []
            for cam, X, sx, uv, spx, _ in self.obs:
                W = cam.effective_w2c()
                Xc = W[:3, :3] @ torch.tensor(X, device="cuda") + W[:3, 3]
                p = torch.stack([cam.Fx * Xc[0] / Xc[2] + cam.Cx, cam.Fy * Xc[1] / Xc[2] + cam.Cy])
                r.append(float((p - torch.tensor(uv, device="cuda")).norm()))
            return float(np.sqrt(np.mean(np.square(r))))


def quaternion_to_matrix(q):
    """Drop-in for pytorch3d.transforms.quaternion_to_matrix (real part first)."""
    r, i, j, k = torch.unbind(q, -1)
    two_s = 2.0 / (q * q).sum(-1)
    o = torch.stack((1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
                     two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
                     two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j)), -1)
    return o.reshape(q.shape[:-1] + (3, 3))
