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
    return se3_exp_batch(xi[None])[0]


def _se3_exp_old(xi):
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


def se3_exp_batch(xi):
    """xi [N,6] = (rho, omega) -> [N,4,4]. Branch-free (no host sync)."""
    rho, omega = xi[:, :3], xi[:, 3:]
    th2 = (omega * omega).sum(-1)
    small = th2 < 1e-10
    th2s = torch.where(small, torch.ones_like(th2), th2)
    ths = torch.sqrt(th2s)
    A = torch.where(small, 1.0 - th2 / 6.0, torch.sin(ths) / ths)
    B = torch.where(small, 0.5 - th2 / 24.0, (1 - torch.cos(ths)) / th2s)
    C = torch.where(small, 1.0 / 6.0 - th2 / 120.0, (ths - torch.sin(ths)) / (th2s * ths))
    K = skew(omega); KK = K @ K
    I = torch.eye(3, device=xi.device, dtype=xi.dtype).expand_as(K)
    R = I + A[:, None, None] * K + B[:, None, None] * KK
    V = I + B[:, None, None] * K + C[:, None, None] * KK
    T = torch.zeros(xi.shape[0], 4, 4, device=xi.device, dtype=xi.dtype)
    T[:, :3, :3] = R; T[:, :3, 3] = (V @ rho[:, :, None])[:, :, 0]; T[:, 3, 3] = 1.0
    return T


class GCPTerm:
    """GCP reprojection term (vectorised over all observations).
    CSV columns: point_id,X,Y,Z,sigma_xyz,image_name,u,v,sigma_px,image_width
      X,Y,Z / sigma_xyz in the scene world frame & units (same frame as the camera poses);
      u,v / sigma_px in pixels of an image `image_width` px wide (rescaled to the training resolution).
    Each residual is whitened by sigma_tot^2 = sigma_px^2 + (f * sigma_xyz / z)^2, then a Huber loss is applied.
    """
    def __init__(self, path, cams, huber_delta=3.0):
        by_name = {c.image_name: c for c in cams}
        rows, skipped = [], 0
        with open(path) as f:
            for r in csv.DictReader(f):
                name = r["image_name"].rsplit(".", 1)[0]
                cam = by_name.get(name) or by_name.get(r["image_name"])
                if cam is None:
                    skipped += 1; continue
                rows.append((cam, r))
        self.cams = list({id(c): c for c, _ in rows}.values())
        cidx = {id(c): i for i, c in enumerate(self.cams)}
        dev = "cuda"
        def t(v): return torch.tensor(v, device=dev, dtype=torch.float32)
        sc = [c.image_width / float(r["image_width"]) for c, r in rows]
        self.ci = torch.tensor([cidx[id(c)] for c, _ in rows], device=dev, dtype=torch.long)
        self.X = t([[float(r["X"]), float(r["Y"]), float(r["Z"])] for _, r in rows])
        self.sx = t([float(r["sigma_xyz"]) for _, r in rows])
        self.uv = t([[float(r["u"]) * s, float(r["v"]) * s] for (_, r), s in zip(rows, sc)])
        self.spx = t([float(r["sigma_px"]) * s for (_, r), s in zip(rows, sc)])
        self.F = t([[c.Fx, c.Fy] for c in self.cams]); self.Cc = t([[c.Cx, c.Cy] for c in self.cams])
        self.obs = rows  # kept for len()/compat
        self.points = sorted({r["point_id"] for _, r in rows})
        self.delta = huber_delta
        print(f"[GCP] {len(rows)} observations of {len(self.points)} points in {len(self.cams)} images loaded, {skipped} skipped (image not in train set)")

    def cam_w2c(self):
        base = torch.stack([c.base_w2c for c in self.cams])
        if all(c.pose_delta is None for c in self.cams):
            return base
        d = torch.stack([c.pose_delta if c.pose_delta is not None else torch.zeros(6, device="cuda") for c in self.cams])
        return se3_exp_batch(d) @ base

    def _proj(self):
        W = self.cam_w2c()[self.ci]
        Xc = (W[:, :3, :3] @ self.X[:, :, None])[:, :, 0] + W[:, :3, 3]
        z = Xc[:, 2:3].clamp(min=1e-6)
        F, C = self.F[self.ci], self.Cc[self.ci]
        return F * Xc[:, :2] / z + C, z[:, 0]

    def residuals(self):
        proj, z = self._proj()
        sig = torch.sqrt(self.spx ** 2 + (self.F[self.ci, 0] * self.sx / z) ** 2)
        return (proj - self.uv) / sig[:, None]

    def loss(self):
        r = self.residuals()
        return F.huber_loss(r, torch.zeros_like(r), delta=self.delta)

    def refresh(self):
        """Batched equivalent of cam.refresh_pose(sync_numpy=False) for all GCP cameras."""
        with torch.no_grad():
            W = self.cam_w2c().detach()
            WT = W.transpose(1, 2).contiguous()
            P = torch.stack([c.projection_matrix for c in self.cams])
            FP = torch.bmm(WT, P)
            Cn = torch.inverse(WT)[:, 3, :3]
            for i, c in enumerate(self.cams):
                c.world_view_transform = WT[i]; c.full_proj_transform = FP[i]; c.camera_center = Cn[i]
                c._RT_stale = True

    def rms_px(self):
        with torch.no_grad():
            proj, _ = self._proj()
            return float(((proj - self.uv) ** 2).sum(-1).mean().sqrt())


def quaternion_to_matrix(q):
    """Drop-in for pytorch3d.transforms.quaternion_to_matrix (real part first)."""
    r, i, j, k = torch.unbind(q, -1)
    two_s = 2.0 / (q * q).sum(-1)
    o = torch.stack((1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
                     two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
                     two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j)), -1)
    return o.reshape(q.shape[:-1] + (3, 3))
