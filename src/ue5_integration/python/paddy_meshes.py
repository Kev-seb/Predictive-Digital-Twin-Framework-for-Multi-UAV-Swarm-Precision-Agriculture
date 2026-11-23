"""
paddy_meshes.py - write the procedural meshes for the UE5 paddy twin as OBJ files.

Runs with plain Python (no UE). paddy_scene.py imports the results into /Game/TwinDemo/Meshes.
Units are centimetres, Z up, X forward (UE convention after import with the default OBJ axes
being converted by Interchange; paddy_scene.py fixes orientation on import).

  SM_RiceCell   one 5 m x 5 m paddy cell: 12 x 12 rice hills, 1 m tall at scale 1
                (pivot at the cell centre, ground level). UV.y = 0 at the base, 1 at leaf tips.
  SM_WaterTile  flat 5 m x 5 m quad (pivot centre)
  SM_DroneBody  agricultural spray quadcopter body: hub, tank, arms, motors, skids, nozzles (~2.4 m span)
  SM_Rotor      two-blade propeller, ~0.9 m, pivot at its hub
  SM_Palm       coconut palm, ~9 m
  SM_Shrub      low bush, ~1.5 m
"""
import math
import os
import random
import sys


class Obj:
    def __init__(self):
        self.v, self.vt, self.vn, self.f = [], [], [], []

    def vert(self, p, uv, n=(0.0, 0.0, 1.0)):
        self.v.append(p); self.vt.append(uv); self.vn.append(n)
        return len(self.v)          # 1-based

    def tri(self, a, b, c):
        self.f.append((a, b, c))

    def quad(self, a, b, c, d):
        self.tri(a, b, c); self.tri(a, c, d)

    def write(self, path):
        with open(path, "w") as fh:
            fh.write("# paddy_meshes.py\n")
            for x, y, z in self.v:
                # UE 5.8's OBJ import maps OBJ (x, y, z) to UE (x, -y, z): write -y so UE gets (x, y, z)
                fh.write(f"v {x:.3f} {-y:.3f} {z:.3f}\n")
            for u, w in self.vt:
                fh.write(f"vt {u:.4f} {w:.4f}\n")
            for x, y, z in self.vn:
                fh.write(f"vn {x:.4f} {-y:.4f} {z:.4f}\n")
            for a, b, c in self.f:
                fh.write(f"f {a}/{a}/{a} {c}/{c}/{c} {b}/{b}/{b}\n")


def _norm(v):
    l = math.sqrt(sum(x * x for x in v)) or 1.0
    return tuple(x / l for x in v)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def blade(o, base, yaw, lean, length, width, segs=4, droop=0.35, u0=0.0):
    """A tapered, arching leaf. lean = outward angle at the base (rad), droop bends the tip down."""
    dx, dy = math.cos(yaw), math.sin(yaw)
    side = (-dy, dx, 0.0)
    rows = []
    for i in range(segs + 1):
        t = i / segs
        ang = lean + droop * t * t                    # angle from vertical grows toward the tip
        # integrate the arc
        rows.append((t, ang))
    pts, p = [], list(base)
    step = length / segs
    for i, (t, ang) in enumerate(rows):
        if i > 0:
            a = rows[i - 1][1]
            p = [p[0] + dx * math.sin(a) * step, p[1] + dy * math.sin(a) * step, p[2] + math.cos(a) * step]
        pts.append((tuple(p), t, ang))
    idx = []
    for (pp, t, ang) in pts:
        w = width * (1.0 - 0.85 * t) * 0.5
        fwd = (dx * math.sin(ang), dy * math.sin(ang), math.cos(ang))
        n = _norm(_cross(side, fwd))
        l = o.vert((pp[0] - side[0] * w, pp[1] - side[1] * w, pp[2]), (u0, t), n)
        r = o.vert((pp[0] + side[0] * w, pp[1] + side[1] * w, pp[2]), (u0 + 0.1, t), n)
        idx.append((l, r))
    for i in range(segs):
        (l0, r0), (l1, r1) = idx[i], idx[i + 1]
        o.quad(l0, r0, r1, l1)


def rice_cell(path, n=20, size=500.0, height=100.0, seed=3, leaves_range=(12, 16)):
    rnd = random.Random(seed)
    o = Obj()
    sp = size / n
    for i in range(n):
        for j in range(n):
            cx = -size / 2 + (i + 0.5) * sp + rnd.uniform(-3, 3)
            cy = -size / 2 + (j + 0.5) * sp + rnd.uniform(-3, 3)
            hh = height * rnd.uniform(0.85, 1.05)
            leaves = rnd.randint(*leaves_range)   # a hill = many tillers; each shows a few leaves
            for k in range(leaves):
                yaw = (k / leaves) * 2 * math.pi + rnd.uniform(-0.35, 0.35)
                inner = k % 3 == 0
                lean = rnd.uniform(0.06, 0.22) if inner else rnd.uniform(0.25, 0.6)
                length = hh * (rnd.uniform(0.9, 1.0) if inner else rnd.uniform(0.65, 0.9))
                base = (cx + math.cos(yaw) * 2.0, cy + math.sin(yaw) * 2.0, -4.0)   # roots under the water
                blade(o, base, yaw, lean, length / max(0.55, math.cos(lean)) * 0.95, rnd.uniform(2.3, 3.2),
                      segs=3, droop=rnd.uniform(0.3, 0.75), u0=rnd.random() * 0.9)
    o.write(path)
    return len(o.f)


def box(o, c, h, rot=0.0, u=(0, 0)):
    """Axis box at centre c with half extents h, rotated about Z by rot (rad)."""
    cr, sr = math.cos(rot), math.sin(rot)
    def P(x, y, z):
        return (c[0] + x * cr - y * sr, c[1] + x * sr + y * cr, c[2] + z)
    hx, hy, hz = h
    faces = [((1, 0, 0), [(hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz), (hx, -hy, hz)]),
             ((-1, 0, 0), [(-hx, hy, -hz), (-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz)]),
             ((0, 1, 0), [(hx, hy, -hz), (-hx, hy, -hz), (-hx, hy, hz), (hx, hy, hz)]),
             ((0, -1, 0), [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz)]),
             ((0, 0, 1), [(hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz), (-hx, -hy, hz)]),
             ((0, 0, -1), [(hx, hy, -hz), (hx, -hy, -hz), (-hx, -hy, -hz), (-hx, hy, -hz)])]
    for n, q in faces:
        nn = (n[0] * cr - n[1] * sr, n[0] * sr + n[1] * cr, n[2])
        ids = [o.vert(P(*p), (u[0] + (k in (1, 2)) * 0.05, u[1] + (k >= 2) * 0.05), nn) for k, p in enumerate(q)]
        o.quad(*ids)


def cylinder(o, c, r, h, segs=16, r_top=None, u=(0, 0), caps=True, axis_dir=(0, 0, 1)):
    """Vertical cylinder (or cone frustum) from c (base centre) up h."""
    r_top = r if r_top is None else r_top
    ring0, ring1 = [], []
    for i in range(segs + 1):
        a = 2 * math.pi * i / segs
        ca, sa = math.cos(a), math.sin(a)
        n = _norm((ca * h, sa * h, r - r_top))
        ring0.append(o.vert((c[0] + ca * r, c[1] + sa * r, c[2]), (u[0] + i / segs * 0.05, u[1]), n))
        ring1.append(o.vert((c[0] + ca * r_top, c[1] + sa * r_top, c[2] + h), (u[0] + i / segs * 0.05, u[1] + 0.05), n))
    for i in range(segs):
        o.quad(ring0[i], ring0[i + 1], ring1[i + 1], ring1[i])
    if caps:
        for z, rr, ring_n in ((c[2] + h, r_top, 1), (c[2], r, -1)):
            ctr = o.vert((c[0], c[1], z), u, (0, 0, ring_n))
            ids = [o.vert((c[0] + math.cos(2 * math.pi * i / segs) * rr, c[1] + math.sin(2 * math.pi * i / segs) * rr, z),
                          u, (0, 0, ring_n)) for i in range(segs + 1)]
            for i in range(segs):
                if ring_n > 0:
                    o.tri(ctr, ids[i], ids[i + 1])
                else:
                    o.tri(ctr, ids[i + 1], ids[i])


def sphere(o, c, r, segs=16, rings=10, squash=1.0, u=(0, 0)):
    grid = []
    for j in range(rings + 1):
        phi = math.pi * j / rings
        row = []
        for i in range(segs + 1):
            th = 2 * math.pi * i / segs
            n = (math.sin(phi) * math.cos(th), math.sin(phi) * math.sin(th), math.cos(phi))
            row.append(o.vert((c[0] + n[0] * r, c[1] + n[1] * r, c[2] + n[2] * r * squash), (u[0] + i / segs * 0.05, u[1] + j / rings * 0.05), n))
        grid.append(row)
    for j in range(rings):
        for i in range(segs):
            o.quad(grid[j][i], grid[j + 1][i], grid[j + 1][i + 1], grid[j][i + 1])


# UV islands so one material can colour parts: body uses UV.x bands
U_BODY, U_DARK, U_TANK, U_ACCENT = (0.05, 0.05), (0.30, 0.05), (0.55, 0.05), (0.80, 0.05)


def drone_body(path):
    o = Obj()
    span = 120.0                                   # hub centre to motor (cm) -> ~2.4 m tip-to-tip incl. props
    # central hub and canopy
    box(o, (0, 0, 0), (32, 26, 9), u=U_BODY)
    sphere(o, (6, 0, 9), 24, squash=0.45, u=U_BODY)
    # spray tank under the hub
    cylinder(o, (0, 0, -42), 20, 34, u=U_TANK)
    # landing gear: two skids + struts
    for s in (-1, 1):
        box(o, (0, s * 34, -62), (48, 2.5, 2.5), u=U_DARK)
        for x in (-28, 28):
            box(o, (x, s * 30, -36), (2.0, 2.0, 25), u=U_DARK)
    # arms + motors
    for k in range(4):
        a = math.radians(45 + 90 * k)
        mx, my = math.cos(a) * span, math.sin(a) * span
        box(o, (mx / 2, my / 2, 2), (span / 2, 3.2, 3.2), rot=a, u=U_DARK)
        cylinder(o, (mx, my, 0), 7, 12, u=U_DARK)
        cylinder(o, (mx, my, 12), 2.5, 4, u=U_ACCENT)         # shaft
        # spray nozzle under each motor
        cylinder(o, (mx, my, -14), 2.2, 14, u=U_ACCENT)
    # spray boom across the front
    box(o, (0, 0, -30), (3, 70, 2), u=U_DARK)
    o.write(path)
    return len(o.f)


def rotor(path):
    o = Obj()
    cylinder(o, (0, 0, -2), 4, 4, u=U_DARK)
    for s in (0, math.pi):
        # slightly twisted blade: two boxes
        box(o, (math.cos(s) * 24, math.sin(s) * 24, 0), (22, 3.8, 0.6), rot=s, u=U_DARK)
        box(o, (math.cos(s) * 45, math.sin(s) * 45, 0.3), (2.0, 2.4, 0.4), rot=s, u=U_ACCENT)   # tip mark
    o.write(path)
    return len(o.f)


def palm(path, seed=11):
    rnd = random.Random(seed)
    o = Obj()
    # curved trunk from stacked frustums
    h, segs, lean = 850.0, 10, rnd.uniform(0.05, 0.12)
    x = 0.0
    for i in range(segs):
        z = h * i / segs
        r0 = 22 - 8 * i / segs
        r1 = 22 - 8 * (i + 1) / segs
        cylinder(o, (x, 0, z), r0, h / segs + 2, segs=10, r_top=r1, u=(0.55, 0.05), caps=False)
        x += math.sin(lean * (1 + i / segs)) * h / segs
    top = (x, 0, h)
    sphere(o, top, 26, segs=10, rings=6, u=(0.55, 0.05))
    # fronds: long arched leaves with leaflets
    for k in range(14):
        yaw = 2 * math.pi * k / 14 + rnd.uniform(-0.15, 0.15)
        dx, dy = math.cos(yaw), math.sin(yaw)
        L = rnd.uniform(380, 480)
        n = 10
        prev = None
        for i in range(n + 1):
            t = i / n
            ang = math.radians(rnd.uniform(-5, 5) + 20 + 95 * t * t)     # up then drooping
            px = top[0] + dx * L * t * 0.9
            py = top[1] + dy * L * t * 0.9
            pz = top[2] + L * 0.35 * math.sin(math.radians(60)) * t - L * 0.55 * t * t
            w = 60 * math.sin(math.pi * min(1.0, t * 1.1 + 0.05))
            sx, sy = -dy * w, dx * w
            l = o.vert((px + sx, py + sy, pz - w * 0.25), (0.05, t), (0, 0, 1))
            c = o.vert((px, py, pz + 4), (0.1, t), (0, 0, 1))
            r = o.vert((px - sx, py - sy, pz - w * 0.25), (0.15, t), (0, 0, 1))
            if prev:
                pl, pc, pr = prev
                o.quad(pl, pc, c, l)
                o.quad(pc, pr, r, c)
            prev = (l, c, r)
    o.write(path)
    return len(o.f)


def shrub(path, seed=5):
    rnd = random.Random(seed)
    o = Obj()
    for k in range(7):
        a = rnd.uniform(0, 2 * math.pi); d = rnd.uniform(0, 45)
        sphere(o, (math.cos(a) * d, math.sin(a) * d, rnd.uniform(50, 100)), rnd.uniform(45, 70), segs=10, rings=7,
               squash=0.8, u=(0.05, 0.05))
    o.write(path)
    return len(o.f)


def rain_streak(path, length=170.0, width=2.6):
    """Two crossed thin quads, pivot at the top; M_RainStreak animates the fall with WPO."""
    o = Obj()
    for ang in (0.0, math.pi / 2):
        dx, dy = math.cos(ang) * width / 2, math.sin(ang) * width / 2
        ids = [o.vert((-dx, -dy, 0), (0, 0)), o.vert((dx, dy, 0), (1, 0)),
               o.vert((dx, dy, -length), (1, 1)), o.vert((-dx, -dy, -length), (0, 1))]
        o.quad(*ids)
    o.write(path)
    return 4


def water_tile(path, size=500.0):
    o = Obj()
    h = size / 2
    ids = [o.vert((-h, -h, 0), (0, 0)), o.vert((h, -h, 0), (1, 0)), o.vert((h, h, 0), (1, 1)), o.vert((-h, h, 0), (0, 1))]
    o.quad(*ids)
    o.write(path)
    return 2


def main(out):
    os.makedirs(out, exist_ok=True)
    for name, fn in [("SM_RiceCell", rice_cell),
                     ("SM_RiceCellLite", lambda p: rice_cell(p, n=12, seed=9, leaves_range=(16, 20))), ("SM_WaterTile", water_tile), ("SM_DroneBody", drone_body),
                     ("SM_Rotor", rotor), ("SM_RainStreak", rain_streak), ("SM_Palm", palm), ("SM_Shrub", shrub)]:
        tris = fn(os.path.join(out, name + ".obj"))
        print(f"{name}: {tris} triangles")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "generated_meshes"))
