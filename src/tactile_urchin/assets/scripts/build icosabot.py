#!/usr/bin/env python3
"""
Icosahedron "push-walker" -> MuJoCo MJCF generator.

  12 vertex hubs  (your truncated pentagonal pyramid, STL or built-in)
  12 linear actuators (slide joint through each hub's centre hole)
  20 tactile pads (triangular faces, each with a grid of touch-sensor taxels)

Run:      python3 build_icosabot.py          -> writes icosabot.xml
View:     python -m mujoco.viewer --mjcf=icosabot.xml

MODULARITY: the three blueprint functions (hub_blueprint, rod_blueprint,
pad_blueprint) are all you need to edit to redesign a part. The assembly code
at the bottom only decides WHERE each blueprint is instanced.
Units in this file: mm and degrees. Output XML is in metres (MuJoCo SI).
"""
import math, os

# ============================ PARAMETERS =====================================
EDGE_LEN   = 80.0     # icosahedron edge length (vertex-to-vertex)  [mm]

# --- Hub: same names as your OpenSCAD file ---
HUB_H           = 4.0     # total_height
HUB_TOP_R       = 6.0     # top_radius (circumradius of top pentagon)
HUB_SLANT       = None    # None = exact icosahedron value (31.72 deg). Your SCAD used 36.
HUB_HOLE_D      = 4.0     # hole_small_d
HUB_CBORE_D     = 8.0     # hole_wide_d
HUB_CBORE_DEPTH = 2.0     # counterbore_depth
HUB_STL         = "hub.stl"   # if this file sits next to icosabot.xml it is used (mm units)

# --- Linear actuator (rod through hub) ---
ROD_D     = 3.6     # shaft diameter (fits your 4 mm hole)
ROD_LEN   = 22.0    # shaft length below hub bottom plane
FOOT_R    = 3.5     # spherical foot radius (fits your 8 mm counterbore)
STROKE    = 25.0    # travel
ROD_MASS  = 0.010   # kg
KP, KV    = 2000.0, 3.0   # position servo gains (N/m, N*s/m)

# --- Tactile pad ---
PAD_T       = 2.0   # pad thickness
PAD_GAP     = 1.0   # clearance between pad and hub shoulder
TAXEL_N     = 4     # taxels per triangle side -> N(N+1)/2 taxels per pad (4 -> 10)
TAXEL_FILL  = 0.6   # taxel size as fraction of its cell (<1 leaves gaps)

# --- Core (electronics / battery) ---
CORE_R, CORE_MASS = 25.0, 0.20

OUT = "icosabot.xml"
# ============================================================================

mm = 1e-3
def f(*v): return " ".join(f"{x:.6g}" for x in v)
def sub(a, b): return tuple(x - y for x, y in zip(a, b))
def add(a, b): return tuple(x + y for x, y in zip(a, b))
def mul(a, s): return tuple(x * s for x in a)
def dot(a, b): return sum(x * y for x, y in zip(a, b))
def cross(a, b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def norm(a): return math.sqrt(dot(a, a))
def unit(a): n = norm(a); return tuple(x / n for x in a)

# ------------------------- icosahedron topology ------------------------------
phi = (1 + 5 ** 0.5) / 2
raw = []
for s1 in (-1, 1):
    for s2 in (-1, 1):
        raw += [(0, s1, s2 * phi), (s1, s2 * phi, 0), (s2 * phi, 0, s1)]
V = [mul(p, EDGE_LEN / 2) for p in raw]           # edge length = EDGE_LEN
Rc = norm(V[0])                                   # circumradius
adj = lambda i, j: abs(norm(sub(V[i], V[j])) - EDGE_LEN) < 1e-6 * EDGE_LEN
FACES = [(i, j, k) for i in range(12) for j in range(i+1, 12) for k in range(j+1, 12)
         if adj(i, j) and adj(j, k) and adj(i, k)]
assert len(V) == 12 and len(FACES) == 20

# ------------------------- derived hub geometry ------------------------------
if HUB_SLANT is None:
    Rp = 1 / (2 * math.sin(math.radians(36)))     # pentagon circumradius, edge = 1
    HUB_SLANT = math.degrees(math.atan2(math.sqrt(1 - Rp**2), Rp))   # = 31.72
HUB_BOT_R = HUB_TOP_R + HUB_H / math.tan(math.radians(HUB_SLANT))
HUB_DEPTH = HUB_BOT_R * math.tan(math.radians(HUB_SLANT))   # vertex tip -> hub bottom plane
# in-face distance from the vertex to the hub's inner (bottom) pentagon side:
CLIP_V = math.sqrt(HUB_DEPTH**2 + (HUB_BOT_R * math.cos(math.radians(36)))**2)
FACE_R = EDGE_LEN / math.sqrt(3)                  # face circumradius
PAD_R  = FACE_R - CLIP_V - PAD_GAP                # pad corner radius


# =============================== BLUEPRINTS ==================================
# Edit these to redesign a part. Local frames are documented in each docstring.

def hub_blueprint(vid):
    """Local frame: origin = centre of hub BOTTOM plane, +Z points radially out,
    pentagon corners at 0,72,144.. deg (same as OpenSCAD cylinder $fn=5).
    Returns geom xml (collision/visual) for the hub body."""
    return f'<geom class="hub" mesh="hub_mesh" name="hub{vid:02d}_geom"/>'


def rod_blueprint(vid):
    """Local frame: same as hub. Returns the child <body> holding the actuator slide.
    The rod slides along +Z through the hub's hole."""
    n = f"rod{vid:02d}"
    z_foot = HUB_H - HUB_CBORE_DEPTH                      # foot rests on counterbore floor
    return f'''<body name="{n}">
          <joint name="slide{vid:02d}" type="slide" axis="0 0 1" range="0 {STROKE*mm:g}" limited="true" damping="0.5"/>
          <geom class="rod" type="cylinder" fromto="0 0 {-ROD_LEN*mm:g} 0 0 {z_foot*mm:g}" size="{ROD_D/2*mm:g}" mass="{ROD_MASS:g}"/>
          <geom class="rod" type="sphere" pos="0 0 {z_foot*mm:g}" size="{FOOT_R*mm:g}" mass="0.002"/>
        </body>'''


def pad_blueprint(fid):
    """Local frame: origin = face centre, +Z = outward normal, +X points at corner 0
    (corners at 0/120/240 deg, circumradius PAD_R). Outer surface is at z = 0.
    Returns (body_inner_xml, [touch_sensor_xml, ...]).

    Physics trick: contacts are generated by the small TAXEL boxes, NOT by one big
    triangle - a flat triangle on a plane only produces contacts at its corners,
    so centre taxels would never fire. The triangle mesh is visual only."""
    corners = [(PAD_R * math.cos(math.radians(a)), PAD_R * math.sin(math.radians(a)))
               for a in (0, 120, 240)]
    inner, sensors = [], []

    # taxel lattice: centroids of the up-pointing sub-triangles
    n = TAXEL_N
    half = TAXEL_FILL * PAD_R / (2 * n)               # PAD_R/(2n) = sub-triangle inradius
    k = 0
    for i in range(n):
        for j in range(n - i):
            l = n - 1 - i - j
            w = ((i + 1/3) / n, (j + 1/3) / n, (l + 1/3) / n)
            x = sum(wi * c[0] for wi, c in zip(w, corners))
            y = sum(wi * c[1] for wi, c in zip(w, corners))
            name = f"pad{fid:02d}_t{k:02d}"
            sz = f"{half*mm:g} {half*mm:g} {PAD_T/2*mm:g}"
            pos = f"{x*mm:g} {y*mm:g} {-PAD_T/2*mm:g}"
            inner.append(f'<geom class="taxel" name="{name}_g" pos="{pos}" size="{sz}"/>')
            sz_s = f"{(half+0.5)*mm:g} {(half+0.5)*mm:g} {(PAD_T/2+0.5)*mm:g}"
            inner.append(f'<site class="taxel" name="{name}" pos="{pos}" size="{sz_s}"/>')
            sensors.append(f'<touch name="{name}_touch" site="{name}"/>')
            k += 1
    inner.append('<geom class="padvis" mesh="pad_mesh"/>')
    return "\n        ".join(inner), sensors
# ============================================================================


def mesh_assets():
    hub_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), HUB_STL)
    if os.path.exists(hub_file):
        hub = f'<mesh name="hub_mesh" file="{HUB_STL}" scale="{mm} {mm} {mm}"/>'
    else:                                              # built-in convex stand-in (no holes)
        pts = []
        for z, r in ((0.0, HUB_BOT_R), (HUB_H, HUB_TOP_R)):
            for k in range(5):
                a = math.radians(72 * k)
                pts += [r * math.cos(a) * mm, r * math.sin(a) * mm, z * mm]
        hub = f'<mesh name="hub_mesh" vertex="{f(*pts)}"/>'
    c = [(PAD_R * math.cos(math.radians(a)), PAD_R * math.sin(math.radians(a))) for a in (0, 120, 240)]
    zt, zb = -0.3 * mm, -PAD_T * mm                    # sink visual 0.3 mm to avoid z-fighting
    pv = [coord for (x, y) in c for coord in (x*mm, y*mm, zt)] + \
         [coord for (x, y) in c for coord in (x*mm, y*mm, zb)]
    pad = f'<mesh name="pad_mesh" vertex="{f(*pv)}"/>'
    return hub, pad


def build():
    hub_mesh, pad_mesh = mesh_assets()
    body_parts, act, sens = [], [], []

    # ---- 12 hubs, each carrying its own actuator rod ----
    for i, v in enumerate(V):
        z = unit(v)
        nb = next(j for j in range(12) if j != i and adj(i, j))
        d = sub(V[nb], v)
        x = unit(sub(d, mul(z, dot(d, z))))            # pentagon corner 0 -> a neighbour vertex
        y = cross(z, x)
        pos = mul(z, Rc - HUB_DEPTH)
        body_parts.append(f'''      <body name="hub{i:02d}" pos="{f(*mul(pos, mm))}" xyaxes="{f(*x, *y)}">
        {hub_blueprint(i)}
        {rod_blueprint(i)}
      </body>''')
        act.append(f'<position name="act{i:02d}" joint="slide{i:02d}" kp="{KP:g}" kv="{KV:g}" '
                   f'ctrlrange="0 {STROKE*mm:g}" ctrllimited="true"/>')

    # ---- 20 pads ----
    for fi, (a, b, c) in enumerate(FACES):
        va, vb, vc = V[a], V[b], V[c]
        cen = mul(add(add(va, vb), vc), 1 / 3)
        n = unit(cen)
        x = unit(sub(va, cen))
        if dot(cross(x, sub(vb, cen)), n) < 0:         # keep corners CCW
            vb, vc = vc, vb
        y = cross(n, x)
        inner, sensors = pad_blueprint(fi)
        body_parts.append(f'''      <body name="pad{fi:02d}" pos="{f(*mul(cen, mm))}" xyaxes="{f(*x, *y)}">
        {inner}
      </body>''')
        sens += sensors

    xml = f'''<mujoco model="icosabot">
  <compiler angle="degree" autolimits="true" inertiafromgeom="auto"/>
  <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"/>

  <default>
    <geom friction="1.0 0.005 0.0001"/>
    <default class="hub"><geom type="mesh" contype="4" conaffinity="4" rgba="0.85 0.85 0.9 1" density="1000"/></default>
    <default class="rod"><geom contype="2" conaffinity="2" rgba="0.95 0.45 0.15 1"/></default>
    <default class="padvis"><geom type="mesh" contype="0" conaffinity="0" group="2" rgba="0.3 0.6 1 0.8" density="0"/></default>
    <default class="taxel">
      <geom type="box" contype="1" conaffinity="1" rgba="0.2 0.85 0.5 1" density="500"/>
      <site type="box" rgba="1 0 0 0.15" group="3"/>
    </default>
  </default>

  <asset>
    {hub_mesh}
    {pad_mesh}
  </asset>

  <worldbody>
    <light pos="0 0 1" dir="0 0 -1"/>
    <!-- contype/conaffinity 7 = collides with pads(1), rods(2), hubs(4) -->
    <geom name="floor" type="plane" size="2 2 0.1" contype="7" conaffinity="7" rgba="0.25 0.25 0.3 1"/>
    <body name="core" pos="0 0 0.12">
      <freejoint name="root"/>
      <geom name="core_geom" type="sphere" size="{CORE_R*mm:g}" mass="{CORE_MASS:g}" contype="0" conaffinity="0" rgba="0.2 0.2 0.2 1"/>
      <site name="imu" size="0.003"/>
{chr(10).join(body_parts)}
    </body>
  </worldbody>

  <actuator>
    {chr(10).join("    " + a if i else a for i, a in enumerate(act))}
  </actuator>

  <sensor>
    <accelerometer name="imu_acc" site="imu"/>
    <gyro name="imu_gyro" site="imu"/>
    {chr(10).join("    " + s if i else s for i, s in enumerate(sens))}
  </sensor>
</mujoco>
'''
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, OUT), "w") as fh:
        fh.write(xml)
    print(f"wrote {OUT}: 12 hubs/actuators, 20 pads, {len(sens)} touch sensors")
    print(f"circumradius {Rc:.1f} mm | hub slant {HUB_SLANT:.2f} deg | hub bottom R {HUB_BOT_R:.2f} mm "
          f"| pad corner radius {PAD_R:.1f} mm (edge {PAD_R*math.sqrt(3):.1f} mm)")


if __name__ == "__main__":
    build()