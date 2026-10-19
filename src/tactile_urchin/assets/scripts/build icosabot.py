#!/usr/bin/env python3
"""
Icosahedron "push-walker" -> MuJoCo MJCF generator   (v2: pads are the OUTER SKIN)

  20 tactile pads  = the outer shell, each with a grid of touch-sensor taxels
  12 vertex hubs   = sit INSIDE the shell, hub sides rest against the pad undersides
  12 linear actuators = slide through each hub; the foot pokes out of a hole in the
                        shell where the 5 pad corners meet at every vertex

Run:      python3 build_icosabot.py          -> writes icosabot.xml
View:     python -m mujoco.viewer --mjcf=icosabot.xml

MODULARITY: hub_blueprint / rod_blueprint / pad_blueprint are the parts to redesign.
The assembly code at the bottom only decides WHERE each blueprint is instanced.
Units in this file: mm and degrees. Output XML is in metres (MuJoCo SI).
"""
import math, os

# ============================ PARAMETERS =====================================
EDGE_LEN   = 80.0     # icosahedron edge length (outer skin, vertex-to-vertex)  [mm]

# --- Hub: same names as your OpenSCAD file ---
HUB_H           = 4.0
HUB_TOP_R       = 6.0
HUB_SLANT       = None    # None = exact icosahedron value (31.72 deg). Your SCAD used 36.
HUB_HOLE_D      = 4.0
HUB_CBORE_D     = 8.0
HUB_CBORE_DEPTH = 2.0
HUB_STL         = "corner.stl"   # used if present next to icosabot.xml (mm units)
HUB_CLEAR       = 0.1         # air gap between pad underside and hub sloped faces

# --- Linear actuator (rod through hub) ---
ROD_D     = 3.6
ROD_LEN   = 22.0
FOOT_R    = 3.5
STROKE    = 30.0     # NOTE: a 12 mm solenoid will really give ~2-5 mm. Change this.
ROD_MASS  = 0.10
KP, KV    = 5000.0, 0.5

# --- Tactile pad (outer skin) ---
PAD_T          = 2.0   # skin thickness
PAD_EDGE_GAP   = 1.0   # gap between neighbouring pads along each icosahedron edge
OPEN_MARGIN    = 0.0   # extra vertex-opening size beyond the hub's top edge
TAXEL_N        = 5     # taxels per triangle side (lattice N(N+1)/2, corner ones dropped)
TAXEL_FILL     = 0.6   # taxel size as fraction of its cell
TAXEL_EDGE_MARGIN = 0.5  # keep taxels this far from the vertex openings

# --- Core (electronics / battery) ---
CORE_R, CORE_MASS = 25.0, 0.10

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
V = [mul(p, EDGE_LEN / 2) for p in raw]
Rc = norm(V[0])                                   # outer circumradius
adj = lambda i, j: abs(norm(sub(V[i], V[j])) - EDGE_LEN) < 1e-6 * EDGE_LEN
FACES = [(i, j, k) for i in range(12) for j in range(i+1, 12) for k in range(j+1, 12)
         if adj(i, j) and adj(j, k) and adj(i, k)]
assert len(V) == 12 and len(FACES) == 20
INR = norm(mul(add(add(V[FACES[0][0]], V[FACES[0][1]]), V[FACES[0][2]]), 1 / 3))  # face distance
THETA = math.acos(INR / Rc)                       # vertex axis <-> face normal (37.38 deg)

# ------------------------- hub geometry (inside the skin) --------------------
if HUB_SLANT is None:
    Rp = 1 / (2 * math.sin(math.radians(36)))
    HUB_SLANT = math.degrees(math.atan2(math.sqrt(1 - Rp**2), Rp))   # 31.72
HUB_BOT_R = HUB_TOP_R + HUB_H / math.tan(math.radians(HUB_SLANT))
HUB_DEPTH = HUB_BOT_R * math.tan(math.radians(HUB_SLANT))    # inner apex -> hub bottom plane
# The hub's sloped faces must lie on planes PAD_T(+gap) BELOW the outer face planes,
# i.e. on a slightly smaller icosahedron whose apex is at radius HUB_APEX_R.
HUB_APEX_R = Rc * (INR - PAD_T - HUB_CLEAR) / INR
DELTA = Rc - HUB_APEX_R                           # axial drop of hub apex below outer vertex

# ------------------------- pad outline (full face minus vertex openings) -----
FACE_R = EDGE_LEN / math.sqrt(3)                  # face circumradius
PAD_R  = FACE_R - PAD_EDGE_GAP                    # inset so neighbours leave a gap
# in-face distance from the OUTER vertex to the hub's top edge, measured along the bisector:
S_TOP = DELTA * math.sin(THETA) + math.hypot(HUB_DEPTH - HUB_H,
                                             HUB_TOP_R * math.cos(math.radians(36)))
L_CUT = S_TOP + OPEN_MARGIN                       # each pad corner is cut off at this distance
CUT_H = FACE_R - L_CUT                            # cut line: p . u_k <= CUT_H
U = [(math.cos(math.radians(a)), math.sin(math.radians(a))) for a in (0, 120, 240)]

def _clip(poly, u, h):
    out = []
    for i in range(len(poly)):
        p, q = poly[i], poly[(i + 1) % len(poly)]
        dp = p[0]*u[0] + p[1]*u[1] - h
        dq = q[0]*u[0] + q[1]*u[1] - h
        if dp <= 0: out.append(p)
        if dp * dq < 0:
            t = dp / (dp - dq)
            out.append((p[0] + t*(q[0]-p[0]), p[1] + t*(q[1]-p[1])))
    return out

TRI = [(PAD_R * u[0], PAD_R * u[1]) for u in U]
PAD_POLY = TRI
for u in U:
    PAD_POLY = _clip(PAD_POLY, u, CUT_H)          # -> hexagon (triangle with 3 corners cut)


# =============================== BLUEPRINTS ==================================
def hub_blueprint(vid):
    """Local frame: origin = centre of hub BOTTOM plane, +Z radially out,
    pentagon corners at 0,72,144.. deg. Hub is inside the skin, so no collision."""
    return f'<geom class="hub" mesh="hub_mesh" name="hub{vid:02d}_geom"/>'


def rod_blueprint(vid):
    """Local frame: same as hub. Slide joint along +Z through the hub's hole.
    The foot retracts into the counterbore and extends out through the vertex opening."""
    n = f"rod{vid:02d}"
    z_foot = HUB_H - HUB_CBORE_DEPTH
    return f'''<body name="{n}">
          <joint name="slide{vid:02d}" type="slide" axis="0 0 1" range="0 {STROKE*mm:g}" limited="true" damping="0.5"/>
          <geom class="rod" type="cylinder" fromto="0 0 {-ROD_LEN*mm:g} 0 0 {z_foot*mm:g}" size="{ROD_D/2*mm:g}" mass="{ROD_MASS:g}"/>
          <geom class="rod" type="sphere" pos="0 0 {z_foot*mm:g}" size="{FOOT_R*mm:g}" mass="0.002"/>
        </body>'''


def _taxel_ok(cx, cy, half):
    for sx in (-1, 1):
        for sy in (-1, 1):
            px, py = cx + sx * half, cy + sy * half
            if any(px*u[0] + py*u[1] > CUT_H - TAXEL_EDGE_MARGIN for u in U):
                return False
    return True


def pad_blueprint(fid):
    """Local frame: origin = face centre, +Z = outward normal, +X toward corner 0.
    Outer (touch) surface at z = 0, skin extends to z = -PAD_T.
    Outline = PAD_POLY (triangle with the 3 vertex openings cut out).
    Returns (body_inner_xml, [touch_sensor_xml, ...]).

    Contacts come from the small TAXEL boxes (a big flat polygon on a plane would
    only touch at its corners). The outline mesh is visual only."""
    inner, sensors = [], []
    n = TAXEL_N
    half = TAXEL_FILL * PAD_R / (2 * n)
    k = 0
    for i in range(n):
        for j in range(n - i):
            l = n - 1 - i - j
            w = ((i + 1/3) / n, (j + 1/3) / n, (l + 1/3) / n)
            x = sum(wi * c[0] for wi, c in zip(w, TRI))
            y = sum(wi * c[1] for wi, c in zip(w, TRI))
            if not _taxel_ok(x, y, half):
                continue                            # would overlap a vertex opening
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
    else:
        pts = []
        for z, r in ((0.0, HUB_BOT_R), (HUB_H, HUB_TOP_R)):
            for k in range(5):
                a = math.radians(72 * k)
                pts += [r * math.cos(a) * mm, r * math.sin(a) * mm, z * mm]
        hub = f'<mesh name="hub_mesh" vertex="{f(*pts)}"/>'
    zt, zb = -0.3 * mm, -PAD_T * mm                    # sink visual 0.3 mm to avoid z-fighting
    pv = [c for (x, y) in PAD_POLY for c in (x*mm, y*mm, zt)] + \
         [c for (x, y) in PAD_POLY for c in (x*mm, y*mm, zb)]
    pad = f'<mesh name="pad_mesh" vertex="{f(*pv)}"/>'
    return hub, pad


def build():
    hub_mesh, pad_mesh = mesh_assets()
    body_parts, act, sens = [], [], []

    # ---- 12 hubs (inside the skin), each carrying its actuator rod ----
    for i, v in enumerate(V):
        z = unit(v)
        nb = next(j for j in range(12) if j != i and adj(i, j))
        d = sub(V[nb], v)
        x = unit(sub(d, mul(z, dot(d, z))))
        y = cross(z, x)
        pos = mul(z, HUB_APEX_R - HUB_DEPTH)           # <- moved inward vs. v1
        body_parts.append(f'''      <body name="hub{i:02d}" pos="{f(*mul(pos, mm))}" xyaxes="{f(*x, *y)}">
        {hub_blueprint(i)}
        {rod_blueprint(i)}
      </body>''')
        act.append(f'<position name="act{i:02d}" joint="slide{i:02d}" kp="{KP:g}" kv="{KV:g}" '
                   f'ctrlrange="0 {STROKE*mm:g}" ctrllimited="true"/>')

    # ---- 20 pads (outer skin) ----
    ntax = 0
    for fi, (a, b, c) in enumerate(FACES):
        va, vb, vc = V[a], V[b], V[c]
        cen = mul(add(add(va, vb), vc), 1 / 3)
        n = unit(cen)
        x = unit(sub(va, cen))
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
    <!-- hubs are inside the skin: visual/mass only -->
    <default class="hub"><geom type="mesh" contype="0" conaffinity="0" rgba="0.85 0.85 0.9 1" density="1000"/></default>
    <default class="rod"><geom contype="2" conaffinity="2" rgba="0.95 0.45 0.15 1"/></default>
    <default class="padvis"><geom type="mesh" contype="0" conaffinity="0" group="2" rgba="0.3 0.6 1 0.45" density="0"/></default>
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
    <!-- contype/conaffinity 3 = collides with taxels(1) and rod feet(2) -->
    <geom name="floor" type="plane" size="2 2 0.1" contype="3" conaffinity="3" rgba="0.25 0.25 0.3 1"/>
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
    here = os.path.dirname(os.path.abspath(__file__)).replace("/scripts","/")
    with open(os.path.join(here, OUT), "w") as fh:
        fh.write(xml)
    print(f"wrote {OUT}: 12 hubs/actuators inside, 20 outer pads, {len(sens)} touch sensors "
          f"({len(sens)//20} per pad)")
    print(f"outer circumradius {Rc:.1f} mm | hub apex radius {HUB_APEX_R:.1f} mm "
          f"(hub moved {DELTA:.2f} mm inward) | slant {HUB_SLANT:.2f} deg")
    print(f"vertex opening apothem {L_CUT*math.cos(THETA):.2f} mm (foot radius {FOOT_R} mm) | "
          f"pad outline {len(PAD_POLY)}-sided, circumradius {PAD_R:.1f} mm")


if __name__ == "__main__":
    build()