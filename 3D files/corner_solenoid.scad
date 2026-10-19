// =============================================================
//  Icosahedron hub  +  PENTAGONAL solenoid sleeve on the INNER (flat) side
//  Solenoid: ~10.5 x 8 x 16 mm, piston exits from a 10.5 x 8 face
// =============================================================
$fn = 100;

// --- Hub parameters ---
hole_wide_d       = 8;     // wider upper hole (mm) - the 5 mm end cap sits in here
hole_small_d      = 8;     // through hole for the spring/shaft/cap
counterbore_depth = 2;     // depth of the 8 mm recess (mm)
face_hole_d       = 2.5;   // holes on the 5 faces (mm)
slant_angle       = 90-54; // = 36. Exact icosahedron value is atan(0.5257/0.8507) = 31.72
total_height      = 4;     // hub height (mm)
top_radius        = 6;     // circumradius of top pentagon (mm)

// --- Solenoid parameters ---
sol_x   = 11;     // pocket size along X (body is 10.5 + a little slack)
sol_y   = 9;      // pocket size along Y (body is 8 + a little slack)
sol_len = 16;     // body length along the piston axis (Z)

fit_tol    = 0.2;  // clearance per side around the solenoid (FDM printing: 0.15-0.25)
sleeve_len = 15;   // length of the pentagonal prism below the hub (must clear the lowest screw head)
pent_shrink = 0;   // shrink the prism's circumradius (mm) if you want it smaller than the hub base
min_wall    = 1.2; // only used for the warning below
sol_rot    = 0;    // rotate the pocket (not the pentagon) about the hub axis (deg)

// If the piston is NOT in the centre of the 10.5 x 8 face, enter its offset here.
piston_off_x = 0;
piston_off_y = 0;

// --- M2 screw holes (both on ONE 16 x 8 face of the solenoid) ---
screw_enable = true;
screw_d      = 2.2;   // clearance hole for M2
screw_head_d = 4.2;   // counterbore for the M2 head (pan head ~3.8-4)
screw_wall_keep = 1.5;// wall left between pocket and screw head (the shank passes through this)
// each entry: [distance from PISTON face along the body, offset across the 8 mm face]
screw_pts    = [[6.9, -2.1],
                [11.9, 2.1]];
screw_axis   = 0;     // 0 = drilled along X, 1 = drilled along Y
screw_side   = 0;     // 0 = both side walls, +1 or -1 = only that wall
                      // (-1 with sol_rot=0 is a FLAT outer wall, easiest for the screw head)

// --- Math derivation ---
base_expansion = total_height / tan(slant_angle);
bottom_radius  = top_radius + base_expansion;
R_pent         = bottom_radius - pent_shrink;      // radius of the sleeve pentagon

// --- Derived sizes ---
pocket_x = sol_x + 2*fit_tol;
pocket_y = sol_y + 2*fit_tol;

// --- Wall check: pocket corners vs the pentagon sides ---
function rot2(p, a) = [p[0]*cos(a) - p[1]*sin(a), p[0]*sin(a) + p[1]*cos(a)];
function pent_margin(p) = min([for (i = [0:4])
        R_pent*cos(36) - (p[0]*cos(36 + 72*i) + p[1]*sin(36 + 72*i))]);
pocket_corners = [for (sx = [-1, 1], sy = [-1, 1])
        rot2([-piston_off_x + sx*pocket_x/2, -piston_off_y + sy*pocket_y/2], sol_rot)];
min_margin = min([for (c = pocket_corners) pent_margin(c)]);
echo(str("Thinnest pentagon wall around the pocket: ", min_margin, " mm"));
if (min_margin < min_wall)
    echo("WARNING: wall is thin - shrink the pocket, reduce pent_shrink, or change sol_rot");

// =============================================================
//  Modules
// =============================================================
module hub_body() {
    difference() {
        cylinder(h = total_height, r1 = bottom_radius, r2 = top_radius, $fn = 5);

        translate([0, 0, total_height - counterbore_depth + 0.01])
            cylinder(h = counterbore_depth, d = hole_wide_d);

        translate([0, 0, -0.01])
            cylinder(h = total_height + 0.02, d = hole_small_d);

        mid_radius = (top_radius + (base_expansion / 2)) * cos(180 / 5);
        for (i = [0 : 4])
            rotate([0, 0, 36 + i*72])
                translate([mid_radius, 0, total_height / 2])
                    cylinder(h = total_height + 2, d = face_hole_d, center = true);
    }
}

// Pentagonal prism that continues the hub's bottom face straight down.
// Same orientation as the hub (corner on +X).
module sleeve_outer() {
    translate([0, 0, -sleeve_len + 0.005])
        cylinder(h = sleeve_len + 0.01, r = R_pent, $fn = 5);
}

module sleeve_cut() {
    rotate([0, 0, sol_rot])
        translate([-piston_off_x, -piston_off_y, 0]) {
            // rectangular pocket: top face flush with the hub's bottom plane, open at the rear
            translate([0, 0, -sleeve_len/2 - 0.05])
                cube([pocket_x, pocket_y, sleeve_len + 0.1], center = true);

            // M2 screw holes: shank through the wall + counterbore for the head
            if (screw_enable) {
                half = (screw_axis == 0 ? pocket_x : pocket_y) / 2;
                far  = R_pent + 2;
                for (p = screw_pts)
                    for (s = (screw_side == 0 ? [-1, 1] : [screw_side]))
                        translate(screw_axis == 0 ? [0, p[1], -p[0]] : [p[1], 0, -p[0]])
                            rotate(screw_axis == 0 ? [0, s*90, 0] : [-s*90, 0, 0]) {
                                cylinder(h = far, d = screw_d, $fn = 32);               // shank
                                translate([0, 0, half + screw_wall_keep])
                                    cylinder(h = far, d = screw_head_d, $fn = 32);      // head recess
                            }
            }
        }
}

// =============================================================
//  Assembly
// =============================================================
difference() {
    union() {
        hub_body();
        sleeve_outer();
    }
    sleeve_cut();

    // Re-open the shaft hole through the full stack
    translate([0, 0, -sleeve_len - 1])
        cylinder(h = sleeve_len + total_height + 2, d = hole_small_d);
}