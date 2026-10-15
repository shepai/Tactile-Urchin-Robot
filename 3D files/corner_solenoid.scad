// =============================================================
//  Icosahedron hub  +  solenoid sleeve on the INNER (flat) side
//  Solenoid: 10.5 x 8 x 16 mm, piston exits from a 10.5 x 8 face
// =============================================================
$fn = 100;

// --- Hub parameters ---
hole_wide_d       = 8;     // wider upper hole (mm) - the 5 mm end cap sits in here
hole_small_d      = 4.6;   // through hole for the spring/shaft. Spring looks ~4.1 mm OD:
                           // MEASURE IT and use OD + ~0.4
counterbore_depth = 2;     // depth of the 8 mm recess (mm)
face_hole_d       = 2.5;   // holes on the 5 faces (mm)
slant_angle       = 90-54; // = 36. Exact icosahedron value is atan(0.5257/0.8507) = 31.72
total_height      = 4;     // hub height (mm)
top_radius        = 6;     // circumradius of top pentagon (mm)

// --- Solenoid parameters ---
sol_x   = 10.5;   // body size along X
sol_y   = 8;      // body size along Y
sol_len = 16;     // body length along the piston axis (Z)

fit_tol    = 0.2;   // clearance per side around the solenoid (FDM printing: 0.15-0.25)
wall       = 1.2;   // sleeve wall thickness
sleeve_len = 14;    // how much of the 16 mm body is wrapped; rear stays open for wires
sol_rot    = 0;     // rotate the sleeve about the hub axis (deg)

// If the piston is NOT in the centre of the 10.5 x 8 face, enter its offset here.
piston_off_x = 0;
piston_off_y = 0;

// --- M2 screw holes (both on ONE 16 x 8 face of the solenoid) ---
screw_enable = true;
screw_d      = 2.2;   // clearance hole for M2
// each entry: [distance from PISTON face along the body, offset across the 8 mm face]
// From the drawing: holes 5.0 mm apart lengthwise, 4.2 mm apart across (+/-2.1).
screw_pts    = [[6.9, -2.1],
                [11.9, 2.1]];
screw_axis   = 0;     // 0 = drilled along X (through the 10.5 mm axis) <- matches the drawing
                      // 1 = drilled along Y
screw_side   = 0;     // 0 = drill BOTH side walls (works for either mirror orientation)
                      // +1 or -1 = drill only that wall

// --- Math derivation ---
base_expansion = total_height / tan(slant_angle);
bottom_radius  = top_radius + base_expansion;

// --- Derived sleeve sizes ---
pocket_x = sol_x + 2*fit_tol;   pocket_y = sol_y + 2*fit_tol;
outer_x  = pocket_x + 2*wall;   outer_y  = pocket_y + 2*wall;

// --- Fit check: sleeve must stay inside the hub's bottom pentagon ---
function rot2(p, a) = [p[0]*cos(a) - p[1]*sin(a), p[0]*sin(a) + p[1]*cos(a)];
function pent_margin(p) = min([for (i = [0:4])
        bottom_radius*cos(36) - (p[0]*cos(36 + 72*i) + p[1]*sin(36 + 72*i))]);
sleeve_corners = [for (sx = [-1, 1], sy = [-1, 1])
        rot2([-piston_off_x + sx*outer_x/2, -piston_off_y + sy*outer_y/2], sol_rot)];
min_margin = min([for (c = sleeve_corners) pent_margin(c)]);
echo(str("Sleeve clearance to hub's bottom pentagon: ", min_margin, " mm"));
if (min_margin < 0)
    echo("WARNING: sleeve sticks out past the hub's inner pentagon - reduce wall/fit_tol or change sol_rot");

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

module sleeve_outer() {
    rotate([0, 0, sol_rot])
        translate([-piston_off_x, -piston_off_y, -sleeve_len/2 + 0.005])
            cube([outer_x, outer_y, sleeve_len + 0.01], center = true);
}

module sleeve_cut() {
    rotate([0, 0, sol_rot])
        translate([-piston_off_x, -piston_off_y, 0]) {
            // pocket: top face flush with the hub's bottom plane, open at the rear
            translate([0, 0, -sleeve_len/2 - 0.05])
                cube([pocket_x, pocket_y, sleeve_len + 0.1], center = true);

            // M2 screw holes through the sleeve wall(s)
            if (screw_enable) {
                reach = (screw_axis == 0 ? outer_x : outer_y) / 2 + 1;
                for (p = screw_pts)
                    for (s = (screw_side == 0 ? [-1, 1] : [screw_side]))
                        translate(screw_axis == 0 ? [s*reach/2, p[1], -p[0]]
                                                  : [p[1], s*reach/2, -p[0]])
                            rotate(screw_axis == 0 ? [0, 90, 0] : [90, 0, 0])
                                cylinder(h = reach, d = screw_d, center = true, $fn = 32);
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

    // Re-open the piston hole through the full stack (removes the sleeve's top skin)
    translate([0, 0, -sleeve_len - 1])
        cylinder(h = sleeve_len + total_height + 2, d = hole_small_d);
}