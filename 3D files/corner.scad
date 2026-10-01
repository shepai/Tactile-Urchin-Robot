// --- Adjustable Parameters ---
$fn = 100;                  // Smoothness of the circular holes

// Hole Dimensions
hole_wide_d = 8;            // Diameter of the wider upper hole (mm)
hole_small_d = 4;           // Diameter of the smaller through hole (mm)
counterbore_depth = 2;      // Depth of the 8mm upper recess (mm)
face_hole_d = 2.5;          // Diameter of the holes on the 5 faces (mm)

// Pyramid Dimensions
slant_angle = 90-54;        // Angle of the sloped sides (degrees)
total_height = 4;           // Total vertical height of the model (mm)
top_radius = 6;             // Circumradius of the top pentagon (mm)

// --- Math Derivation ---
base_expansion = total_height / tan(slant_angle);
bottom_radius = top_radius + base_expansion;

// --- Main Model Assembly ---
difference() {
    // 1. Exterior Truncated Pentagonal Pyramid
    cylinder(h = total_height, r1 = bottom_radius, r2 = top_radius, $fn = 5);
    
    // 2. Centered Counterbore Hole (Upper Recess)
    translate([0, 0, total_height - counterbore_depth + 0.01])
        cylinder(h = counterbore_depth, d = hole_wide_d);
    
    // 3. Centered Through Hole (Smaller Hole)
    translate([0, 0, -0.01])
        cylinder(h = total_height + 0.02, d = hole_small_d);
        
    // 4. Five Radial Face Holes (Drilling Straight Down)
    // Calculate the distance from center to the midpoint of the flat face (Inradius)
    // at the halfway height of the model
    mid_radius = (top_radius + (base_expansion / 2)) * cos(180 / 5); 
    
    for (i = [0 : 4]) {
        rotate([0, 0, 36 + (i * 72)]) 
        // Position on the face, but elevate the drill above the model
        translate([mid_radius, 0, total_height / 2]) 
        // No Y-rotation means the cylinder remains perfectly vertical
        cylinder(h = total_height + 2, d = face_hole_d, center = true); 
    }
}
