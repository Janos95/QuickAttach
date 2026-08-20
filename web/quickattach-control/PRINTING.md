# Printable tool rack

The generated rack is a modular, rounded, under-slung truss designed to keep the
robot's approach corridor open while giving every load a continuous path to the
bench. All dimensions below come from
`public/model/printable_fixture_manifest.json`.

## Parts

| Quantity | Part | Print orientation |
| ---: | --- | --- |
| 1 each | `station_gripper`, `station_spoon`, `station_whisk` | Upright on the rounded feet; support the short crown cantilever |
| 1 each | `saddle_gripper`, `saddle_spoon`, `saddle_whisk` | Flat, with dock bosses upward |
| 1 each | two front and two rear link bars | Broad face on the bed |

The ten STL files are under `public/model/`. The largest envelope is
190.39 × 66.76 × 5.5 mm. Total modeled printed volume is 244.01 cm³. The
generator checks that every part is one valid solid and that assembled printed
parts do not intersect.

## Hardware

- 6 × M5 bench fasteners, plus bench-appropriate washers and anchors
- M4 screws for the 8 base-link joints and 6 saddle joints
- M4 screws and heat-set inserts for the dock interfaces
- 0.75 mm washers between station feet and link bars
- 0.5 mm washers between each station crown and saddle
- 2 × short 3.5 mm metal stays for the spoon dock and 2 × for the whisk dock

Select screw lengths after measuring the printed parts, washer stack, insert
depth, and retained source adapter. No screw may protrude into a tool or flange
envelope.

## Assembly

1. Print and deburr all holes. Install heat-set inserts with a temperature and
   pull-out test appropriate to the chosen material.
2. Join each T-saddle to its A-frame crown using the 0.5 mm washer gap. The
   saddle must remain below the parked tool envelope.
3. Fasten the gripper dock through the two M4 centers in its retained source
   support head.
4. Connect each spoon/whisk saddle stay pad to the dock's retained rear-anchor
   block using two short metal stays. Do not substitute printed cantilevers for
   these stays.
5. Join adjacent station feet with the four link bars and 0.75 mm washers.
6. Square the assembly, then fasten all six station feet to a rigid bench.
7. Move the unpowered robot through each docking path by hand before enabling
   actuators. Verify screw clearance, flange access, and cable routing.

## Qualification boundary

The CAD generator and collision sweep establish geometric realizability, not
strength, fatigue life, print quality, or safe robot operation. Material,
infill, layer direction, inserts, bench anchors, and dynamic loads remain to be
qualified on physical hardware. Prototype one complete station first and use a
low-speed, current-limited commissioning pass.
