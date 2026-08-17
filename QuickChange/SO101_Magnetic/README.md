# SO-101 powered magnetic quick changer v0.2

This is a retrofit, passively docked quick changer for an existing SO-101. Two
permanent magnets perform forgiving capture and face preload; they are **not**
the safety lock. A spring-return stainless keyhole slider positively captures
two metal pull studs. The passive rack cams that slider open, so the robot only
needs straight approach/withdrawal and one rack-exit translation—no wrist
twist, release servo, solenoid, or electromagnet.

The same interface carries power and the existing half-duplex TTL bus. The
first tool can therefore be the standard SO-101 gripper on a detachable adapter
plate while its STS3215 remains motor ID 6.

The calibrated arm model, mesh assets, and stock-gripper STEP used here are
derived from TheRobotStudio/SO-ARM100. Exact source revision and license details
are recorded in the repository-root `THIRD_PARTY_NOTICES.md`.

## What is where

- **Robot side:** 48 x 48 x 9.5 mm main plate, plus a left electrical wing;
  overall envelope 60 x 48 mm. It replaces the gripper at the four existing
  holes on the measured 9.9 mm-square wrist-horn pattern.
- **Magnets:** two purple 12 x 12 x 4 mm blocks, flush in the robot plate at
  `(x, y) = (0, -16)` and `(0, +16)` mm. Each is mechanically fastened, not
  glued.
- **Tool side:** 56 x 50 x 9.5 mm main plate, overall envelope 64 x 50 mm. It
  carries two matching 12 x 12 x 3 mm Q235 steel targets, not more magnets, so
  tool polarity is irrelevant.
- **Location:** two 3.5 mm tapered pins; the second socket is tangentially
  relieved to avoid an over-constrained fit.
- **Positive lock:** two 4 mm pull-stud shoulders enter 6.5 mm keyholes. On
  leaving the rack, the E-GUL4-10 return spring shifts a 1.6 mm stainless slider
  by 3 mm so 4.25 mm necks sit under the 6 mm stud heads. The printed roof, not
  the M2 guide screw, reacts separation load. Loss of power leaves it locked.
- **Cam corridor:** the passive cam retains its full `x = 24.05 mm` unlock
  datum. A narrow local recess in the fixed robot plate provides 0.50 mm exact
  clearance through the complete rack-exit sweep without changing slider
  travel, keyhole alignment, spring stroke, or the 2.95 mm locked engagement.
- **Electrical interface:** four Mill-Max pins on 5 mm pitch: GND, +12 V,
  TTL_DATA, and a spare TOOL_ID line. GND protrudes 0.2 mm farther so it mates
  first and breaks last.

In the CAD and MuJoCo model, purple identifies the two magnets, silver their
targets and pull studs, green the positive-lock slider, and gold the four
contacts. Use the full-arm MuJoCo example below as the executable acceptance
check.

## Attach and detach sequence

1. Approach the docked tool axially. Tapered locators enter first; magnets then
   close the final gap and compress the contacts.
2. Verify contact—for the first gripper, read servo ID 6 with torque still off.
3. Translate out of the rack. The rack releases the slider tab and the spring
   closes the positive lock.
4. Enable the tool and work normally.
5. To detach, command gripper torque off and stop bus packets, then return the
   still-locked assembly to the rack.
6. During the final rack travel, the fixed wedge pushes the exposed tab 3 mm,
   aligning both large keyholes with the stud heads.
7. The rack retains the tool plate while the arm withdraws axially. Contacts and
   magnets separate; no powered release actuator is involved.

The MuJoCo model uses conditional welds for capture and lock state. It validates
this state sequence and geometry; it does not predict magnetic fields, contact
arcing, printed-part strength, fatigue, or wear.

## CAD/simulation geometry contract

`generate_cad.py` publishes the exact core dock-stop bounds, stock-gripper STEP
mount, and robot-plate cam-relief bounds in both `design_parameters.json` and
`exports/core_cad_manifest.json`. The stock dock stop is not interchangeable
with the spoon/whisk stops: core bounds are `X[-45,37]`, `Y[26,32]`,
`Z[-3,12.5]` mm, while the two-bay matcha package publishes its own per-bay
contracts. Simulation builders must consume the matching contract rather than
reuse one generic box for all docks.

The calibrated stock-gripper wrapper pose is solved from the live child-geom
transform and the released CAD datum. `sim/validate_cad_clearance.py` records
that composed transform and rejects a nonzero composition residual. It also
checks the complete 0–80 mm rack path, using a focused 0.10 mm grid for the cam
corridor; the 0.50 mm sampled clearance yields a conservative 0.45 mm
continuous certificate against the 0.20 mm manufacturing floor.

Core exports are timestamp-canonicalized and hash-closed. To verify byte
reproducibility in two temporary directories, run `generate_cad.py` twice with
`--output-dir`; the canonical checked-in generation and exact report are:

```bash
XDG_CACHE_HOME=/tmp/cq-cache .venv/bin/python \
  QuickChange/SO101_Magnetic/generate_cad.py
XDG_CACHE_HOME=/tmp/cq-cache .venv/bin/python \
  QuickChange/SO101_Magnetic/sim/validate_cad_clearance.py
```

`core_cad_manifest.json` excludes itself to avoid circular hashing, but records
the generator, deterministic inventory digest, every contained artifact's
repo-relative path/byte count/SHA-256, and all three geometry contracts. The
clearance report separately pins the manifest file record and independently
recomputes its contents.

## Full SO-101 detachable-gripper MuJoCo example

`sim/so101_gripper_change_demo.py` builds on the calibrated upstream
`Simulation/SO101/so101_new_calib.xml` rather than a simplified arm. At load
time it makes one deliberate topology change:

- motor 5's `wrist_roll` joint stays on the robot and receives the v0.2 robot
  plate;
- the exact stock fixed-gripper and moving-jaw mesh subtree, including the
  original `gripper` joint and actuator, moves onto
  `so101_stock_gripper_tool_plate.stl` as a free docked tool;
- that existing gripper actuator represents the normal Feetech ID-6 control
  path after the pogo-pin bus is connected.

The deterministic sequence begins with a 55 mm gap, approaches the dock with
constant coupling orientation, changes from a rack weld to magnetic capture,
changes to the fail-locked positive-lock constraint as the arm withdraws, then
opens and closes the stock jaw through the unchanged gripper actuator. It then
returns the still-locked tool to the rack, switches off the simulated bus,
opens the rack-actuated lock, and withdraws the bare wrist. It exits nonzero
unless the complete attach/lock/use/return/release cycle, 45+ mm withdrawal in
both directions, coupling retention, and at least 0.8 rad of ID-6 jaw travel
are all observed.

From the repository root:

```bash
.venv/bin/pip install -r Simulation/SO101/requirements.txt
.venv/bin/python QuickChange/SO101_Magnetic/sim/so101_gripper_change_demo.py
```

For CI or a machine without a display:

```bash
.venv/bin/python QuickChange/SO101_Magnetic/sim/so101_gripper_change_demo.py --headless
```

To record the same MuJoCo state sequence on a headless machine without EGL,
OSMesa, X11, or a GPU, use the deterministic CPU z-buffer renderer (system
`ffmpeg` is required):

```bash
.venv/bin/python QuickChange/SO101_Magnetic/sim/render_gripper_change_video.py
```

The recorder produces a split-screen view of the complete arm and a tracked
coupler close-up. It renders MuJoCo's compiled geometry and live body poses; it
does not reuse the discarded conceptual animation.

The XML file beside the controller is a scene overlay. The controller merges it
with the upstream robot in memory so there is no duplicated SO-101 description
to fall out of sync. `--save-preview PATH.png` is optional and requires a
working OpenGL/EGL/OSMesa backend.

This simulation intentionally treats magnetic seating and the positive lock as
switched weld constraints. It checks kinematics, model topology, sequencing,
retention, and continued jaw control. It does **not** simulate field strength,
rack-cam contact, slider motion, electrical hot-plug behavior, compliance,
strength, fatigue, or wear. The ID-6 handshake is a state-machine gate rather
than a serial-protocol emulation, and quick-changer/gripper collision geometry
is disabled, so this demo does not prove physical clearance. The physical
validation plan below still applies.

## Retrofitting the standard gripper

The official assembly guide attaches the gripper to motor 5's wrist horn with
four M3x6 screws. This adapter uses that same interface and does not require
drilling or cutting the arm.

1. Power the arm down and remove the four M3x6 gripper-to-horn screws.
2. Fit the robot plate to the existing horn with four M3x10 socket-head screws.
   Confirm at least 3 mm thread engagement and no bottoming on the actual horn
   before tightening; shorten only if the particular horn requires it.
3. Install four Ruthex RX-M3x5.7 inserts in the rear of
   `so101_stock_gripper_tool_plate`.
4. Drop the two DIN 934 M3 pull-stud nuts into their rear hex pockets and fit
   the two shoulder screws from the coupling side. The gripper body captures
   the nuts when installed.
5. Attach the original printed gripper body to the tool plate with its original
   four M3x6 screws.
6. Terminate the gripper motor's three conductors on P1-P3 of the tool PCB:
   GND, +12 V, TTL_DATA. Use the four plated wire holes accessible from the rear.
7. Add a short, reversible adapter harness from the wrist-roll motor's outgoing
   gripper connector to the four robot-side pogo pins. Avoid cutting the stock
   cable; use mating 3-pin housings or a small inline adapter.

The stock gripper keeps ID 6 and is still controlled through the normal LeRobot
Feetech bus. P4 is unused for this first tool. A later controller can read P4
with a pull-up and a tool-specific resistor to ground for passive tool ID.

### Electrical mating caveat

The cited Feetech material specifies 12 V operation and half-duplex serial
control, but does not qualify the servo connector for live hot-plugging. For
the minimum retrofit, command ID 6 torque off, stop bus traffic, mate or unmate,
then verify communication before re-enabling torque. For unattended repeated
tool changing, add a robot-side switched/e-fused +12 V tool branch and enable it
only after mechanical attachment; this is an electrical reliability upgrade,
not an electromagnet.

The supplied target PCB is functional rather than just a drawing: each 4 mm
ENIG target is routed to a rear-accessible plated hole. Fabricate it as 1.0 mm
FR-4, 2 oz copper, ENIG. `electrical_pinout.csv`, the KiCad source, and an SVG
fab drawing are included. Retain the PCB by bonding only its perimeter into the
recess with an electronics-compatible epoxy; keep the ENIG targets free of
adhesive and verify that the board remains flush.

## Purchasable hardware

| Qty | Side | Part | Geometry / purpose |
|---:|---|---|---|
| 2 | robot | [Supermagnete CS-Q-12-12-04-N](https://www.supermagnete.de/eng/screw-on-neodymium-magnets/screw-on-block-magnet-12-x-12-x-4mm_CS-Q-12-12-04-N), EAN 7640172691830 | N35 NdFeB, 12 x 12 x 4 mm, M3 countersunk hole, 29.4 N nominal axial force each |
| 2/tool | tool | [Supermagnete MC-12-12-03](https://www.supermagnete.de/eng/magnet-counterparts-to-screw-on/metal-plates-with-countersunk-hole-12-x-12-x-3mm_MC-12-12-03), EAN 7640172691892 | Q235 target, 12 x 12 x 3 mm, M5 countersunk hole |
| 2/tool | tool | [McMaster 90318A720](https://www.mcmaster.com/90318A720/) | 316 SS ultra-low-profile shoulder screw; M3 thread, 4 mm shoulder diameter, 5 mm shoulder length, 6 x 1.3 mm head, 4 mm thread length |
| 2/tool | tool | [DIN 934 M3 nut](https://accu-components.com/us/hexagon-nuts/7888-HPN-M3-A2) | 5.5 mm across flats x 2.4 mm; captive pull-stud retention |
| 1 | robot | [MISUMI E-GUL4-10](https://us.misumi-ec.com/vona2/detail/110310903689/?HissuCode=E-GUL4-10) | 304 SS compression spring, OD 4 x 10 mm free, 0.98 N/mm, 4 mm permitted deflection |
| 4 | robot | [Mill-Max 7983-1-15-20-75-14-11-0](https://www.mill-max.com/products/new/high-current-small-scale-spring-loaded-pins) | Solder-cup spring pin, 1.397 mm full stroke, 0.7 mm midstroke; 8 A max / 6.4 A derated catalog values |
| 4 | gripper tool | [Ruthex RX-M3x5.7](https://www.ruthex.de/en/products/ruthex-gewindeeinsatz-m3-100-stuck-rx-m3x5-7-messing-gewindebuchsen) | M3 heat-set inserts for the original gripper-hole pattern |

Standard fasteners per robot: four M3x10 socket-head wrist screws, two ISO
10642 M3x10 countersunk magnet screws with two DIN 934 M3 nuts, and one flush
M2x6 slider guide screw. Per tool: two ISO 10642 M5x10 countersunk target screws
with two DIN 934 M5 nuts. Tighten the brittle magnets by hand only.

## Load basis and why the lock is required

There is no whole-arm payload figure in the official SO-101 documentation used
for this design. The official follower uses six STS3215 motors; Feetech gives
the servo a **10 kg.cm rated torque** and **30 kg.cm peak stall torque** at 12 V,
equivalent to 0.9807 and 2.942 N.m. Servo torque is not an arm payload rating:
arm pose, link mass, lever arm, acceleration, thermal limits, print strength,
and collision loads all matter.

For intuition only, 1 kg at a 100 mm horizontal lever is about 0.981 N.m before
adding the tool, adapter, robot links, or dynamics. It therefore must not be
presented as a supported 1 kg SO-101 payload.

The generated stock-gripper pair contains about 46.3 cm3 of printed plate
material, roughly 47 g if made from 1.01 g/cm3 PA12. Magnets, targets, slider,
contacts, PCB, nuts, screws, spring, and harness will put the real adapter pair
roughly in the 65-75 g range; weigh the finished parts and include that mass and
its offset in the robot load model.

The two selected magnets have 58.8 N combined catalog pull only at their ideal
test condition. With a 16 mm moment arm, the optimistic magnet-only pry estimate
is 0.9408 N.m—already below the servo's 0.9807 N.m rated-torque reference. Air
gaps, coatings, edge loading, and shock reduce real holding force further.
Magnets alone are therefore rejected.

The positive lock is sized to be physically proof-tested at 2.942 N.m, three
times the servo rated-torque reference. At a 24 mm interface edge that is about
122.6 N total reaction, or 61.3 N per pull stud if sharing is equal. This is a
**test target**, not a certification or a claim that every printed assembly
will pass. Test both bending axes because load sharing will not be perfectly
equal.

A conventional electromagnet is not recommended here: it adds wiring, heat,
and a drop-on-power-loss failure mode. Permanent magnets plus the spring-closed
metal lock are simpler and fail locked. Consider an electro-permanent magnet
only if release away from the rack becomes a hard requirement.

Official references: [SO-101 assembly and motor IDs](https://huggingface.co/docs/lerobot/so101),
[Feetech STS3215 specifications](https://www.feetechrc.com/525603.html), and
[Feetech three-wire pinout PDF](https://www.feetechrc.com/Data/feetechrc/upload/file/20240428/6384991403763545451893182.pdf).

## Fabrication

- Print robot and tool plates in PA12/SLS nylon or a well-characterized PA-CF
  process for load tests. Tough PETG is acceptable for early light-duty
  prototypes; PLA is for fit checks only.
- For FDM, start at 0.2 mm layers, at least five perimeters, five top/bottom
  layers, and 40% infill. Print the large rear face down and mating features up.
- Laser- or waterjet-cut the working slider from 1.5-1.6 mm 304 stainless using
  `so101_positive_lock_slider_profile.dxf`. The slider STL is a fit-check model,
  not the working load-bearing part. Deburr and polish both faces and keyholes.
- Ream the printed pogo pilots to the 1.575 mm Mill-Max recommendation only
  after a fit coupon establishes the needed compensation for the chosen print
  process. Do not force the pins through undersized holes.
- Keep magnets and steel targets flush to 0.05 mm below their mating faces so
  impact lands on printed face lands, not brittle nickel plating.
- Check that the slider moves the full 3.0 mm without binding. At the current
  endpoints the spring is 9.4 mm locked and 6.4 mm unlocked: 3.6 mm maximum
  compression, leaving 0.4 mm margin to the catalog deflection limit.

## Generate, inspect, and simulate

From the repository root:

```bash
.venv/bin/pip install -r Simulation/SO101/requirements.txt
.venv/bin/python QuickChange/SO101_Magnetic/generate_cad.py
.venv/bin/python QuickChange/SO101_Magnetic/load_check.py
.venv/bin/python QuickChange/SO101_Magnetic/sim/so101_gripper_change_demo.py --headless
```

The CAD generator exports printable STEP/STL parts, a stainless-slider DXF,
reference models for every selected special component, complete assembly STEP
files, design and engineering JSON, and PCB files. The older isolated-coupler
demo remains useful for constraint debugging, but the full-arm demo is the
integration example and acceptance check.

## Required physical validation

This remains a prototype, not a safety-rated tool changer. Before robot use:

1. Gauge every printed pocket, locator, stud, slider, and pogo fit before
   installing magnets or electronics.
2. Perform axial pull, shear, and edge-pry tests on the assembled interface.
3. Proof both bending axes to 2.942 N.m behind a guard, then inspect the printed
   roof, captive-nut floors, stud heads, slider keyholes, and fasteners.
4. Cycle attach/lock/unlock/release at least 1,000 times while monitoring slider
   travel, contact resistance, pogo temperature, magnet damage, and screw
   loosening.
5. Start robot trials at low speed over a padded surface with a tether and a
   payload far below the demonstrated test limit. Never put people under a
   retained tool or payload.
