# SO-101 powered magnetic quick changer v0.2

This is a retrofit, passively docked quick changer for an existing SO-101. Two
permanent magnets perform forgiving capture and face preload; they are **not**
the safety lock. A spring-return stainless keyhole slider positively captures
two metal pull studs. The passive rack cams that slider open. A narrow axial
lead couples the last 3.2 mm of approach to a 0.20-to-0.00 mm lateral recenter;
the existing X/Y wedge then controls spring return during rack exit. No release
servo, solenoid, or electromagnet is required.

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
  by 3 mm so 4.25 mm necks sit under the 6 mm stud heads. Each neck is a 7.25 mm
  capsule covering the complete 3 mm shoulder-centre path, not a short
  centre-to-centre rectangle. It provides 0.125 mm radial shoulder clearance
  and 0.875 mm radial head-retention overlap. The printed roof, not the M2 guide
  screw, reacts separation load. Loss of power leaves it locked.
- **Cam corridor:** the passive cam retains its `x = 24.05 mm` unlock datum and
  rack-exit wedge. Its integral source lead is a 45-degree ruled loft from
  `x = 27.25` at `z = -9.60` mm to `x = 24.05` at `z = -6.40` mm over the
  narrow `y = 0..2` mm tab land. A vertical finger joins the main wedge, and a
  1 mm root bridge at `x = 28..29` mm lies beyond the locked tab's `x = 27` mm
  maximum. The full-depth robot-plate recess preserves the certified approach
  clearance without changing slider travel, keyhole alignment, or retention.
- **Electrical interface:** four Mill-Max pins on 5 mm pitch: GND, +12 V,
  TTL_DATA, and a spare TOOL_ID line. GND protrudes 0.2 mm farther so it mates
  first and breaks last.

In the CAD and MuJoCo model, purple identifies the two magnets, silver their
targets and pull studs, green the positive-lock slider, and gold the four
contacts. Use the full-arm MuJoCo example below as the executable acceptance
check.

## Attach and detach sequence

1. Approach at the published +0.20 mm open-side offset. From 6.4 to 3.2 mm
   preseated, recenter linearly to zero while the 45-degree lead opens the
   slider to `q <= 0.05 mm`. This finishes before the stud heads first reach
   the slider plane at 3.1 mm preseated.
2. Continue axially at zero offset. Tapered locators enter first; magnets then
   close the final gap while the hold finger keeps the keyhole entries open.
3. Verify contact—for the first gripper, read servo ID 6 with torque still off.
4. Translate in dock-local -Y. The main wedge permits spring return after the
   first 2 mm; `q = 3 mm` is reached at 13.9494 mm and has 0.2518 mm exact cam
   clearance at the nominal 15 mm witness.
5. Enable the tool and work normally.
6. To detach, command gripper torque off and stop bus packets, then return the
   still-locked assembly to the rack.
7. During the final rack travel, the fixed wedge pushes the exposed tab 3 mm,
   aligning both large keyholes with the stud heads.
8. The rack retains the tool plate while the arm withdraws axially. Contacts and
   magnets separate; no powered release actuator is involved.

The MuJoCo model uses conditional welds for capture and lock state. It validates
this state sequence and geometry; it does not predict magnetic fields, contact
arcing, printed-part strength, fatigue, or wear.

## CAD/simulation geometry contract

`generate_cad.py` publishes the exact core dock-stop bounds, stock-gripper STEP
mount, full-depth robot-plate cam-relief bounds, swept keyhole-capsule contract,
and executable passive-cam `p/x/q` and `-Y/q` laws in both
`design_parameters.json` and `exports/core_cad_manifest.json`. The relief
contract also records the 0.20 mm guided approach offset and retained 8.225 mm
stud-well / 7.150 mm slider-lobe ligaments. The stock dock stop is not interchangeable
with the spoon/whisk stops: core bounds are `X[-45,37]`, `Y[26,32]`,
`Z[-3,12.5]` mm, while the two-bay matcha package publishes its own per-bay
contracts. Simulation builders must consume the matching contract rather than
reuse one generic box for all docks.

The calibrated stock-gripper wrapper pose is solved from the live child-geom
transform and the released CAD datum. `sim/validate_cad_clearance.py` records
that composed transform and rejects a nonzero composition residual. It also
checks the complete 0–80 mm rack path, using a focused 0.10 mm grid for the cam
corridor; the 0.50 mm sampled clearance yields a conservative 0.45 mm
continuous bound. A separate exact OCCT sweep checks both 4 mm shoulders every
0.05 mm from unlocked through the full 3.0 mm stroke, verifies the analytic
0.125 mm continuous capsule clearance, confirms that both 6.5 mm entries pass
the heads when unlocked, and confirms projected head retention when locked.
The 15→0 mm capture sweep follows the coupled recenter rather than assuming a
fixed lateral offset. Its 0.3000 mm sampled robot-plate/cam gap minus the full
two-axis half-step motion certifies 0.249902 mm continuous clearance. A tighter
0.01 mm slider/stud sweep certifies 0.205264 mm continuous clearance before the
3.1 mm head-entry event. OCCT also requires zero cam/stud overlap throughout
capture, full component closure, passive -Y return, and at least 0.20 mm q=3
cam clearance at the 15 mm exit witness.

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
repo-relative path/byte count/SHA-256, and every published geometry contract. The
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
| 4 | robot | [Mill-Max 7983-1-15-20-75-14-11-0](https://www.mill-max.com/products/discrete-spring-loaded-pins/spring-loaded-pin-with-solder-cup-termination/7983/7983-1-15-20-75-14-11-0) | Solder-cup spring pin, drawing Ø1.0668 mm plunger and 1.397 ± 0.127 mm full stroke. This design selects solder-cup-first knurl retention: Ø1.58 mm land, separate Ø2.31 mm body counterbore, and shoulder hard-stop. Process fit/pull-out and installed reliability remain unqualified. |
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
- The CAD now implements one specific Mill-Max mounting mode rather than a
  straight pilot: insert the solder-cup side first through a Ø1.58 mm knurl
  land into a separate Ø2.31 mm body counterbore, stopping the shoulder on
  the signal-specific internal ledge. This is dimensionally reconstructed from
  the official drawing and [press-fit application note](https://www.mill-max.com/sites/default/files/external/assets/2020-10/spring-loaded_solder-cup_pin_2.pdf),
  but it is **not released for fabrication** until a process-specific coupon
  establishes bore tolerance, insertion force, and pull-out retention. The
  nominal GND shoulder datum is 0.20 mm ahead; four independent ±0.1524 mm
  drawing-length terms yield a conservative worst-case lead of -0.4096 mm, so
  first-mate is not qualified.
- Manufacturer artwork is not redistributed because its redistribution terms
  were not established. The derived ledger at
  `source_authority/millmax_7983/authority_ledger.json` records the official
  URLs, byte counts, SHA-256 digests, exact inch callouts, and this provenance
  limitation. The reconstructed solids are official-drawing-derived nominal
  collision envelopes, not official Mill-Max 3D CAD; deterministic generation
  uses the checked-in ledger and never silently refetches mutable web content.
- Keep magnets and steel targets flush to 0.05 mm below their mating faces so
  impact lands on printed face lands, not brittle nickel plating.
- The current exact CAD leaves 0.0499 mm after its route interval and 0.20 mm
  clearance deductions using an unqualified local motion allowance. That
  number is arithmetic residue, not a sampled route or an SLS/FDM or
  catalog-part tolerance. Release remains blocked until a process/fit coupon
  qualifies the combined error budget. The pogo exterior is a conservative
  nominal drawing-derived envelope with unqualified part/process tolerances;
  knurl pull-out, installed electrical cycling, and magnetic fastener
  seating/preload still need physical evidence. Do not fabricate the
  electrical mounting interface from this checkpoint.
- Check that the slider moves the full 3.0 mm without binding. At the current
  endpoints the spring is 9.4 mm locked and 6.4 mm unlocked: 3.6 mm maximum
  compression, leaving 0.4 mm margin to the catalog deflection limit.

## Rolled core-dock floor-support source checkpoint

The core interface now has a source-only installation contract for a
`-87.21086925015224 deg` tool-view roll about the mating normal. Its published
world pose is `(0.19082795371216685, 0.1330713713445051,
0.1939154579377553) m`, quaternion `wxyz=(0.6440855284765126,
-0.6440855284765125, 0.2918112952014223, -0.2918112952014225)`. In that frame,
dock-local `-Y` is world-up. The exact 0–15 mm release continuation has 31
rows at 0.5 mm, SHA-256
`f30b0c178917945fcd45358710e5127302bc5240ca6cb4cdaa7f49d16c4f0293`,
and a 0.2360031833 deg maximum joint step.

The dock remains a separate one-solid BRep and bolts to a hollow floor
pedestal. The pedestal is a one-solid 43 mm square / 35 mm square hollow post
with 4 mm walls, a 100 x 80 x 8 mm floor plate, a 56 x 8 x 8.5 mm head, and a
right bolt reinforcement. Its exact post/head and post/base positive overlaps
are 716.8886804667 and 1248 mm3 after the hardware cuts. Two countersunk M4
fasteners join the stop to the head; four countersunk M6 fasteners join the
base to a future tapped fixture. The source BReps close at:

- stop: 7379.269784962569 mm3;
- complete dock: 21743.904784962568 mm3;
- support: 162415.4180526403 mm3;
- installed printed total: 184159.32283760287 mm3.

The earlier 162308.50715898623 mm3 support estimate was not the volume of the
specified Boolean construction and is superseded. No material mass is claimed
until material condition and density are selected. The minimum fixed-feature
clearance is 1.0 mm; the explicit 0.70 mm nominal tolerance allocation leaves
0.30 mm arithmetic residue, but the print process is not qualified. The
screening-only 1.5 GPa load proxy gives 4.25271213611 N.m combined moment,
0.5720 MPa bending stress, and 0.1940 mm tip deflection; these are not material
or joint allowables.

This checkpoint intentionally does not regenerate STEP/STL, manifests,
reports, or runtime placements. `release_ready` remains false pending sourced
M4/M6 hardware, an authoritative tapped floor substrate, PA12
modulus/strength/creep and tolerance evidence, anchor-strength tests, runtime
full-arm regeneration, and physical cam/contact/friction and reverse-insertion
validation.

## Generate, inspect, and simulate

From the repository root:

```bash
.venv/bin/pip install -r Simulation/SO101/requirements.txt
.venv/bin/python QuickChange/SO101_Magnetic/generate_cad.py
.venv/bin/python QuickChange/SO101_Magnetic/load_check.py
.venv/bin/python QuickChange/SO101_Magnetic/sim/so101_gripper_change_demo.py --headless
```

The CAD generator exports STEP/STL geometry, a stainless-slider DXF,
illustrative assembly STEP files, design and engineering JSON, and PCB files.
The assemblies are not complete hardware authorities: several fasteners, the
harness, and the exact purchased pogo section are still absent. The older
isolated-coupler demo remains useful for constraint debugging, while the
full-arm demo is a development integration example, not an acceptance check.
The clearance validator deliberately keeps `release_ready` false until the
machine-readable interface blockers are closed.

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
