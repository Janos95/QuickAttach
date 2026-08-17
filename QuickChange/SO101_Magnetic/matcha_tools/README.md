# SO-101 matcha tools

This package is the CAD authority for two food-task end effectors on the
powered SO-101 magnetic quick changer:

| Tool | Fixed tool ID | TTL bus address | Working element |
|---|---:|---:|---|
| Matcha dose spoon | 21 | none | 0.70-0.85 mL stainless dosing cup |
| Actuated matcha whisk | 22 | 7 | eccentric-drive replaceable brush |

The existing stock gripper remains the third tool, but stays on the separate
core quick-change dock and stock-adapter package.  This add-on rack has only
`spoon` and `whisk` bays at 96 mm pitch.  That boundary is intentional: the
recovered official fixed-body STEP does not include the moving jaw, so this
keeper must not be presented as exact authority for the whole gripper.

## Geometry authority

`generate_matcha_tool_cad.py` imports `../generate_cad.py` and calls
`tool_plate(stock_gripper=False)` directly.  It does not redraw the coupling.
Consequently the target pockets, tapered locator sockets, tangential relief,
positive-lock studs, contact-board recess, four tool-mount holes, and all face
datums remain the exact generic interface.

Every rigid interface item is a separate named solid:

- two MC-12-12-03 targets, two ISO 10642 M5x10 screws, and two DIN 934 M5 nuts;
- two McMaster 90318A720 shoulder screws and two DIN 934 M3 captive nuts;
- the 10 x 22 x 1 mm FR-4 target board and all four copper pads.

This naming is intentional: simulation conversion must create collision
geometry for the screws, nuts, targets, PCB, pads, carriers, actuator parts,
and working elements.  A visual-only assembly mesh is not an acceptable
collision replacement.

## Spoon

The spoon uses a centred 3.2 mm stainless stem retained by two opposed M3 set
screws.  Its shallow elliptical cup opens sideways, so the robot can scoop
powder without the tool shaft entering the canister.  The calculated cavity is
about 0.75 mL; powder dose still has to be calibrated by mass because packing
and humidity dominate a volumetric estimate.

## Whisk

The whisk source includes a 25 mm 12 V motor envelope, shaft, balanced
eccentric rotor, 4 mm crank pin, X carriage, axial compliance carriage and
spring, splash bellows, removable brush hub, individual bamboo-bristle solids,
and the bus-driver PCB.  The compliance stroke is 5.10 mm with authored limits
`[-5.05, +0.05] mm`.  The brush has a separate conservative collision envelope;
that envelope is marked `fabrication=false` and has zero mass, so it cannot be
mistaken for solid bamboo in the mass ledger.

The motor and driver masses are explicit design inputs (72 g and 8 g), not
densities inferred from their outer envelopes.  A finished tool must be
weighed and its measured COM/inertia substituted before controller tuning.

## Mass and balance

Each ledger publishes every component's volume, density or mass override,
centre of mass, and inertia about its own COM.  Assembly inertia is recomputed
with the parallel-axis theorem.  A computed steel slug in the right-side
carrier pocket cancels the electrical wing and PCB's negative-X moment.  The
focused test requires the resulting X/Y COM magnitude to be at most 0.05 mm.

PA12, FR-4, bamboo, silicone and electronics densities are clearly labelled as
design assumptions.  They are useful for dry-run dynamics, not fabrication
certificates.

## Rack status

The source includes a two-bay matcha rack with 0.50 mm wall clearance and 0.30 mm
rear-face clearance around the complete spoon and whisk envelopes.  The left
lower rail ends 0.30 mm outboard of the FR-4 board, preventing the board edge
from becoming a support land. Lower
ledges are tangent to the coupling face; upper ledges act only on the plate's
outer lands. The passive lock cam stays below the tool face and acts on the
robot-side slider. Both bays call the core generator's complete
`positive_lock_cam()` directly, including its axial lead, hold finger, root
bridge and rack-exit wedge; the matcha generator carries no shifted polygon or
second cam approximation. The same executable cam contract is stored in the
matcha manifest.

Each bay publishes its exact stop contract in the canonical manifest. In the
bay-local frame the stop bounds are `X[-41,33]`, `Y[25,31]`,
`Z[-3,12.5]` mm; the spoon and whisk records add their respective rack-X
offsets. These are intentionally distinct from the separate core stock-gripper
dock bounds and must remain distinct in simulation.

The gripper is not a reserved matcha-rack bay.  Its existing core adapter,
passive dock, retrofit assembly and moving-jaw mesh are hash-pinned in the
matcha report as an external scope boundary.  The report makes no claim that
the spoon/whisk keeper profile fits the gripper overhang.

## Seconds-scale source checks

From the repository root, after installing `Simulation/SO101/requirements.txt`:

```bash
XDG_CACHE_HOME=/tmp/cq-cache .venv/bin/python \
  QuickChange/SO101_Magnetic/matcha_tools/test_matcha_tool_cad.py
```

The focused test checks fixed IDs, exact common-plate and complete core-cam
identity, the complete
15-solid interface-hardware roster, unique component inventories, balance and
inertia arithmetic, spoon capacity, and rack source structure.  It is not an
exact swept-interference release test.

The rack preflight is also seconds-scale:

```bash
XDG_CACHE_HOME=/tmp/cq-cache .venv/bin/python \
  QuickChange/SO101_Magnetic/matcha_tools/validate_matcha_rack.py --preflight
XDG_CACHE_HOME=/tmp/cq-cache .venv/bin/python \
  QuickChange/SO101_Magnetic/matcha_tools/test_matcha_rack_validation.py
```

It closes every named spoon/whisk component against every rack component,
including adjacent bays and all seven whisk mechanism extrema.  The fast path
uses a rigorous continuous swept-AABB lower bound.  Only the six exact named
plate/ledge and plate/stop tangencies use an FCPW mesh screen followed by OCCT
B-rep distance and overlap-volume diagnostics.  FCPW is screening evidence,
not STEP clearance authority.  This report covers only the two matcha bays;
normal-gripper release remains the responsibility of the separate core
quick-change package.

## Reproducible exports

`generate_matcha_tool_cad.py` canonicalizes OCCT's wall-clock STEP header and
accepts `--output-dir`, so byte reproducibility can be checked in two temporary
directories without modifying the checked-in package.  The canonical run is:

```bash
XDG_CACHE_HOME=/tmp/cq-cache .venv/bin/python \
  QuickChange/SO101_Magnetic/matcha_tools/generate_matcha_tool_cad.py
XDG_CACHE_HOME=/tmp/cq-cache .venv/bin/python \
  QuickChange/SO101_Magnetic/matcha_tools/validate_matcha_rack.py --release-report
```

The single `exports/matcha_tool_manifest.json` records repo-relative paths,
roles, byte counts, and SHA-256 digests for all ten CAD/ledger artifacts.  The
rack report pins that manifest and independently recomputes the full collision
inventory.  It does not hash itself or create a circular manifest dependency.

Generated STEP/STL assemblies, JSON ledgers, the single canonical hash-pinned
`exports/matcha_tool_manifest.json`, and the exact rack-sweep report form one
release package. Any source or contract change makes those artifacts stale;
fabrication remains blocked until deterministic regeneration, manifest hash
closure, and the exact rack report all pass together.
