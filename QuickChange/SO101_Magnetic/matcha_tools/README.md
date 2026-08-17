# SO-101 matcha tools

This package is the CAD authority for two food-task end effectors on the
powered SO-101 magnetic quick changer:

| Tool | Fixed tool ID | TTL bus address | Working element |
|---|---:|---:|---|
| Matcha dose spoon | 21 | none | 0.70-0.85 mL stainless dosing cup |
| Actuated matcha whisk | 22 | 7 | eccentric-drive replaceable brush |

The existing stock gripper remains the third tool.  The rack reserves bays in
the order `gripper`, `spoon`, `whisk` at 96 mm pitch.

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

The source includes a three-bay rack with 0.50 mm wall clearance and 0.30 mm
rear-face clearance around the complete spoon and whisk envelopes.  Lower
ledges are tangent to the coupling face; upper ledges act only on the plate's
outer lands.  The passive lock cam stays below the tool face and acts on the
robot-side slider.

The gripper bay is presently a reserved position, not a clearance claim.  The
official stock-gripper body overhangs the adapter plate, so its keeper geometry
must be validated independently before the rack is released for fabrication.
This package will fail closed in the forthcoming exact rack report rather than
silently applying the spoon/whisk rail profile to that overhang.

## Seconds-scale source checks

From the repository root, after installing `Simulation/SO101/requirements.txt`:

```bash
XDG_CACHE_HOME=/tmp/cq-cache .venv/bin/python \
  QuickChange/SO101_Magnetic/matcha_tools/test_matcha_tool_cad.py
```

The focused test checks fixed IDs, exact common-plate identity, the complete
15-solid interface-hardware roster, unique component inventories, balance and
inertia arithmetic, spoon capacity, and rack source structure.  It is not an
exact swept-interference release test.

Generated STEP/STL assemblies, JSON ledgers, the single canonical hash-pinned
`exports/matcha_tool_manifest.json`, and the
exact rack-sweep report are intentionally produced only after this source
checkpoint is reviewed.  Until those artifacts and the stock-gripper bay are
green, this CAD is for simulation dry runs and fit coupons—not fabrication.
