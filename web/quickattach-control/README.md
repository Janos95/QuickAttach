# QuickAttach Robot Control

An interactive Three.js view of the source-bound SO-101 QuickAttach MuJoCo
workcell. The robot starts with a bare wrist. Three detachable end effectors
sit in a printable, bench-fastened rack with their mating flanges open toward
the robot.

There is no scripted animation. In the page you can:

- orbit freely or switch to a top view;
- open the compact **Move robot** control and move all five arm joints;
- click any tool to place a translate/rotate gizmo on its mating flange;
- drag that flange to the wrist, attach it inside the capture range, detach it,
  and return it to its rack position.

The rack is a ten-part rounded under-slung truss rather than a collection of
solid blocks. It uses three hollow A-frame stations, separate flat-printing
T-saddles, and four removable base links. The printed assembly is 244.01 cm³,
its largest part dimension is 190.39 mm, and the generated solids are valid and
non-overlapping. Existing dock datums are retained: the gripper fastens through
the source support-head M4 centers, while spoon and whisk use two short metal
stays into their existing rear-anchor blocks. See [PRINTING.md](PRINTING.md) for
the generated parts, hardware, and assembly notes.

The geometry has passed digital checks, including a 486-sample sweep across the
parked-to-approach and approach-to-seat paths for all three tools with zero
fixture or adapter contacts. This is not a structural load rating or physical
qualification; print and bench-test one station before relying on the complete
rack.

The deployed owner-only build is at
[quickattach-control.janos95.chatgpt.site](https://quickattach-control.janos95.chatgpt.site).

## Browser automation and rendering

The browser exposes a deterministic `window.quickAttach` API:

`getState`, `setState`, `setJoint`, `setPreset`, `setTool`, `selectTool`,
`setGizmoMode`, `getAttachmentError`, `attachSelected`, `releaseAttached`,
`returnSelectedToRack`, `moveToDock`, `step`, `reset`, `resetCamera`,
`topCamera`, and `renderFrame`.

Add `?headless=1` to show only the render surface.
`scripts/render-threejs-video.py` drives this API from Python Playwright,
captures fixed-size frames, and encodes them with FFmpeg. Three.js is therefore
the shared visual layer for the interactive page and headless video, while
MuJoCo/Python remain the numerical validation authority.

## Regenerating the model bundle

The exporter imports the source model from this QuickAttach checkout, builds
the printable rack with CadQuery, and writes the browser-ready MJCF, assets,
and manifests to `public/model/`.

```bash
XDG_CACHE_HOME=/tmp/quickattach-cache \
../../.venv/bin/python scripts/export-matcha-model.py --output public/model
```

## Development

Requires Node.js `>=22.13.0`.

```bash
npm run install:ci
npm run dev
npm run lint
npm test
```

The app is a vinext site. Source lives under `app/`; generated model assets live
under `public/model/`. `npm test` builds, validates the deployment artifact, and
checks rendered preview metadata.

## Rendering the video

Start the same page on a fixed local port:

```bash
npm run dev -- --port 4173
```

Then drive its public browser API from another terminal:

```bash
python3 scripts/render-threejs-video.py \
  --url http://127.0.0.1:4173/ \
  --output outputs/quickattach-threejs-workflow.mp4
```

Python Playwright, its Chromium browser, and FFmpeg must be installed. The
renderer captures the exact interactive Three.js scene; there is no separate
video-only renderer or scene definition.
