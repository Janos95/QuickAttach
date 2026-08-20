"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  CadAsset,
  ControllerState,
  GeometryView,
  QuickAttachApi,
  QuickAttachRenderer,
  ToolName,
} from "./quickattach-renderer";

declare global {
  interface Window {
    quickAttach?: QuickAttachApi | { ready: false };
  }
}

export default function Home() {
  const viewportRef = useRef<HTMLDivElement>(null);
  const runtimeRef = useRef<QuickAttachRenderer | null>(null);
  const [runtime, setRuntime] = useState<QuickAttachRenderer | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [error, setError] = useState("");
  const [sceneState, setSceneState] = useState<ControllerState | null>(null);
  const [tab, setTab] = useState<"scene" | "cad">("scene");
  const [cadQuery, setCadQuery] = useState("");
  const headless = useMemo(
    () => typeof window !== "undefined" && new URLSearchParams(window.location.search).has("headless"),
    [],
  );

  useEffect(() => {
    let cancelled = false;
    window.quickAttach = { ready: false };

    const start = async () => {
      try {
        if (!viewportRef.current) return;
        const runtime = await QuickAttachRenderer.create(viewportRef.current);
        if (cancelled) {
          runtime.dispose();
          return;
        }
        runtimeRef.current = runtime;
        setRuntime(runtime);
        runtime.subscribe(setSceneState);
        window.quickAttach = runtime.api();
        setStatus("ready");
      } catch (reason) {
        console.error(reason);
        setError(reason instanceof Error ? reason.message : String(reason));
        setStatus("error");
      }
    };

    start();
    return () => {
      cancelled = true;
      runtimeRef.current?.dispose();
      runtimeRef.current = null;
      window.quickAttach = { ready: false };
    };
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") runtimeRef.current?.selectTool(null);
      if (event.key.toLowerCase() === "g") runtimeRef.current?.setGizmoMode("translate");
      if (event.key.toLowerCase() === "r") runtimeRef.current?.setGizmoMode("rotate");
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const toolLabels: Record<ToolName, string> = {
    gripper: "Gripper",
    spoon: "Matcha spoon",
    whisk: "Whisk",
  };
  const selectedTool = sceneState?.selectedTool ?? null;
  const attached = Boolean(selectedTool && sceneState?.attachedTool === selectedTool);
  const attachmentError = runtime?.getAttachmentError() ?? null;
  const canAttach = runtime?.canAttach() ?? false;
  const jointLabels: Record<string, string> = {
    shoulder_pan: "Base",
    shoulder_lift: "Shoulder",
    elbow_flex: "Elbow",
    wrist_flex: "Wrist",
    wrist_roll: "Roll",
  };
  const cadAssets = runtime?.manifest.cadAssets ?? [];
  const selectedCad = cadAssets.find((asset) => asset.id === sceneState?.cadAssetId) ?? cadAssets[0];
  const filteredCad = cadAssets.filter((asset) => {
    const query = cadQuery.trim().toLowerCase();
    return !query || `${asset.label} ${asset.sourceFile} ${asset.category}`.toLowerCase().includes(query);
  });
  const cadCategories = Array.from(new Set(filteredCad.map((asset) => asset.category)));

  const switchTab = (next: "scene" | "cad") => {
    setTab(next);
    if (next === "scene") runtimeRef.current?.showScene();
    else {
      const asset = selectedCad ?? cadAssets[0];
      if (asset) void runtimeRef.current?.inspectCad(asset.id);
    }
  };

  const selectCad = (asset: CadAsset) => {
    void runtimeRef.current?.inspectCad(asset.id);
  };

  const setGeometryView = (view: GeometryView) => {
    runtimeRef.current?.setGeometryView(view);
  };

  return (
    <main suppressHydrationWarning className={headless ? "scene-page headless" : "scene-page"}>
      <div className="viewport-wrap">
        <div
          ref={viewportRef}
          className="render-surface"
          aria-label="Interactive 3D view of the SO-101 robot and radial tool fixture"
        />

        {status !== "ready" && (
          <div className="loading-card">
            {status === "loading" ? <span className="spinner" /> : <span className="error-mark">!</span>}
            <strong>{status === "loading" ? "Loading scene" : "Scene unavailable"}</strong>
            {status === "error" && <p>{error}</p>}
          </div>
        )}

        {!headless && status === "ready" && (
          <nav className="mode-tabs" aria-label="QuickAttach views">
            <button className={tab === "scene" ? "active" : ""} onClick={() => switchTab("scene")}>Scene</button>
            <button className={tab === "cad" ? "active" : ""} onClick={() => switchTab("cad")}>CAD inspector</button>
          </nav>
        )}

        {!headless && status === "ready" && tab === "scene" && (
          <>
            <div className="tool-ui" aria-live="polite">
              <div className="tool-status">
                <strong>{selectedTool ? toolLabels[selectedTool] : "Select a tool"}</strong>
                <span>
                  {!selectedTool
                    ? "Click a tool to place the gizmo on its flange"
                    : attached
                      ? "Attached to the wrist"
                      : canAttach
                        ? `Within capture range · ${Math.round(attachmentError?.positionMm ?? 0)} mm`
                        : attachmentError
                          ? `Drag to wrist · ${Math.round(attachmentError.positionMm)} mm away`
                          : "Drag the flange gizmo near the bare wrist"}
                </span>
              </div>
              {selectedTool && (
                <div className="tool-actions" aria-label="Selected tool actions">
                  {!attached && (
                    <>
                      <button
                        className={sceneState?.gizmoMode === "translate" ? "active" : ""}
                        onClick={() => runtimeRef.current?.setGizmoMode("translate")}
                        title="Move gizmo (G)"
                      >Move</button>
                      <button
                        className={sceneState?.gizmoMode === "rotate" ? "active" : ""}
                        onClick={() => runtimeRef.current?.setGizmoMode("rotate")}
                        title="Rotate gizmo (R)"
                      >Rotate</button>
                      <button onClick={() => runtimeRef.current?.returnSelectedToRack()}>Rack</button>
                      <button
                        className="attach-action"
                        disabled={!canAttach}
                        onClick={() => runtimeRef.current?.attachSelected()}
                      >Attach</button>
                    </>
                  )}
                  {attached && (
                    <button
                      className="detach-action"
                      onClick={() => runtimeRef.current?.releaseAttached()}
                    >Detach</button>
                  )}
                </div>
              )}
            </div>

            <div className="camera-ui">
              <div className="view-actions" aria-label="Camera views">
                <button onClick={() => runtimeRef.current?.resetCamera()}>Perspective</button>
                <button onClick={() => runtimeRef.current?.topCamera()}>Top view</button>
              </div>
              <p>Drag scene to orbit · Click a tool to manipulate its flange</p>
            </div>

            <details className="arm-ui">
              <summary>Move robot</summary>
              <div className="arm-controls">
                <div className="arm-heading">
                  <strong>Arm joints</strong>
                  <div className="arm-presets" aria-label="Robot pose presets">
                    <button onClick={() => runtimeRef.current?.setPreset("home")}>Home</button>
                    <button onClick={() => runtimeRef.current?.setPreset("parked")}>Park</button>
                  </div>
                </div>
                {runtime.manifest.armJoints.map((joint) => {
                  const range = runtime.jointRanges[joint];
                  const value = sceneState?.joints[joint] ?? 0;
                  return (
                    <label className="joint-control" key={joint}>
                      <span>{jointLabels[joint] ?? joint}</span>
                      <input
                        aria-label={`${jointLabels[joint] ?? joint} joint`}
                        type="range"
                        min={range[0]}
                        max={range[1]}
                        step="0.01"
                        value={value}
                        onChange={(event) => runtimeRef.current?.setJoint(joint, Number(event.target.value))}
                      />
                      <output>{Math.round(value * 180 / Math.PI)}°</output>
                    </label>
                  );
                })}
              </div>
            </details>
          </>
        )}


        {!headless && status === "ready" && tab === "cad" && (
          <>
            <aside className="cad-browser" aria-label="CAD file browser">
              <div className="cad-browser-heading">
                <div>
                  <strong>CAD files</strong>
                  <span>{cadAssets.length} production parts</span>
                </div>
                <input
                  type="search"
                  value={cadQuery}
                  onChange={(event) => setCadQuery(event.target.value)}
                  placeholder="Filter"
                  aria-label="Filter CAD files"
                />
              </div>
              <div className="cad-file-list">
                {cadCategories.map((category) => (
                  <section key={category}>
                    <h2>{category}</h2>
                    {filteredCad.filter((asset) => asset.category === category).map((asset) => (
                      <button
                        key={asset.id}
                        className={selectedCad?.id === asset.id ? "active" : ""}
                        onClick={() => selectCad(asset)}
                      >
                        <span>{asset.label}</span>
                        <small>{asset.sourceFile}</small>
                      </button>
                    ))}
                  </section>
                ))}
              </div>
            </aside>

            <div className="cad-view-controls">
              <div className="geometry-toggle" aria-label="Geometry representation">
                {(["cad", "collision", "overlay"] as GeometryView[]).map((view) => (
                  <button
                    key={view}
                    className={sceneState?.geometryView === view ? "active" : ""}
                    onClick={() => setGeometryView(view)}
                  >{view === "cad" ? "CAD" : view === "collision" ? "Collision" : "Overlay"}</button>
                ))}
              </div>
              <button
                className="cad-refit"
                disabled={!selectedCad}
                onClick={() => selectedCad && selectCad(selectedCad)}
              >Fit view</button>
            </div>

            {selectedCad && (
              <div className="cad-file-info" aria-live="polite">
                <strong>{selectedCad.label}</strong>
                <span>{selectedCad.sourceFile}</span>
                <div>
                  <b>{sceneState?.cadLoading ? "Loading…" : `${sceneState?.cadCollisionCount ?? 0} collision geoms`}</b>
                  {selectedCad.printEnvelopeMm && <b>{selectedCad.printEnvelopeMm.map(Math.round).join(" × ")} mm</b>}
                </div>
              </div>
            )}

            <p className="cad-hint">Drag to orbit · Scroll to zoom · Compare the source tessellation with emitted MuJoCo geometry</p>
          </>
        )}
      </div>
    </main>
  );
}
