/* eslint-disable @typescript-eslint/no-explicit-any -- MuJoCo's Embind views are generated as any. */
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";
import { ConvexGeometry } from "three/examples/jsm/geometries/ConvexGeometry.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { SVGRenderer } from "three/examples/jsm/renderers/SVGRenderer.js";

export type ToolName = "gripper" | "spoon" | "whisk";
export type GeometryView = "cad" | "collision" | "overlay";

export type CadAsset = {
  id: string;
  label: string;
  category: string;
  asset: string;
  sourceFile: string;
  bodyRoots: string[];
  geomPrefixes: string[];
  printEnvelopeMm?: number[];
  volumeMm3?: number;
};

export type ControllerState = {
  joints: Record<string, number>;
  gripper: number;
  whiskAngle: number;
  selectedTool: ToolName | null;
  attachedTool: ToolName | null;
  gizmoMode: "translate" | "rotate";
  collisions: boolean;
  view: "scene" | "cad";
  cadAssetId: string | null;
  geometryView: GeometryView;
  cadLoading: boolean;
  cadCollisionCount: number;
};

type Manifest = {
  assets: string[];
  armJoints: string[];
  joints: Record<string, { range: [number, number]; qposAddress: number }>;
  presets: Record<string, number[]>;
  dockCapture: Record<ToolName, number[]>;
  cadAssets: CadAsset[];
  counts: { geoms: number };
  xmlSha256: string;
};

type RelativePose = {
  position: THREE.Vector3;
  rotation: THREE.Matrix3;
};

export type QuickAttachApi = {
  ready: true;
  getState: () => ControllerState;
  setJoint: (name: string, value: number) => ControllerState;
  setPreset: (name: string) => ControllerState;
  setState: (state: Partial<ControllerState>) => ControllerState;
  setTool: (tool: ToolName | null) => ControllerState;
  selectTool: (tool: ToolName | null) => ControllerState;
  moveToDock: (phase: "approach" | "seat") => ControllerState;
  attachSelected: () => ControllerState;
  releaseAttached: () => ControllerState;
  returnSelectedToRack: () => ControllerState;
  setGizmoMode: (mode: "translate" | "rotate") => ControllerState;
  getAttachmentError: () => {
    positionMm: number;
    angleDegrees: number;
  } | null;
  canAttach: () => boolean;
  canRelease: () => boolean;
  step: (count?: number) => ControllerState;
  reset: () => ControllerState;
  resetCamera: () => void;
  topCamera: () => void;
  renderFrame: () => ControllerState;
  showScene: () => ControllerState;
  inspectCad: (assetId: string) => Promise<ControllerState>;
  setGeometryView: (view: GeometryView) => ControllerState;
  setCaptureMode: (enabled: boolean) => void;
};

const MJ_GEOM = {
  PLANE: 0,
  SPHERE: 2,
  CAPSULE: 3,
  ELLIPSOID: 4,
  CYLINDER: 5,
  BOX: 6,
  MESH: 7,
} as const;

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

function matrix3FromRowMajor(values: ArrayLike<number>, offset = 0) {
  return new THREE.Matrix3().set(
    values[offset],
    values[offset + 1],
    values[offset + 2],
    values[offset + 3],
    values[offset + 4],
    values[offset + 5],
    values[offset + 6],
    values[offset + 7],
    values[offset + 8],
  );
}

function matrix4FromRotation(rotation: THREE.Matrix3) {
  const e = rotation.elements;
  return new THREE.Matrix4().set(
    e[0], e[3], e[6], 0,
    e[1], e[4], e[7], 0,
    e[2], e[5], e[8], 0,
    0, 0, 0, 1,
  );
}

function cloneState(state: ControllerState): ControllerState {
  return { ...state, joints: { ...state.joints } };
}

export class QuickAttachRenderer {
  readonly manifest: Manifest;
  readonly jointRanges: Record<string, [number, number]>;

  private readonly module: any;
  private readonly model: any;
  private readonly data: any;
  private readonly vfs: any;
  private readonly scene = new THREE.Scene();
  private readonly root = new THREE.Group();
  private readonly inspectorRoot = new THREE.Group();
  private readonly inspectorCad = new THREE.Group();
  private readonly inspectorCollision = new THREE.Group();
  private readonly stlLoader = new STLLoader();
  private readonly camera = new THREE.PerspectiveCamera(47, 1, 0.01, 20);
  private readonly renderer: THREE.WebGLRenderer | SVGRenderer;
  private readonly softwareFallback: boolean;
  private readonly controls: OrbitControls;
  private readonly transformControls: TransformControls;
  private readonly transformHelper: THREE.Object3D;
  private readonly gizmoProxy = new THREE.Object3D();
  private readonly raycaster = new THREE.Raycaster();
  private readonly pointer = new THREE.Vector2();
  private readonly geomObjects = new Map<number, THREE.Mesh>();
  private readonly meshGeometry = new Map<number, THREE.BufferGeometry>();
  private readonly toolDockQpos = new Map<ToolName, Float64Array>();
  private readonly toolFreeQpos = new Map<ToolName, Float64Array>();
  private readonly toolBodyId = new Map<ToolName, number>();
  private readonly relativeToolPoses = new Map<ToolName, RelativePose>();
  private readonly robotBodyId: number;
  private readonly fixtureBodyId: number;
  private readonly toolQposAddress = new Map<ToolName, number>();
  private readonly armQposAddresses: number[];
  private readonly armActuatorIds: number[];
  private readonly gripperQposAddress: number;
  private readonly gripperActuatorId: number;
  private readonly whiskQposAddress: number;
  private state: ControllerState;
  private resizeObserver: ResizeObserver;
  private onState?: (state: ControllerState) => void;
  private animationFrame = 0;
  private captureMode = false;
  private gizmoDragging = false;
  private cadRequest = 0;
  private pointerDownPosition: { x: number; y: number } | null = null;

  private constructor(
    mount: HTMLElement,
    module: any,
    model: any,
    data: any,
    vfs: any,
    manifest: Manifest,
  ) {
    this.module = module;
    this.model = model;
    this.data = data;
    this.vfs = vfs;
    this.manifest = manifest;
    this.jointRanges = Object.fromEntries(
      Object.entries(manifest.joints).map(([name, joint]) => [name, joint.range]),
    );

    const jointQposAddress = (name: string) => {
      const jointId = Number(model.jnt(name).id);
      return Number(model.jnt_qposadr[jointId]);
    };
    this.armQposAddresses = manifest.armJoints.map(jointQposAddress);
    this.armActuatorIds = manifest.armJoints.map(
      (name) => Number(model.actuator(name).id),
    );
    this.gripperQposAddress = jointQposAddress("gripper");
    this.gripperActuatorId = Number(model.actuator("gripper").id);
    this.whiskQposAddress = jointQposAddress("whisk_rotor_joint");
    this.robotBodyId = Number(model.body("robot_plate_frame").id);
    this.fixtureBodyId = Number(model.body("radial_three_tool_fixture").id);

    const initialJoints = Object.fromEntries(
      manifest.armJoints.map((name, index) => [name, manifest.presets.parked[index]]),
    );
    this.state = {
      joints: initialJoints,
      gripper: 0.15,
      whiskAngle: 0,
      selectedTool: null,
      attachedTool: null,
      gizmoMode: "translate",
      collisions: false,
      view: "scene",
      cadAssetId: manifest.cadAssets[0]?.id ?? null,
      geometryView: "cad",
      cadLoading: false,
      cadCollisionCount: 0,
    };

    let renderer: THREE.WebGLRenderer | SVGRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: false,
        preserveDrawingBuffer: true,
        powerPreference: "high-performance",
      });
      this.softwareFallback = false;
    } catch {
      renderer = new SVGRenderer();
      renderer.setQuality("high");
      this.softwareFallback = true;
    }
    this.renderer = renderer;
    mount.replaceChildren(renderer.domElement);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    if (this.renderer instanceof THREE.WebGLRenderer) {
      this.renderer.outputColorSpace = THREE.SRGBColorSpace;
      this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
      this.renderer.toneMappingExposure = 1.15;
      this.renderer.shadowMap.enabled = true;
      this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    }

    this.scene.background = new THREE.Color(0x071019);
    this.scene.fog = new THREE.FogExp2(0x071019, 0.72);
    this.root.rotation.x = -Math.PI / 2;
    this.scene.add(this.root);
    this.inspectorRoot.rotation.x = -Math.PI / 2;
    this.inspectorRoot.visible = false;
    this.inspectorRoot.add(this.inspectorCad, this.inspectorCollision);
    this.scene.add(this.inspectorRoot);

    const hemi = new THREE.HemisphereLight(0xc7e6ff, 0x11161d, 2.1);
    this.scene.add(hemi);
    const key = new THREE.DirectionalLight(0xfff0cf, 4.2);
    key.position.set(1.2, 1.7, 1.1);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.left = -1;
    key.shadow.camera.right = 1;
    key.shadow.camera.top = 1;
    key.shadow.camera.bottom = -1;
    this.scene.add(key);
    const rim = new THREE.DirectionalLight(0x74c7ff, 2.5);
    rim.position.set(-1.1, 0.8, -0.8);
    this.scene.add(rim);

    this.controls = new OrbitControls(this.camera, renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.075;
    this.controls.enableRotate = true;
    this.controls.enablePan = true;
    this.controls.screenSpacePanning = true;
    this.controls.rotateSpeed = 0.8;
    this.controls.minPolarAngle = 0.001;
    this.controls.maxPolarAngle = Math.PI - 0.001;
    this.controls.minDistance = 0.28;
    this.controls.maxDistance = 2.6;
    this.controls.mouseButtons.LEFT = THREE.MOUSE.ROTATE;
    this.controls.mouseButtons.MIDDLE = THREE.MOUSE.DOLLY;
    this.controls.mouseButtons.RIGHT = THREE.MOUSE.PAN;
    this.controls.touches.ONE = THREE.TOUCH.ROTATE;
    this.controls.touches.TWO = THREE.TOUCH.DOLLY_PAN;
    this.controls.addEventListener("change", () => this.render());
    this.resetCamera();

    this.gizmoProxy.name = "tool_flange_gizmo_target";
    this.root.add(this.gizmoProxy);
    this.transformControls = new TransformControls(this.camera, renderer.domElement);
    this.transformControls.setMode("translate");
    this.transformControls.setSpace("world");
    this.transformControls.setSize(0.72);
    this.transformControls.setColors(0xef5350, 0x66d17a, 0x42a5f5, 0xffc857);
    this.transformHelper = this.transformControls.getHelper();
    this.scene.add(this.transformHelper);
    this.transformControls.addEventListener("dragging-changed", (event) => {
      this.gizmoDragging = Boolean(event.value);
      this.controls.enabled = !this.gizmoDragging;
      if (!this.gizmoDragging) this.snapSelectedToDockIfClose();
    });
    this.transformControls.addEventListener("objectChange", () => this.applyGizmoPose());
    renderer.domElement.addEventListener("pointerdown", this.onPointerDown);
    renderer.domElement.addEventListener("pointerup", this.onPointerUp);
    renderer.domElement.addEventListener("pointercancel", this.onPointerCancel);

    for (const tool of ["gripper", "spoon", "whisk"] as ToolName[]) {
      const address = jointQposAddress(`tool_${tool}_free`);
      this.toolQposAddress.set(tool, address);
      this.toolBodyId.set(tool, Number(model.body(`tool_${tool}`).id));
      const dockPose = new Float64Array(
        Array.from(data.qpos.slice(address, address + 7), Number),
      );
      this.toolDockQpos.set(tool, dockPose);
      this.toolFreeQpos.set(tool, new Float64Array(dockPose));
    }

    this.captureRelativeToolPoses();
    this.buildVisibleGeometry();
    this.applyState(false);

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(mount);
    this.resize();
    this.animate();
  }

  static async create(mount: HTMLElement) {
    const [mujocoImport, manifestResponse, xmlResponse] = await Promise.all([
      import("@mujoco/mujoco"),
      fetch("/model/manifest.json"),
      fetch("/model/model.xml"),
    ]);
    if (!manifestResponse.ok || !xmlResponse.ok) {
      throw new Error("The QuickAttach model bundle could not be loaded.");
    }
    const mujoco = await mujocoImport.default();
    const manifest = (await manifestResponse.json()) as Manifest;
    const xml = await xmlResponse.text();
    const vfs = new mujoco.MjVFS();
    await Promise.all(
      manifest.assets.map(async (name) => {
        const response = await fetch(`/model/${name}`);
        if (!response.ok) throw new Error(`Missing model asset: ${name}`);
        vfs.addBuffer(name, new Uint8Array(await response.arrayBuffer()));
      }),
    );
    const model = mujoco.MjModel.from_xml_string(xml, vfs);
    const data = new mujoco.MjData(model);
    mujoco.mj_forward(model, data);
    return new QuickAttachRenderer(mount, mujoco, model, data, vfs, manifest);
  }

  subscribe(callback: (state: ControllerState) => void) {
    this.onState = callback;
    callback(cloneState(this.state));
  }

  getState() {
    return cloneState(this.state);
  }

  setJoint(name: string, value: number) {
    const range = this.jointRanges[name];
    if (!range || !this.manifest.armJoints.includes(name)) return this.getState();
    this.state.joints[name] = clamp(Number(value), range[0], range[1]);
    return this.applyState();
  }

  setState(next: Partial<ControllerState>) {
    if (next.joints) {
      for (const [name, value] of Object.entries(next.joints)) this.setJointValue(name, value);
    }
    if (typeof next.gripper === "number") {
      const range = this.jointRanges.gripper;
      this.state.gripper = clamp(next.gripper, range[0], range[1]);
    }
    if (typeof next.whiskAngle === "number") this.state.whiskAngle = next.whiskAngle;
    if (next.selectedTool !== undefined) this.state.selectedTool = next.selectedTool;
    if (next.attachedTool !== undefined) {
      this.state.attachedTool = next.attachedTool;
      if (next.attachedTool) this.state.selectedTool = next.attachedTool;
    }
    if (next.gizmoMode === "translate" || next.gizmoMode === "rotate") {
      this.state.gizmoMode = next.gizmoMode;
      this.transformControls.setMode(next.gizmoMode);
    }
    if (typeof next.collisions === "boolean" && next.collisions !== this.state.collisions) {
      this.state.collisions = next.collisions;
      if (next.collisions) this.buildCollisionGeometry();
      this.updateVisibility();
    }
    return this.applyState();
  }

  setTool(tool: ToolName | null) {
    if (!tool && this.state.attachedTool) this.captureAttachedAsFreePose();
    this.state.attachedTool = tool;
    this.state.selectedTool = tool;
    return this.applyState();
  }

  selectTool(tool: ToolName | null) {
    if (this.state.attachedTool && tool !== this.state.attachedTool) return this.getState();
    this.state.selectedTool = tool;
    this.syncGizmo();
    this.render();
    this.notify();
    return this.getState();
  }

  setGizmoMode(mode: "translate" | "rotate") {
    this.state.gizmoMode = mode;
    this.transformControls.setMode(mode);
    this.notify();
    return this.getState();
  }

  moveToDock(phase: "approach" | "seat") {
    const tool = this.state.attachedTool ?? this.state.selectedTool;
    if (!tool) return this.getState();
    const values = phase === "seat"
      ? this.manifest.dockCapture[tool]
      : this.manifest.presets[`${tool}Dock`];
    this.manifest.armJoints.forEach((joint, index) => {
      this.state.joints[joint] = values[index];
    });
    return this.applyState();
  }

  canAttach() {
    const error = this.getAttachmentError();
    return Boolean(
      error
      && !this.state.attachedTool
      && error.positionMm <= 65
      && error.angleDegrees <= 90,
    );
  }

  getAttachmentError() {
    const tool = this.state.selectedTool;
    if (!tool) return null;
    const flange = this.toolFlangePose(tool);
    const robot = this.robotFlangePose();
    const flangeQuaternion = new THREE.Quaternion().setFromRotationMatrix(
      matrix4FromRotation(flange.rotation),
    );
    const robotQuaternion = new THREE.Quaternion().setFromRotationMatrix(
      matrix4FromRotation(robot.rotation),
    );
    const angularError = 2.0 * Math.acos(clamp(Math.abs(flangeQuaternion.dot(robotQuaternion)), 0, 1));
    return {
      positionMm: flange.position.distanceTo(robot.position) * 1000,
      angleDegrees: THREE.MathUtils.radToDeg(angularError),
    };
  }

  canRelease() {
    return Boolean(this.state.attachedTool);
  }

  attachSelected() {
    if (!this.canAttach() || !this.state.selectedTool) return this.getState();
    this.state.attachedTool = this.state.selectedTool;
    return this.applyState();
  }

  releaseAttached() {
    if (!this.canRelease()) return this.getState();
    this.applyState(false);
    this.captureAttachedAsFreePose();
    this.state.attachedTool = null;
    return this.applyState();
  }

  returnSelectedToRack() {
    const tool = this.state.selectedTool;
    if (!tool || this.state.attachedTool) return this.getState();
    this.toolFreeQpos.set(tool, new Float64Array(this.toolDockQpos.get(tool)!));
    return this.applyState();
  }

  setPreset(name: string) {
    const preset = this.manifest.presets[name];
    if (!preset) return this.getState();
    this.manifest.armJoints.forEach((joint, index) => {
      this.state.joints[joint] = preset[index];
    });
    return this.applyState();
  }

  step(count = 1) {
    this.applyState(false);
    for (let index = 0; index < Math.max(1, Math.floor(count)); index += 1) {
      this.module.mj_step(this.model, this.data);
    }
    this.updateGeometryTransforms();
    this.render();
    this.notify();
    return this.getState();
  }

  reset() {
    this.state = {
      joints: Object.fromEntries(
        this.manifest.armJoints.map((name, index) => [
          name,
          this.manifest.presets.parked[index],
        ]),
      ),
      gripper: 0.15,
      whiskAngle: 0,
      selectedTool: null,
      attachedTool: null,
      gizmoMode: "translate",
      collisions: false,
      view: "scene",
      cadAssetId: this.manifest.cadAssets[0]?.id ?? null,
      geometryView: "cad",
      cadLoading: false,
      cadCollisionCount: 0,
    };
    for (const tool of ["gripper", "spoon", "whisk"] as ToolName[]) {
      this.toolFreeQpos.set(tool, new Float64Array(this.toolDockQpos.get(tool)!));
    }
    this.transformControls.setMode("translate");
    this.showScene();
    this.updateVisibility();
    this.resetCamera();
    return this.applyState();
  }

  resetCamera() {
    this.camera.position.set(0.82, 0.60, 0.42);
    this.controls.target.set(0.05, 0.22, 0.0);
    this.controls.update();
    this.render();
  }

  topCamera() {
    const distance = clamp(this.camera.position.distanceTo(this.controls.target), 0.72, 1.55);
    this.camera.position.set(
      this.controls.target.x + 0.001,
      this.controls.target.y + distance,
      this.controls.target.z + distance * 0.22,
    );
    this.controls.update();
    this.render();
  }

  renderFrame() {
    this.applyState(false);
    this.render();
    return this.getState();
  }

  showScene() {
    this.cadRequest += 1;
    this.state.view = "scene";
    this.state.cadLoading = false;
    this.root.visible = true;
    this.inspectorRoot.visible = false;
    this.transformHelper.visible = Boolean(this.state.selectedTool && !this.state.attachedTool);
    this.resetCamera();
    this.notify();
    return this.getState();
  }

  async inspectCad(assetId: string) {
    const asset = this.manifest.cadAssets.find((candidate) => candidate.id === assetId);
    if (!asset) return this.getState();
    const request = ++this.cadRequest;
    this.state.view = "cad";
    this.state.cadAssetId = asset.id;
    this.state.cadLoading = true;
    this.state.cadCollisionCount = 0;
    this.root.visible = false;
    this.inspectorRoot.visible = true;
    this.transformHelper.visible = false;
    this.clearGroup(this.inspectorCad);
    this.clearGroup(this.inspectorCollision);
    this.notify();
    this.render();

    const geometry = await this.stlLoader.loadAsync(`/model/${asset.asset}`);
    if (request !== this.cadRequest) {
      geometry.dispose();
      return this.getState();
    }
    geometry.scale(0.001, 0.001, 0.001);
    geometry.computeVertexNormals();
    const cadMesh = new THREE.Mesh(
      geometry,
      new THREE.MeshStandardMaterial({
        color: 0xd8dee5,
        roughness: 0.42,
        metalness: 0.08,
        side: THREE.DoubleSide,
      }),
    );
    cadMesh.castShadow = true;
    cadMesh.receiveShadow = true;
    this.inspectorCad.add(cadMesh);

    this.buildCollisionGeometry();
    this.updateGeometryTransforms();
    const collisions = this.matchingCollisionMeshes(asset);
    const bodyFrame = this.bodyWorldMatrix(asset.bodyRoots[0]);
    const inverseBodyFrame = bodyFrame?.clone().invert();
    for (const [, source] of collisions) {
      const clone = new THREE.Mesh(
        source.geometry,
        new THREE.MeshBasicMaterial({
          color: 0x35d4ff,
          transparent: true,
          opacity: 0.68,
          wireframe: true,
          depthWrite: false,
        }),
      );
      clone.matrixAutoUpdate = false;
      clone.matrix.copy(source.matrix);
      if (inverseBodyFrame) clone.matrix.premultiply(inverseBodyFrame);
      this.inspectorCollision.add(clone);
    }
    this.centerInspector();
    this.state.cadCollisionCount = collisions.length;
    this.state.cadLoading = false;
    this.updateInspectorVisibility();
    this.fitInspectorCamera();
    this.notify();
    this.render();
    return this.getState();
  }

  setGeometryView(view: GeometryView) {
    this.state.geometryView = view;
    this.updateInspectorVisibility();
    this.fitInspectorCamera();
    this.notify();
    this.render();
    return this.getState();
  }

  setCaptureMode(enabled: boolean) {
    if (enabled === this.captureMode) return;
    this.captureMode = enabled;
    if (enabled) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = 0;
      this.render();
    } else {
      this.animate();
    }
  }

  api(): QuickAttachApi {
    return {
      ready: true,
      getState: () => this.getState(),
      setJoint: (name, value) => this.setJoint(name, value),
      setPreset: (name) => this.setPreset(name),
      setState: (state) => this.setState(state),
      setTool: (tool) => this.setTool(tool),
      selectTool: (tool) => this.selectTool(tool),
      moveToDock: (phase) => this.moveToDock(phase),
      attachSelected: () => this.attachSelected(),
      releaseAttached: () => this.releaseAttached(),
      returnSelectedToRack: () => this.returnSelectedToRack(),
      setGizmoMode: (mode) => this.setGizmoMode(mode),
      getAttachmentError: () => this.getAttachmentError(),
      canAttach: () => this.canAttach(),
      canRelease: () => this.canRelease(),
      step: (count) => this.step(count),
      reset: () => this.reset(),
      resetCamera: () => this.resetCamera(),
      topCamera: () => this.topCamera(),
      renderFrame: () => this.renderFrame(),
      showScene: () => this.showScene(),
      inspectCad: (assetId) => this.inspectCad(assetId),
      setGeometryView: (view) => this.setGeometryView(view),
      setCaptureMode: (enabled) => this.setCaptureMode(enabled),
    };
  }

  dispose() {
    cancelAnimationFrame(this.animationFrame);
    this.resizeObserver.disconnect();
    this.renderer.domElement.removeEventListener("pointerdown", this.onPointerDown);
    this.renderer.domElement.removeEventListener("pointerup", this.onPointerUp);
    this.renderer.domElement.removeEventListener("pointercancel", this.onPointerCancel);
    this.transformControls.detach();
    this.transformControls.dispose();
    this.transformHelper.removeFromParent();
    this.controls.dispose();
    this.clearGroup(this.inspectorCad);
    this.clearGroup(this.inspectorCollision);
    if (this.renderer instanceof THREE.WebGLRenderer) this.renderer.dispose();
    for (const geometry of this.meshGeometry.values()) geometry.dispose();
    for (const mesh of this.geomObjects.values()) {
      mesh.geometry.dispose();
      (mesh.material as THREE.Material).dispose();
    }
    this.data.delete();
    this.model.delete();
    this.vfs.delete();
  }

  private setJointValue(name: string, value: number) {
    const range = this.jointRanges[name];
    if (range && this.manifest.armJoints.includes(name)) {
      this.state.joints[name] = clamp(Number(value), range[0], range[1]);
    }
  }

  private captureRelativeToolPoses() {
    const originalArm = this.armQposAddresses.map((address) => Number(this.data.qpos[address]));
    for (const tool of ["gripper", "spoon", "whisk"] as ToolName[]) {
      this.manifest.dockCapture[tool].forEach((value, index) => {
        this.data.qpos[this.armQposAddresses[index]] = value;
      });
      this.module.mj_kinematics(this.model, this.data);

      const robotPosition = new THREE.Vector3().fromArray(this.data.xpos, this.robotBodyId * 3);
      const robotRotation = matrix3FromRowMajor(this.data.xmat, this.robotBodyId * 9);
      const toolBodyId = Number(this.model.body(`tool_${tool}`).id);
      const toolPosition = new THREE.Vector3().fromArray(this.data.xpos, toolBodyId * 3);
      const toolRotation = matrix3FromRowMajor(this.data.xmat, toolBodyId * 9);
      const inverseRobot = robotRotation.clone().invert();
      const relativePosition = toolPosition.clone().sub(robotPosition).applyMatrix3(inverseRobot);
      const relativeRotation = inverseRobot.clone().multiply(toolRotation);
      this.relativeToolPoses.set(tool, { position: relativePosition, rotation: relativeRotation });
    }
    originalArm.forEach((value, index) => {
      this.data.qpos[this.armQposAddresses[index]] = value;
    });
    this.module.mj_kinematics(this.model, this.data);
  }

  private robotFlangePose(): RelativePose {
    return {
      position: new THREE.Vector3().fromArray(this.data.xpos, this.robotBodyId * 3),
      rotation: matrix3FromRowMajor(this.data.xmat, this.robotBodyId * 9),
    };
  }

  private toolFlangePose(tool: ToolName): RelativePose {
    const bodyId = this.toolBodyId.get(tool)!;
    const toolPosition = new THREE.Vector3().fromArray(this.data.xpos, bodyId * 3);
    const toolRotation = matrix3FromRowMajor(this.data.xmat, bodyId * 9);
    const relative = this.relativeToolPoses.get(tool)!;
    const flangeRotation = toolRotation.clone().multiply(relative.rotation.clone().invert());
    const flangePosition = toolPosition
      .clone()
      .sub(relative.position.clone().applyMatrix3(flangeRotation));
    return { position: flangePosition, rotation: flangeRotation };
  }

  private captureAttachedAsFreePose() {
    const tool = this.state.attachedTool;
    if (!tool) return;
    const address = this.toolQposAddress.get(tool)!;
    this.toolFreeQpos.set(
      tool,
      new Float64Array(Array.from(this.data.qpos.slice(address, address + 7), Number)),
    );
  }

  private syncGizmo() {
    if (this.gizmoDragging) return;
    const tool = this.state.selectedTool;
    if (!tool || this.state.attachedTool) {
      this.transformControls.detach();
      return;
    }
    const flange = this.toolFlangePose(tool);
    this.gizmoProxy.position.copy(flange.position);
    this.gizmoProxy.quaternion.setFromRotationMatrix(matrix4FromRotation(flange.rotation));
    this.gizmoProxy.updateMatrixWorld(true);
    this.transformControls.attach(this.gizmoProxy);
  }

  private applyGizmoPose() {
    const tool = this.state.selectedTool;
    if (!tool || this.state.attachedTool) return;
    const relative = this.relativeToolPoses.get(tool)!;
    const flangeRotation = new THREE.Matrix3().setFromMatrix4(
      new THREE.Matrix4().makeRotationFromQuaternion(this.gizmoProxy.quaternion),
    );
    const toolRotation = flangeRotation.clone().multiply(relative.rotation);
    const toolPosition = relative.position
      .clone()
      .applyMatrix3(flangeRotation)
      .add(this.gizmoProxy.position);
    const toolQuaternion = new THREE.Quaternion().setFromRotationMatrix(
      matrix4FromRotation(toolRotation),
    );
    const pose = this.toolFreeQpos.get(tool)!;
    pose[0] = toolPosition.x;
    pose[1] = toolPosition.y;
    pose[2] = toolPosition.z;
    pose[3] = toolQuaternion.w;
    pose[4] = toolQuaternion.x;
    pose[5] = toolQuaternion.y;
    pose[6] = toolQuaternion.z;
    const address = this.toolQposAddress.get(tool)!;
    for (let index = 0; index < 7; index += 1) this.data.qpos[address + index] = pose[index];
    this.module.mj_kinematics(this.model, this.data);
    this.updateGeometryTransforms();
    this.render();
    this.notify();
  }

  private snapSelectedToDockIfClose() {
    const tool = this.state.selectedTool;
    if (!tool || this.state.attachedTool) return;
    const pose = this.toolFreeQpos.get(tool)!;
    const dock = this.toolDockQpos.get(tool)!;
    const positionError = Math.hypot(
      pose[0] - dock[0],
      pose[1] - dock[1],
      pose[2] - dock[2],
    );
    const poseQuaternion = new THREE.Quaternion(pose[4], pose[5], pose[6], pose[3]);
    const dockQuaternion = new THREE.Quaternion(dock[4], dock[5], dock[6], dock[3]);
    const angularError = 2.0 * Math.acos(
      clamp(Math.abs(poseQuaternion.dot(dockQuaternion)), 0, 1),
    );
    if (positionError <= 0.030 && angularError <= Math.PI / 4.0) {
      this.toolFreeQpos.set(tool, new Float64Array(dock));
      this.applyState();
    }
  }

  private toolForBody(bodyId: number): ToolName | null {
    let current = bodyId;
    while (current > 0) {
      for (const [tool, toolBody] of this.toolBodyId) {
        if (current === toolBody) return tool;
      }
      current = Number(this.model.body_parentid[current]);
    }
    return null;
  }

  private onPointerDown = (event: Event) => {
    if (!(event instanceof PointerEvent)) return;
    this.pointerDownPosition = { x: event.clientX, y: event.clientY };
  };

  private onPointerUp = (event: Event) => {
    if (!(event instanceof PointerEvent)) return;
    const start = this.pointerDownPosition;
    this.pointerDownPosition = null;
    if (!start || this.gizmoDragging || Math.hypot(event.clientX - start.x, event.clientY - start.y) > 6) {
      return;
    }
    const bounds = this.renderer.domElement.getBoundingClientRect();
    this.pointer.set(
      ((event.clientX - bounds.left) / bounds.width) * 2 - 1,
      -((event.clientY - bounds.top) / bounds.height) * 2 + 1,
    );
    this.raycaster.setFromCamera(this.pointer, this.camera);
    this.root.updateMatrixWorld(true);
    const candidates = Array.from(this.geomObjects.values()).filter(
      (mesh) => mesh.visible && Boolean(mesh.userData.tool),
    );
    const hit = this.raycaster.intersectObjects(candidates, false)[0];
    const tool = hit?.object.userData.tool as ToolName | undefined;
    if (tool) this.selectTool(tool);
  };

  private onPointerCancel = () => {
    this.pointerDownPosition = null;
  };

  private attachSelectedTool() {
    for (const tool of ["gripper", "spoon", "whisk"] as ToolName[]) {
      const address = this.toolQposAddress.get(tool)!;
      const pose = this.toolFreeQpos.get(tool)!;
      for (let index = 0; index < 7; index += 1) this.data.qpos[address + index] = pose[index];
    }
    this.module.mj_kinematics(this.model, this.data);
    const tool = this.state.attachedTool;
    if (!tool) return;

    const robotPosition = new THREE.Vector3().fromArray(this.data.xpos, this.robotBodyId * 3);
    const robotRotation = matrix3FromRowMajor(this.data.xmat, this.robotBodyId * 9);
    const relative = this.relativeToolPoses.get(tool)!;
    const worldPosition = relative.position.clone().applyMatrix3(robotRotation).add(robotPosition);
    const worldRotation = robotRotation.clone().multiply(relative.rotation);
    const quaternion = new THREE.Quaternion().setFromRotationMatrix(matrix4FromRotation(worldRotation));
    const address = this.toolQposAddress.get(tool)!;
    this.data.qpos[address] = worldPosition.x;
    this.data.qpos[address + 1] = worldPosition.y;
    this.data.qpos[address + 2] = worldPosition.z;
    this.data.qpos[address + 3] = quaternion.w;
    this.data.qpos[address + 4] = quaternion.x;
    this.data.qpos[address + 5] = quaternion.y;
    this.data.qpos[address + 6] = quaternion.z;
    this.module.mj_kinematics(this.model, this.data);
  }

  private applyState(notify = true) {
    this.manifest.armJoints.forEach((name, index) => {
      const value = this.state.joints[name];
      this.data.qpos[this.armQposAddresses[index]] = value;
      this.data.ctrl[this.armActuatorIds[index]] = value;
    });
    this.data.qpos[this.gripperQposAddress] = this.state.gripper;
    this.data.ctrl[this.gripperActuatorId] = this.state.gripper;
    this.data.qpos[this.whiskQposAddress] = this.state.whiskAngle;
    this.module.mj_kinematics(this.model, this.data);
    this.attachSelectedTool();
    this.updateGeometryTransforms();
    this.syncGizmo();
    this.render();
    if (notify) this.notify();
    return this.getState();
  }

  private notify() {
    this.onState?.(this.getState());
  }

  private buildVisibleGeometry() {
    for (let geomId = 0; geomId < this.model.ngeom; geomId += 1) {
      const group = Number(this.model.geom_group[geomId]);
      if (geomId === 0) {
        this.buildGeom(geomId);
        continue;
      }
      const fixtureGeom = Number(this.model.geom_bodyid[geomId]) === this.fixtureBodyId;
      if (this.softwareFallback) {
        if (group === 4 || (group === 2 && !fixtureGeom)) this.buildGeom(geomId);
      } else if (group === 2) {
        this.buildGeom(geomId);
      }
    }
  }

  private buildCollisionGeometry() {
    for (let geomId = 0; geomId < this.model.ngeom; geomId += 1) {
      if (Number(this.model.geom_group[geomId]) === 3) this.buildGeom(geomId);
    }
  }

  private buildGeom(geomId: number) {
    if (this.geomObjects.has(geomId)) return;
    const type = Number(this.model.geom_type[geomId]);
    const sizeOffset = geomId * 3;
    const sx = Number(this.model.geom_size[sizeOffset]);
    const sy = Number(this.model.geom_size[sizeOffset + 1]);
    const sz = Number(this.model.geom_size[sizeOffset + 2]);
    let geometry: THREE.BufferGeometry;

    if (type === MJ_GEOM.MESH) {
      const meshId = Number(this.model.geom_dataid[geomId]);
      geometry = this.meshGeometry.get(meshId) ?? this.makeMeshGeometry(meshId);
      this.meshGeometry.set(meshId, geometry);
    } else if (type === MJ_GEOM.BOX) {
      geometry = new THREE.BoxGeometry(sx * 2, sy * 2, sz * 2);
    } else if (type === MJ_GEOM.CYLINDER) {
      geometry = new THREE.CylinderGeometry(sx, sx, sy * 2, 20);
      geometry.rotateX(Math.PI / 2);
    } else if (type === MJ_GEOM.CAPSULE) {
      geometry = new THREE.CapsuleGeometry(sx, sy * 2, 8, 16);
      geometry.rotateX(Math.PI / 2);
    } else if (type === MJ_GEOM.SPHERE) {
      geometry = new THREE.SphereGeometry(sx, 20, 14);
    } else if (type === MJ_GEOM.ELLIPSOID) {
      geometry = new THREE.SphereGeometry(1, 20, 14);
      geometry.scale(sx, sy, sz);
    } else if (type === MJ_GEOM.PLANE) {
      geometry = new THREE.PlaneGeometry(Math.max(sx * 2, 2.2), Math.max(sy * 2, 2.2));
    } else {
      return;
    }

    const group = Number(this.model.geom_group[geomId]);
    const collision = group === 3 && geomId !== 0;
    const rgba = this.geomColor(geomId);
    const material = collision
      ? new THREE.MeshBasicMaterial({
          color: new THREE.Color(0x43d5ff),
          transparent: true,
          opacity: 0.16,
          wireframe: true,
          depthWrite: false,
        })
      : this.softwareFallback
        ? new THREE.MeshBasicMaterial({
            color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
            transparent: rgba[3] < 0.995,
            opacity: rgba[3],
            side: THREE.DoubleSide,
          })
        : new THREE.MeshStandardMaterial({
          color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
          roughness: geomId === 0 ? 0.92 : 0.46,
          metalness: geomId === 0 ? 0.03 : 0.16,
          transparent: rgba[3] < 0.995,
          opacity: rgba[3],
          side: THREE.DoubleSide,
          });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.matrixAutoUpdate = false;
    mesh.castShadow = !collision && geomId !== 0;
    mesh.receiveShadow = geomId === 0 || !collision;
    mesh.userData.collision = collision;
    const tool = this.toolForBody(Number(this.model.geom_bodyid[geomId]));
    if (tool && !collision) mesh.userData.tool = tool;
    this.geomObjects.set(geomId, mesh);
    this.root.add(mesh);
  }

  private makeMeshGeometry(meshId: number) {
    const vertexAddress = Number(this.model.mesh_vertadr[meshId]);
    const vertexCount = Number(this.model.mesh_vertnum[meshId]);
    const faceAddress = Number(this.model.mesh_faceadr[meshId]);
    const faceCount = Number(this.model.mesh_facenum[meshId]);
    const positions = new Float32Array(vertexCount * 3);
    for (let index = 0; index < positions.length; index += 1) {
      positions[index] = Number(this.model.mesh_vert[vertexAddress * 3 + index]);
    }
    if (this.softwareFallback) {
      const points: THREE.Vector3[] = [];
      const sampleCount = Math.min(vertexCount, 360);
      const sampleStride = Math.max(1, Math.floor(vertexCount / sampleCount));
      for (let vertex = 0; vertex < vertexCount; vertex += sampleStride) {
        const offset = vertex * 3;
        points.push(new THREE.Vector3(
          positions[offset],
          positions[offset + 1],
          positions[offset + 2],
        ));
      }
      if (points.length >= 4) {
        try {
          return new ConvexGeometry(points);
        } catch {
          // Degenerate planar meshes fall through to a compact bounding proxy.
        }
      }
      const bounds = new THREE.Box3().setFromBufferAttribute(
        new THREE.BufferAttribute(positions, 3),
      );
      const size = bounds.getSize(new THREE.Vector3());
      const center = bounds.getCenter(new THREE.Vector3());
      const box = new THREE.BoxGeometry(
        Math.max(size.x, 0.0005),
        Math.max(size.y, 0.0005),
        Math.max(size.z, 0.0005),
      );
      box.translate(center.x, center.y, center.z);
      return box;
    }
    const indices = new Uint32Array(faceCount * 3);
    for (let index = 0; index < indices.length; index += 1) {
      indices[index] = Number(this.model.mesh_face[faceAddress * 3 + index]);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    geometry.computeVertexNormals();
    return geometry;
  }

  private geomColor(geomId: number): [number, number, number, number] {
    const materialId = Number(this.model.geom_matid[geomId]);
    const source = materialId >= 0 ? this.model.mat_rgba : this.model.geom_rgba;
    const offset = (materialId >= 0 ? materialId : geomId) * 4;
    if (geomId === 0) return [0.055, 0.075, 0.095, 1];
    return [
      Number(source[offset]),
      Number(source[offset + 1]),
      Number(source[offset + 2]),
      Number(source[offset + 3]),
    ];
  }

  private updateGeometryTransforms() {
    for (const [geomId, mesh] of this.geomObjects) {
      const p = geomId * 3;
      const r = geomId * 9;
      mesh.matrix.set(
        this.data.geom_xmat[r], this.data.geom_xmat[r + 1], this.data.geom_xmat[r + 2], this.data.geom_xpos[p],
        this.data.geom_xmat[r + 3], this.data.geom_xmat[r + 4], this.data.geom_xmat[r + 5], this.data.geom_xpos[p + 1],
        this.data.geom_xmat[r + 6], this.data.geom_xmat[r + 7], this.data.geom_xmat[r + 8], this.data.geom_xpos[p + 2],
        0, 0, 0, 1,
      );
      mesh.matrixWorldNeedsUpdate = true;
    }
  }

  private updateVisibility() {
    for (const mesh of this.geomObjects.values()) {
      if (mesh.userData.collision) mesh.visible = this.state.collisions;
    }
  }

  private matchingCollisionMeshes(asset: CadAsset) {
    const result: Array<[number, THREE.Mesh]> = [];
    for (const [geomId, mesh] of this.geomObjects) {
      if (!mesh.userData.collision) continue;
      const name = String(this.model.geom(geomId).name ?? "");
      if (asset.geomPrefixes.length && !asset.geomPrefixes.some((prefix) => name.startsWith(prefix))) {
        continue;
      }
      let bodyId = Number(this.model.geom_bodyid[geomId]);
      let matchesBody = false;
      while (bodyId > 0) {
        const bodyName = String(this.model.body(bodyId).name ?? "");
        if (asset.bodyRoots.includes(bodyName)) {
          matchesBody = true;
          break;
        }
        bodyId = Number(this.model.body_parentid[bodyId]);
      }
      if (matchesBody) result.push([geomId, mesh]);
    }
    return result;
  }

  private bodyWorldMatrix(bodyName: string | undefined) {
    if (!bodyName) return null;
    let bodyId: number;
    try {
      bodyId = Number(this.model.body(bodyName).id);
    } catch {
      return null;
    }
    const p = bodyId * 3;
    const r = bodyId * 9;
    return new THREE.Matrix4().set(
      this.data.xmat[r], this.data.xmat[r + 1], this.data.xmat[r + 2], this.data.xpos[p],
      this.data.xmat[r + 3], this.data.xmat[r + 4], this.data.xmat[r + 5], this.data.xpos[p + 1],
      this.data.xmat[r + 6], this.data.xmat[r + 7], this.data.xmat[r + 8], this.data.xpos[p + 2],
      0, 0, 0, 1,
    );
  }

  private clearGroup(group: THREE.Group) {
    for (const child of [...group.children]) {
      group.remove(child);
      child.traverse((object) => {
        if (!(object instanceof THREE.Mesh)) return;
        if (group === this.inspectorCad) object.geometry.dispose();
        const materials = Array.isArray(object.material) ? object.material : [object.material];
        for (const material of materials) material.dispose();
      });
    }
    group.position.set(0, 0, 0);
  }

  private centerInspector() {
    this.inspectorCad.position.set(0, 0, 0);
    this.inspectorCollision.position.set(0, 0, 0);
    this.inspectorRoot.updateMatrixWorld(true);
    const bounds = new THREE.Box3().setFromObject(this.inspectorCad);
    if (bounds.isEmpty()) bounds.setFromObject(this.inspectorCollision);
    if (bounds.isEmpty()) return;
    const center = bounds.getCenter(new THREE.Vector3());
    this.inspectorCad.position.sub(center);
    this.inspectorCollision.position.sub(center);
    this.inspectorRoot.updateMatrixWorld(true);
  }

  private updateInspectorVisibility() {
    this.inspectorCad.visible = this.state.geometryView !== "collision";
    this.inspectorCollision.visible = this.state.geometryView !== "cad";
  }

  private fitInspectorCamera() {
    const bounds = new THREE.Box3();
    if (this.inspectorCad.visible) bounds.expandByObject(this.inspectorCad);
    if (this.inspectorCollision.visible) bounds.expandByObject(this.inspectorCollision);
    if (bounds.isEmpty()) return;
    const sphere = bounds.getBoundingSphere(new THREE.Sphere());
    const distance = Math.max(0.12, sphere.radius / Math.sin(THREE.MathUtils.degToRad(this.camera.fov * 0.42)));
    this.controls.target.copy(sphere.center);
    this.camera.position.copy(sphere.center).add(new THREE.Vector3(0.82, 0.62, 0.72).normalize().multiplyScalar(distance));
    this.camera.near = Math.max(0.001, distance / 100);
    this.camera.far = Math.max(5, distance * 8);
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  private resize() {
    const surface = this.renderer.domElement;
    const parent = surface.parentElement;
    const width = Math.max(1, parent?.clientWidth ?? surface.clientWidth);
    const height = Math.max(1, parent?.clientHeight ?? surface.clientHeight);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.render();
  }

  private render() {
    this.renderer.render(this.scene, this.camera);
  }

  private animate = () => {
    this.controls.update();
    this.render();
    if (!this.captureMode) this.animationFrame = requestAnimationFrame(this.animate);
  };
}
