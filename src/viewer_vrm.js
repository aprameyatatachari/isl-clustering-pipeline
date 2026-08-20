// VRM avatar stage: loads the user's .vrm and retargets pose keypoints onto its
// humanoid bones. Loaded only when the viewer was built with --vrm.
(() => {
  const FILE = window.VRM_FILE;
  if (!FILE) return;

  const host = document.getElementById("vrmstage");
  const status = document.getElementById("vrmstatus");
  let vrm = null, renderer = null, camera = null, scene = null, controls = null;

  // Keypoint indices, mirroring src/features.py
  const K = {LSHO:0, RSHO:1, LELB:2, RELB:3, LWRI:4, RWRI:5};
  const LH = 15, RH = 36;

  // Convert MediaPipe axes to the VRM world frame of a model facing +Z:
  // +x is already the signer's left, image y grows downward, and MediaPipe
  // depth grows more negative toward the camera. So y and z both flip.
  const dir = (pose, a, b) => {
    const v = new THREE.Vector3(pose[b][0]-pose[a][0],
                                -(pose[b][1]-pose[a][1]),
                                -(pose[b][2]-pose[a][2]));
    return v.lengthSq() < 1e-9 ? null : v.normalize();
  };

  // Which child each bone points at. A bone's direction is defined by where its
  // child sits, so the child offset is all the retargeting needs.
  const CHAIN = {
    leftUpperArm: "leftLowerArm", leftLowerArm: "leftHand", leftHand: "leftMiddleProximal",
    rightUpperArm: "rightLowerArm", rightLowerArm: "rightHand", rightHand: "rightMiddleProximal",
  };

  function boneNode(name) {
    if (!vrm || !vrm.humanoid) return null;
    return vrm.humanoid.getNormalizedBoneNode
      ? vrm.humanoid.getNormalizedBoneNode(name)
      : vrm.humanoid.getBoneNode(name);
  }

  // Aim a bone along `target`, a world-space direction.
  //
  // The child's position is a constant offset in the bone's own local frame, so
  // the bone's local rotation q satisfies: q * childOffset = target expressed in
  // the parent's frame. Solving for q directly makes this work on any rig --
  // T-pose or A-pose, VRM 0.x or 1.x, however the root is oriented -- with no
  // assumption about rest orientation.
  const _t = new THREE.Vector3(), _rest = new THREE.Vector3(), _pq = new THREE.Quaternion();
  function aim(name, target) {
    const node = boneNode(name), child = boneNode(CHAIN[name]);
    if (!node || !child || !target) return;
    if (child.parent !== node) return;      // not a direct child on this rig
    _rest.copy(child.position);
    if (_rest.lengthSq() < 1e-12) return;
    _rest.normalize();

    node.parent.updateWorldMatrix(true, false);
    node.parent.getWorldQuaternion(_pq);
    _t.copy(target).applyQuaternion(_pq.invert());   // target in the parent's frame
    node.quaternion.setFromUnitVectors(_rest, _t);
    node.updateWorldMatrix(false, false);
  }

  function applyPose(pose) {
    if (!vrm) return;
    if (vrm.humanoid.resetNormalizedPose) vrm.humanoid.resetNormalizedPose();
    aim("leftUpperArm",  dir(pose, K.LSHO, K.LELB));
    aim("leftLowerArm",  dir(pose, K.LELB, K.LWRI));
    aim("rightUpperArm", dir(pose, K.RSHO, K.RELB));
    aim("rightLowerArm", dir(pose, K.RELB, K.RWRI));
    // Wrists follow the palm direction, but only when the hand was detected --
    // an undetected hand collapses onto the wrist and carries no direction.
    for (const [bone, base, wri] of [["leftHand", LH, K.LWRI], ["rightHand", RH, K.RWRI]]) {
      const knuckle = pose[base + 9], root = pose[base];
      if (Math.hypot(knuckle[0]-root[0], knuckle[1]-root[1]) < 0.03) continue;
      aim(bone, dir(pose, wri, base + 9));
    }
    if (vrm.humanoid.update) vrm.humanoid.update();
  }

  function frameCamera() {
    const box = new THREE.Box3().setFromObject(vrm.scene);
    const size = box.getSize(new THREE.Vector3());
    const mid = box.getCenter(new THREE.Vector3());
    // Look at the upper body, where signing happens.
    const target = new THREE.Vector3(mid.x, box.max.y - size.y * 0.28, mid.z);
    camera.position.set(target.x, target.y, target.z + Math.max(size.y * 0.85, 0.9));
    controls.target.copy(target);
    controls.update();
  }

  function init() {
    const w = host.clientWidth || 320, h = Math.max(host.clientHeight, 340);
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(30, w/h, 0.05, 40);
    renderer = new THREE.WebGLRenderer({antialias: true, alpha: true});
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.setSize(w, h);
    host.appendChild(renderer.domElement);

    scene.add(new THREE.AmbientLight(0xffffff, 0.75));
    const key = new THREE.DirectionalLight(0xffffff, 0.85);
    key.position.set(1, 2, 2);
    scene.add(key);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    const loader = new THREE.GLTFLoader();
    loader.register(parser => new THREE_VRM.VRMLoaderPlugin(parser));
    status.textContent = "loading avatar…";
    loader.load(FILE, gltf => {
      vrm = gltf.userData.vrm;
      if (!vrm) { status.textContent = "not a VRM file"; return; }
      if (THREE_VRM.VRMUtils) {
        THREE_VRM.VRMUtils.removeUnnecessaryVertices(gltf.scene);
        THREE_VRM.VRMUtils.removeUnnecessaryJoints(gltf.scene);
      }
      // VRM 0.x models face -Z; turn them around so they face the camera.
      if (vrm.meta && vrm.meta.metaVersion === "0") vrm.scene.rotation.y = Math.PI;
      scene.add(vrm.scene);
      frameCamera();
      status.textContent = "";
      window.VRM_READY = true;
      window.VRM_INSTANCE = vrm;   // exposed for debugging and rig inspection
      if (window.lastPose) applyPose(window.lastPose);
    }, undefined, err => {
      status.textContent = "failed to load: " + (err && err.message ? err.message : err);
    });

    const tick = () => {
      requestAnimationFrame(tick);
      controls.update();
      if (vrm && vrm.update) vrm.update(1/60);
      renderer.render(scene, camera);
    };
    tick();

    addEventListener("resize", () => {
      const w2 = host.clientWidth || 320, h2 = Math.max(host.clientHeight, 340);
      camera.aspect = w2/h2;
      camera.updateProjectionMatrix();
      renderer.setSize(w2, h2);
    });
  }

  window.VRM_APPLY = applyPose;
  init();
})();
