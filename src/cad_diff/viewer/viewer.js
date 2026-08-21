import * as THREE from "three";
import { GLTFLoader } from "three-gltf-loader";
import { OrbitControls } from "three-orbit-controls";

const STATUS_COLOR = { unchanged: "#9e9e9e", modified: "#e6b81a", added: "#45ad59", removed: "#d13d33" };

function base64ToArrayBuffer(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}

function buildLegend(statusesPresent) {
  const legend = document.getElementById("legend");
  for (const status of statusesPresent) {
    const row = document.createElement("label");
    row.className = "legend-row";
    row.dataset.status = status;
    row.dataset.visible = "true";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = true;
    checkbox.dataset.status = status;

    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = STATUS_COLOR[status];

    row.appendChild(checkbox);
    row.appendChild(swatch);
    row.appendChild(document.createTextNode(status));
    legend.appendChild(row);
  }
}

function init(glbBase64) {
  const container = document.getElementById("viewer");
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1b1d1f);

  const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 100000);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.appendChild(renderer.domElement);

  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const key = new THREE.DirectionalLight(0xffffff, 1.1);
  key.position.set(1, 1.4, 1);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xffffff, 0.4);
  fill.position.set(-1, -0.5, -1);
  scene.add(fill);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  const meshesByStatus = { unchanged: [], modified: [], added: [], removed: [] };

  new GLTFLoader().parse(base64ToArrayBuffer(glbBase64), "", (gltf) => {
    scene.add(gltf.scene);

    gltf.scene.traverse((child) => {
      if (!child.isMesh) return;
      child.geometry.computeVertexNormals();
      const status = child.material.name;
      if (meshesByStatus[status]) meshesByStatus[status].push(child);
    });

    const box = new THREE.Box3().setFromObject(gltf.scene);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const radius = Math.max(size.x, size.y, size.z) || 1;

    camera.position.copy(center).add(new THREE.Vector3(radius, radius * 0.8, radius));
    camera.near = radius / 100;
    camera.far = radius * 100;
    camera.updateProjectionMatrix();
    controls.target.copy(center);
    controls.update();

    buildLegend(Object.keys(meshesByStatus).filter((status) => meshesByStatus[status].length > 0));
    document.querySelectorAll("#legend input[type=checkbox]").forEach((checkbox) => {
      checkbox.addEventListener("change", () => {
        for (const mesh of meshesByStatus[checkbox.dataset.status]) mesh.visible = checkbox.checked;
        checkbox.closest(".legend-row").dataset.visible = String(checkbox.checked);
      });
    });
    document.getElementById("loading").style.display = "none";
  }, (error) => {
    document.getElementById("loading").textContent = "Failed to load model: " + error.message;
  });

  window.addEventListener("resize", () => {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  });

  (function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  })();
}

init(window.__CAD_DIFF_GLB_BASE64__);
