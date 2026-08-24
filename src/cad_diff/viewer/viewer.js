import * as THREE from "three";
import { GLTFLoader } from "three-gltf-loader";
import { OrbitControls } from "three-orbit-controls";

const STATUS_COLOR = { unchanged: "#9e9e9e", modified: "#e6b81a", added: "#45ad59", removed: "#d13d33", model: "#46b8a9" };

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
  controls.enablePan = true;

  const meshesByStatus = { unchanged: [], modified: [], added: [], removed: [], model: [] };
  const pickableMeshes = [];
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  const selectedPoints = [];
  const markers = [];
  let modelRadius = 1;
  let grid = null;
  let axes = null;

  function materialsFor(mesh) {
    return Array.isArray(mesh.material) ? mesh.material : [mesh.material];
  }

  function updateSurfaceMode() {
    const useNormals = document.getElementById("viewer-normals").checked;
    const useWireframe = document.getElementById("viewer-wireframe").checked;
    const useTexture = document.getElementById("viewer-texture").checked;
    for (const mesh of pickableMeshes) {
      if (useNormals) {
        if (!mesh.userData.normalMaterial) {
          mesh.userData.normalMaterial = new THREE.MeshNormalMaterial({ side: THREE.DoubleSide });
        }
        mesh.material = mesh.userData.normalMaterial;
      } else if (useTexture) {
        mesh.material = mesh.userData.originalMaterial;
      } else {
        if (!mesh.userData.neutralMaterial) {
          mesh.userData.neutralMaterial = new THREE.MeshStandardMaterial({
            color: 0x8c969d,
            roughness: 0.82,
            metalness: 0.04,
            side: THREE.DoubleSide,
          });
        }
        mesh.material = mesh.userData.neutralMaterial;
      }
      for (const material of materialsFor(mesh)) material.wireframe = useWireframe;
    }
  }

  function updatePickedPointText() {
    const output = document.getElementById("picked-points");
    if (!selectedPoints.length) {
      output.textContent = "No calibration point selected.";
      return;
    }
    output.textContent = selectedPoints.map((point, index) => {
      const values = point.toArray().map((value) => value.toPrecision(6)).join(", ");
      return `${index === 0 ? "A" : "B"}: ${values}`;
    }).join("\n");
  }

  function selectPoint(point) {
    if (selectedPoints.length === 2) {
      selectedPoints.length = 0;
      for (const marker of markers.splice(0)) {
        scene.remove(marker);
        marker.geometry.dispose();
        marker.material.dispose();
      }
    }
    const selected = point.clone();
    selectedPoints.push(selected);
    const marker = new THREE.Mesh(
      new THREE.SphereGeometry(Math.max(modelRadius * 0.014, 0.0001), 20, 12),
      new THREE.MeshBasicMaterial({
        color: selectedPoints.length === 1 ? 0x63e6be : 0xffd166,
        depthTest: false,
      }),
    );
    marker.position.copy(selected);
    marker.renderOrder = 100;
    scene.add(marker);
    markers.push(marker);
    updatePickedPointText();
    if (window.parent !== window) {
      window.parent.postMessage(
        { type: "cadpro-point-picked", point: selected.toArray() },
        window.location.origin,
      );
    }
  }

  new GLTFLoader().parse(base64ToArrayBuffer(glbBase64), "", (gltf) => {
    scene.add(gltf.scene);

    gltf.scene.traverse((child) => {
      if (!child.isMesh) return;
      child.geometry.computeVertexNormals();
      child.userData.originalMaterial = child.material;
      pickableMeshes.push(child);
      const namedMaterial = Array.isArray(child.material) ? child.material[0] : child.material;
      const candidate = String(namedMaterial?.name || "").toLowerCase();
      const status = meshesByStatus[candidate] ? candidate : "model";
      meshesByStatus[status].push(child);
    });

    const box = new THREE.Box3().setFromObject(gltf.scene);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const radius = Math.max(size.x, size.y, size.z) || 1;
    modelRadius = radius;

    camera.position.copy(center).add(new THREE.Vector3(radius, radius * 0.8, radius));
    camera.near = radius / 100;
    camera.far = radius * 100;
    camera.updateProjectionMatrix();
    controls.target.copy(center);
    controls.update();

    grid = new THREE.GridHelper(radius * 3, 20, 0x51616d, 0x303940);
    grid.position.set(center.x, box.min.y, center.z);
    scene.add(grid);
    axes = new THREE.AxesHelper(radius * 0.7);
    axes.position.copy(center);
    scene.add(axes);
    document.getElementById("model-bounds").textContent =
      `Bounds: ${size.x.toPrecision(6)} x ${size.y.toPrecision(6)} x ${size.z.toPrecision(6)} model units`;

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

  document.getElementById("viewer-grid").addEventListener("change", (event) => {
    if (grid) grid.visible = event.target.checked;
  });
  document.getElementById("viewer-axes").addEventListener("change", (event) => {
    if (axes) axes.visible = event.target.checked;
  });
  document.getElementById("viewer-wireframe").addEventListener("change", updateSurfaceMode);
  document.getElementById("viewer-normals").addEventListener("change", updateSurfaceMode);
  document.getElementById("viewer-texture").addEventListener("change", updateSurfaceMode);

  renderer.domElement.addEventListener("dblclick", (event) => {
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObjects(
      pickableMeshes.filter((mesh) => mesh.visible),
      false,
    )[0];
    if (hit) selectPoint(hit.point);
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
