const $ = (id) => document.getElementById(id);
const MAX_BYTES = 10 * 1024 * 1024;   // 10 MiB, khớp hợp đồng API

let image = null;   // trạng thái ảnh do server trả về (version luôn lấy từ server)
let busy = false;   // đang gọi API: bỏ qua thao tác mới, kể cả thả ảnh
const objectUrls = { orig: null, result: null };
let originalLoadedFor = null;

// ---------- Lỗi và gọi API ----------

class ApiError extends Error {
  constructor(message, { status = 0, code = "UNKNOWN", data = null } = {}) {
    super(message);
    this.status = status;
    this.code = code;   // data.code của server, hoặc NETWORK_ERROR / BAD_RESPONSE
    this.data = data;
  }
}

// Trả về { data, message, res }. Mọi lỗi (JSON hoặc không phải JSON, mất mạng) đều ném ApiError.
// Frontend dựa vào HTTP status và data.code, không so sánh câu thông báo.
async function request(url, options = {}) {
  let res;
  try {
    res = await fetch(url, { credentials: "same-origin", ...options });
  } catch {
    throw new ApiError("Không kết nối được máy chủ.", { code: "NETWORK_ERROR" });
  }
  const type = res.headers.get("Content-Type") || "";
  if (type.includes("application/json")) {
    let body;
    try {
      body = await res.json();
    } catch {
      throw new ApiError("Phản hồi từ máy chủ không hợp lệ.", { status: res.status, code: "BAD_RESPONSE" });
    }
    if (!res.ok || !body.success) {
      const data = body.data || {};
      throw new ApiError(body.message || "Có lỗi xảy ra.", { status: res.status, code: data.code, data });
    }
    return { data: body.data, message: body.message, res };
  }
  if (!res.ok) {
    throw new ApiError("Lỗi máy chủ (" + res.status + ").", { status: res.status, code: "BAD_RESPONSE" });
  }
  return { res };   // phản hồi nhị phân thành công
}

// ---------- Trạng thái và hiển thị ----------

function setStatus(message, isError = false) {
  const el = $("status");
  el.textContent = message;
  el.className = isError ? "error" : "";
}

function setBusy(value) {
  busy = value;
  document.querySelectorAll("button").forEach((b) => (b.disabled = value));
}

// Gộp dữ liệu server trả về (upload, áp dụng thao tác, hoặc lỗi 409) vào trạng thái hiện tại
function syncFrom(data) {
  const prev = image || {};
  image = {
    id: data.image_id ?? prev.id,
    version: data.version ?? data.current_version ?? prev.version,
    historyLength: data.history_length ?? prev.historyLength,
    current: data.current ?? prev.current,
    previewUrl: data.preview_url ?? prev.previewUrl,
    originalUrl: data.original_preview_url ?? prev.originalUrl,
    metadata: data.metadata ?? prev.metadata,
    expiresAt: data.expires_at ?? prev.expiresAt,
  };
}

function gcd(a, b) {
  return b ? gcd(b, a % b) : a;
}

function renderInfo() {
  const m = image.metadata;
  const c = image.current;
  const g = gcd(c.width, c.height);
  const rows = [
    ["Tên gốc", m.original_name],
    ["Định dạng gốc", m.format.toUpperCase()],
    ["Dung lượng gốc", (m.size_bytes / 1024).toFixed(1) + " KB"],
    ["Kích thước hiện tại", c.width + " × " + c.height + " px"],
    ["Tỷ lệ", c.width / g + ":" + c.height / g + " (" + (c.width / c.height).toFixed(3) + ")"],
    ["Chế độ màu", c.color_mode === "grayscale" ? "Ảnh xám" : "Màu"],
    ["Kênh alpha", c.has_alpha ? "Có" : "Không"],
    ["Số bước đã áp dụng", String(image.historyLength)],
    ["Hết hạn lúc", new Date(image.expiresAt).toLocaleTimeString("vi-VN")],
  ];
  const meta = $("meta");
  meta.replaceChildren();
  for (const [k, v] of rows) {   // dùng textContent để tên tệp không chèn được HTML
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = k;
    dd.textContent = v;
    meta.append(dt, dd);
  }
  $("editor").hidden = false;
}

// <img src> không đọc được JSON lỗi (409, 404...), nên tải bằng fetch rồi tạo Object URL
async function loadPreview(key, img, url) {
  const { res } = await request(url);
  const type = res.headers.get("Content-Type") || "";
  if (!type.startsWith("image/")) {
    throw new ApiError("Phản hồi ảnh không hợp lệ.", { code: "BAD_RESPONSE" });
  }
  const objectUrl = URL.createObjectURL(await res.blob());
  if (objectUrls[key]) URL.revokeObjectURL(objectUrls[key]);   // giải phóng bản cũ
  objectUrls[key] = objectUrl;
  img.src = objectUrl;
}

async function refreshView() {
  renderInfo();
  if (originalLoadedFor !== image.id) {   // ảnh gốc chỉ cần tải một lần cho mỗi ảnh
    await loadPreview("orig", $("orig"), image.originalUrl);
    originalLoadedFor = image.id;
  }
  await loadPreview("result", $("result"), image.previewUrl);
}

function resetEditor() {
  image = null;
  originalLoadedFor = null;
  $("editor").hidden = true;
  for (const key of Object.keys(objectUrls)) {
    if (objectUrls[key]) URL.revokeObjectURL(objectUrls[key]);
    objectUrls[key] = null;
  }
  $("orig").removeAttribute("src");
  $("result").removeAttribute("src");
}

async function handleError(err) {
  if (err.code === "IMAGE_NOT_FOUND") {
    resetEditor();
    setStatus("Ảnh đã hết hạn hoặc không còn khả dụng. Hãy tải ảnh lên lại.", true);
  } else if (err.code === "VERSION_CONFLICT" && image) {
    // Đồng bộ theo trạng thái server trả về; không tự đổi version rồi gửi lại thao tác
    syncFrom(err.data);
    try {
      await refreshView();
    } catch { /* giữ thông báo bên dưới */ }
    setStatus("Ảnh đã thay đổi ở nơi khác nên giao diện đã đồng bộ lại. Hãy kiểm tra ảnh rồi thử lại.", true);
  } else {
    setStatus(err.message, true);
  }
}

async function run(task) {
  setBusy(true);
  try {
    await task();
  } catch (err) {
    await handleError(err);
  } finally {
    setBusy(false);   // luôn mở khóa giao diện, kể cả khi phản hồi không phải JSON hợp lệ
  }
}

// ---------- Nhận ảnh: chọn tệp hoặc kéo thả ----------

// Kiểm tra dung lượng nhanh cho đỡ chờ; loại tệp do server quyết định theo nội dung
function handleFiles(files) {
  if (busy || files.length === 0) return;
  const file = files[0];
  const extra = files.length > 1;
  run(async () => {
    if (file.size > MAX_BYTES) throw new ApiError("Ảnh vượt quá 10 MiB.");
    const form = new FormData();   // không tự đặt Content-Type: trình duyệt thêm boundary
    form.append("image", file);
    setStatus("Đang tải ảnh lên...");
    const { data } = await request("/api/images", { method: "POST", body: form });
    // Chỉ thay ảnh đang chỉnh khi upload thành công; upload lỗi thì ảnh cũ vẫn giữ nguyên
    syncFrom(data);
    originalLoadedFor = null;
    await refreshView();
    setStatus(extra ? "Đã tải ảnh lên. Chỉ nhận một ảnh, đã dùng ảnh đầu tiên." : "Đã tải ảnh lên.");
  });
}

$("file").addEventListener("change", (e) => {
  handleFiles(Array.from(e.target.files));   // chép ra mảng trước khi xóa giá trị input
  e.target.value = "";                       // để chọn lại cùng một tệp vẫn kích hoạt sự kiện
});

const zone = $("dropzone");
const hasFiles = (e) => e.dataTransfer && Array.from(e.dataTransfer.types).includes("Files");

["dragenter", "dragover"].forEach((type) =>
  zone.addEventListener(type, (e) => {
    if (!hasFiles(e)) return;   // bỏ qua khi kéo văn bản hoặc liên kết
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
    zone.classList.add("over");
  })
);

zone.addEventListener("dragleave", (e) => {
  if (!zone.contains(e.relatedTarget)) zone.classList.remove("over");
});

zone.addEventListener("drop", (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  zone.classList.remove("over");
  handleFiles(Array.from(e.dataTransfer.files));
});

// Thả ảnh trượt ra ngoài vùng nhận: không để trình duyệt mở ảnh và rời khỏi trang
["dragover", "drop"].forEach((type) =>
  window.addEventListener(type, (e) => {
    if (hasFiles(e)) e.preventDefault();
  })
);

// ---------- Thao tác trên ảnh ----------

$("gray").addEventListener("click", () =>
  run(async () => {
    setStatus("Đang xử lý...");
    let data, message;
    try {
      ({ data, message } = await request("/api/images/" + image.id + "/operations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ operation: "grayscale", parameters: {}, expected_version: image.version }),
      }));
    } catch (err) {
      if (err.code === "NETWORK_ERROR") {
        // Chưa biết server đã ghi bước này chưa: không tự gửi lặp, bắt đầu phiên mới
        resetEditor();
        throw new ApiError(
          "Mất kết nối khi áp dụng thao tác nên chưa rõ ảnh đã đổi hay chưa. Hãy tải ảnh lên lại để bắt đầu phiên chỉnh sửa mới.",
          { code: "NETWORK_ERROR_ON_APPLY" }
        );
      }
      throw err;
    }
    syncFrom(data);
    await refreshView();
    setStatus(message);
  })
);

// Tên tệp tải xuống: bỏ đuôi, bỏ ký tự hợp đồng không cho phép, tối đa 100 ký tự
function downloadStem(name) {
  const stem = name.replace(/\.[^.]+$/, "").replace(/[\u0000-\u001f\u007f<>:"\/\\|?*]/g, "").trim().slice(0, 100);
  return stem || "edited-image";
}

$("download").addEventListener("click", () =>
  run(async () => {
    const format = $("fmt").value;
    const filename = downloadStem(image.metadata.original_name);
    const body = { expected_version: image.version, format, filename };
    if (format !== "png") body.quality = 90;   // PNG không được gửi quality
    setStatus("Đang chuẩn bị tệp...");
    const { res } = await request("/api/images/" + image.id + "/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const type = res.headers.get("Content-Type") || "";
    if (!type.startsWith("image/")) {   // không biến JSON lỗi thành tệp ảnh
      throw new ApiError("Phản hồi tải xuống không hợp lệ.", { code: "BAD_RESPONSE" });
    }
    const url = URL.createObjectURL(await res.blob());
    const a = document.createElement("a");
    a.href = url;
    a.download = filename + "." + format;
    a.click();
    URL.revokeObjectURL(url);
    setStatus("Đã tải xuống.");
  })
);