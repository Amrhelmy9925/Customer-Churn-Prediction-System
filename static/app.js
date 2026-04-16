// ─── Sidebar Toggle ───
document.getElementById("menuToggle").addEventListener("click", () => {
  document.getElementById("sidebar").classList.toggle("collapsed");
  // Mobile: use mobile-open class
  if (window.innerWidth <= 768) {
    document.getElementById("sidebar").classList.toggle("mobile-open");
  }
});

// ─── Loading ───
function showLoading() {
  document.getElementById("loading").classList.add("show");
}
function hideLoading() {
  document.getElementById("loading").classList.remove("show");
}

// ─── Toast Notifications ───
function showToast(message, type = "info") {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  const icons = {
    success: "fa-check-circle",
    error: "fa-times-circle",
    info: "fa-info-circle",
  };
  toast.innerHTML = `<i class="fas ${icons[type] || icons.info}"></i>${message}`;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateX(40px)";
    toast.style.transition = "all 0.3s ease";
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}

// ─── LocalStorage helpers ───
function getPredictions() {
  return JSON.parse(localStorage.getItem("churn_predictions") || "[]");
}
function savePredictions(data) {
  localStorage.setItem("churn_predictions", JSON.stringify(data));
}
