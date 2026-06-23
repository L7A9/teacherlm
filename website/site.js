const releaseLabel = document.querySelector("#release-label");

fetch("https://api.github.com/repos/L7A9/teacherlm/releases/latest", {
  headers: { Accept: "application/vnd.github+json" },
})
  .then((response) => (response.ok ? response.json() : Promise.reject(new Error("release unavailable"))))
  .then((release) => {
    const asset = release.assets?.find((item) => item.name === "TeacherLM-Setup.exe");
    const size = asset?.size ? ` · ${(asset.size / 1024 / 1024).toFixed(0)} MB` : "";
    releaseLabel.textContent = `${release.tag_name} · Windows 10/11 · 64-bit${size}`;
  })
  .catch(() => {
    releaseLabel.textContent = "Windows 10/11 · 64-bit";
  });
