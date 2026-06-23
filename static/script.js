// Configuration: Set this to your deployed backend URL (e.g. "https://synthoct-backend.onrender.com")
// Leave empty for local relative path development.
const API_BASE = "";

document.addEventListener("DOMContentLoaded", () => {
    // --- DOM Elements ---
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabPanes = document.querySelectorAll(".tab-pane");
    
    const selectDirection = document.getElementById("select-direction");
    const sampleBtns = document.querySelectorAll(".sample-btn");
    
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const dropPrompt = document.getElementById("drop-prompt");
    const previewContainer = document.getElementById("preview-container");
    const sourcePreview = document.getElementById("source-preview");
    const btnClear = document.getElementById("btn-clear");
    
    const btnTranslate = document.getElementById("btn-translate");
    
    const resultZone = document.getElementById("result-zone");
    const resultPlaceholder = document.getElementById("result-placeholder");
    const resultContainer = document.getElementById("result-container");
    const resultImage = document.getElementById("result-image");
    const btnDownload = document.getElementById("btn-download");
    const spinnerContainer = document.getElementById("spinner-container");
    
    const statusBadge = document.getElementById("status-badge");
    const statusText = document.getElementById("status-text");

    let activeSourceImage = null; // Store base64 or file of uploaded image
    
    // --- Navigation Tabs ---
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            tabBtns.forEach(b => b.classList.remove("active"));
            tabPanes.forEach(p => p.classList.remove("active"));
            
            btn.classList.add("active");
            const tabId = btn.getAttribute("data-tab");
            document.getElementById(`tab-${tabId}`).classList.add("active");
        });
    });

    // --- Status Badge ---
    async function checkServerStatus() {
        try {
            const response = await fetch(`${API_BASE}/api/status`);
            if (response.ok) {
                const data = await response.json();
                statusBadge.className = "badge"; // Reset classes
                if (data.status === "live") {
                    statusBadge.classList.add("badge-live");
                    statusText.textContent = "Live Inference Model Active";
                } else {
                    statusBadge.classList.add("badge-simulated");
                    statusText.textContent = "Simulated Demo Mode (No model checkpoint)";
                }
            } else {
                throw new Error("Server error");
            }
        } catch (e) {
            statusBadge.className = "badge badge-offline";
            statusText.textContent = "Server Offline";
        }
    }
    checkServerStatus();

    // --- Translation Direction Toggle ---
    selectDirection.addEventListener("change", () => {
        const direction = selectDirection.value;
        
        // Toggle sample image visibility based on direction
        sampleBtns.forEach(btn => {
            const btnType = btn.getAttribute("data-type");
            if (direction === "xray2ct") {
                if (btnType === "xray") btn.classList.remove("hidden");
                else btn.classList.add("hidden");
            } else {
                if (btnType === "ct") btn.classList.remove("hidden");
                else btn.classList.add("hidden");
            }
            btn.classList.remove("selected");
        });

        // Reset inputs and results on direction change
        clearWorkspace();
    });

    // --- Loading Sample Images ---
    sampleBtns.forEach(btn => {
        btn.addEventListener("click", async () => {
            sampleBtns.forEach(b => b.classList.remove("selected"));
            btn.classList.add("selected");
            
            const relativePath = btn.getAttribute("data-img");
            showLoadingInPreview();
            
            try {
                // Fetch sample image and convert to base64
                const response = await fetch(relativePath);
                const blob = await response.blob();
                
                const reader = new FileReader();
                reader.readAsDataURL(blob);
                reader.onloadend = () => {
                    setSourceImage(reader.result);
                };
            } catch (err) {
                console.error("Failed to load sample image", err);
                resetSourcePreview();
            }
        });
    });

    // --- File Upload & Drag-and-Drop ---
    dropZone.addEventListener("click", (e) => {
        // Prevent clicking fileInput again if clicking the clear button
        if (e.target.closest("#btn-clear")) return;
        fileInput.click();
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files && fileInput.files[0]) {
            handleImageFile(fileInput.files[0]);
        }
    });

    // Drag-and-Drop handlers
    ["dragenter", "dragover"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add("dragover");
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove("dragover");
        }, false);
    });

    dropZone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files[0]) {
            handleImageFile(files[0]);
        }
    });

    function handleImageFile(file) {
        if (!file.type.startsWith("image/")) {
            alert("Please upload a valid image file.");
            return;
        }
        
        // Remove sample selections
        sampleBtns.forEach(b => b.classList.remove("selected"));
        
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = () => {
            setSourceImage(reader.result);
        };
    }

    // --- State setters ---
    function setSourceImage(base64Str) {
        activeSourceImage = base64Str;
        sourcePreview.src = base64Str;
        
        dropPrompt.classList.add("hidden");
        previewContainer.classList.remove("hidden");
        btnTranslate.disabled = false;
        
        // Reset old results
        resetResultZone();
    }

    function showLoadingInPreview() {
        dropPrompt.classList.add("hidden");
        previewContainer.classList.add("hidden");
        btnTranslate.disabled = true;
    }

    function resetSourcePreview() {
        activeSourceImage = null;
        sourcePreview.src = "";
        fileInput.value = "";
        
        previewContainer.classList.add("hidden");
        dropPrompt.classList.remove("hidden");
        btnTranslate.disabled = true;
    }

    function resetResultZone() {
        resultImage.src = "";
        resultContainer.classList.add("hidden");
        spinnerContainer.classList.add("hidden");
        resultPlaceholder.classList.remove("hidden");
    }

    function clearWorkspace() {
        resetSourcePreview();
        resetResultZone();
        sampleBtns.forEach(b => b.classList.remove("selected"));
    }

    btnClear.addEventListener("click", (e) => {
        e.stopPropagation();
        clearWorkspace();
    });

    // --- Running AI Translation (POST request to backend) ---
    btnTranslate.addEventListener("click", async () => {
        if (!activeSourceImage) return;

        // UI Updates: Disable buttons, show spinner, hide result image
        btnTranslate.disabled = true;
        resultPlaceholder.classList.add("hidden");
        resultContainer.classList.add("hidden");
        spinnerContainer.classList.remove("hidden");

        const direction = selectDirection.value;

        try {
            const response = await fetch(`${API_BASE}/api/translate`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    image: activeSourceImage,
                    direction: direction
                })
            });

            if (response.ok) {
                const data = await response.json();
                
                // Show translated result
                resultImage.src = data.image;
                btnDownload.href = data.image;
                
                spinnerContainer.classList.add("hidden");
                resultContainer.classList.remove("hidden");
            } else {
                const errData = await response.json();
                throw new Error(errData.error || "Translation failed");
            }
        } catch (e) {
            console.error(e);
            alert(`Translation error: ${e.message}`);
            resetResultZone();
        } finally {
            btnTranslate.disabled = false;
        }
    });
});
