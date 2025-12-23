const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const resultsArea = document.getElementById('resultsArea');
const detectionCanvas = document.getElementById('detectionCanvas');
const ctx = detectionCanvas.getContext('2d');
const loader = document.getElementById('loader');
const resetBtn = document.getElementById('resetBtn');
const statusValue = document.getElementById('statusValue');

const analyzeBtn = document.getElementById('analyzeBtn');

let selectedFile = null;

// Event Listeners
dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
});
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files);
});
fileInput.addEventListener('change', () => {
    if (fileInput.files.length) handleFiles(fileInput.files);
});

analyzeBtn.addEventListener('click', () => {
    if (selectedFile) uploadImage(selectedFile);
});

resetBtn.addEventListener('click', () => {
    resultsArea.classList.add('hidden');
    dropzone.parentElement.querySelector('.analysis-controls').classList.remove('hidden'); // Show controls again
    dropzone.classList.remove('hidden');
    ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);
    fileInput.value = '';
    selectedFile = null;

    // Reset UI state
    analyzeBtn.disabled = true;


    // Reset Preview
    document.getElementById('previewImage').classList.add('hidden');
    dropzone.querySelector('.upload-content').classList.remove('hidden');
    dropzone.style.padding = '5rem 2rem'; // Restore original padding
});


function handleFiles(files) {
    const file = files[0];
    if (file.type.startsWith('image/')) {
        selectedFile = file;

        // Create Preview
        const reader = new FileReader();
        reader.onload = (e) => {
            const previewImage = document.getElementById('previewImage');
            const uploadContent = dropzone.querySelector('.upload-content');

            previewImage.src = e.target.result;
            previewImage.classList.remove('hidden');
            uploadContent.classList.add('hidden');

            // Adjust dropzone padding for image
            dropzone.style.padding = '2rem';
        };
        reader.readAsDataURL(file);

        // Enable Analyze Button
        analyzeBtn.disabled = false;

    } else {
        alert('Please upload an image file (JPG, PNG, WEBP).');
    }
}

async function uploadImage(file) {
    // Hide controls during processing
    dropzone.parentElement.querySelector('.analysis-controls').classList.add('hidden');
    dropzone.classList.add('hidden');
    resultsArea.classList.remove('hidden');
    loader.classList.remove('hidden');

    const sourceImage = document.getElementById('sourceImage');
    const objectUrl = URL.createObjectURL(file);
    sourceImage.src = objectUrl;

    sourceImage.onload = async () => {
        // Prepare Canvas
        detectionCanvas.width = sourceImage.naturalWidth;
        detectionCanvas.height = sourceImage.naturalHeight;
        ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height); // Clear for new drawing (overlay mode)

        // Resize if too big (Client-side optimization)
        let processedFile = file;
        // Optimization: Resize to 640px (YOLO Native). 
        // This is the fastest possible setting (matching model input) without losing accuracy.
        if (file.size > 1024 * 1024 || sourceImage.naturalWidth > 640) {
            processedFile = await resizeImage(file, 640);
        }

        const formData = new FormData();
        formData.append('file', processedFile);

        const apiUrl = '/predict';

        try {
            const response = await fetch(`${apiUrl}`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errText = await response.text();
                throw new Error(`API Error: ${response.status} ${errText}`);
            }

            const data = await response.json();
            drawDetections(data);

            // Check Backend Status
            statusValue.innerHTML = '<i class="fa-solid fa-circle-check"></i> Online';
            statusValue.style.color = 'var(--secondary)'; // Lime Green

        } catch (error) {
            console.error('Error:', error);
            alert('Detection failed. See console for details.');
            statusValue.innerHTML = '<i class="fa-solid fa-circle-xmark"></i> Offline';
            statusValue.style.color = '#ef4444'; // Red
            loader.classList.add('hidden');
        }
    };
}

// Client-side Resize Function
function resizeImage(file, maxWidth) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                const scale = maxWidth / img.width;
                if (scale >= 1) {
                    resolve(file); // No resize needed
                    return;
                }

                const canvas = document.createElement('canvas');
                canvas.width = maxWidth;
                canvas.height = img.height * scale;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

                canvas.toBlob((blob) => {
                    resolve(new File([blob], file.name, {
                        type: file.type,
                        lastModified: Date.now()
                    }));
                }, file.type, 0.7); // 0.7 Speed Quality
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    });
}

function drawDetections(data) {
    loader.classList.add('hidden');
    const detections = data.detections;
    const imgWidth = detectionCanvas.width;
    const imgHeight = detectionCanvas.height;

    // Dynamic Scale Factor:
    // Scale elements relative to a reference width (e.g., 1000px).
    // This ensures labels are readable even on 4K+ images when downscaled by CSS.
    const scale = Math.max(1, imgWidth / 1000);

    // Update Metrics
    document.getElementById('detectionCount').textContent = detections.length;
    const avgConf = detections.length ? (detections.reduce((acc, d) => acc + d.confidence, 0) / detections.length * 100).toFixed(1) : 0;
    document.getElementById('avgConf').textContent = `${avgConf}%`;

    detections.forEach(det => {
        const [x1, y1, x2, y2] = det.box; // Normalized [0-1]

        const boxX = x1 * imgWidth;
        const boxY = y1 * imgHeight;
        const boxW = (x2 - x1) * imgWidth;
        const boxH = (y2 - y1) * imgHeight;

        // Random Color for Class
        const color = `hsl(${Math.random() * 360}, 100%, 50%)`;

        // Scaled Styles
        const lineWidth = 6 * scale; // Thicker lines
        const fontSize = Math.round(24 * scale); // Larger font for readability
        const padding = 8 * scale;

        // Draw Box
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.strokeRect(boxX, boxY, boxW, boxH);

        // Draw Label Background
        ctx.fillStyle = color;
        const text = `${det.class} ${Math.round(det.confidence * 100)}%`;
        ctx.font = `bold ${fontSize}px "Inter", sans-serif`; // Set font before measuring
        const textMetrics = ctx.measureText(text);
        const textWidth = textMetrics.width + (padding * 2);
        const textHeight = fontSize + (padding * 2);

        // Ensure label stays within image bounds (optional logic could go here)
        // Draw background above box (or inside if at top edge)
        let labelY = boxY - textHeight;
        if (labelY < 0) labelY = boxY; // Flip inside if too high

        ctx.fillRect(boxX, labelY, textWidth, textHeight);

        // Draw Text
        ctx.fillStyle = '#fff';
        ctx.textBaseline = 'top';
        ctx.fillText(text, boxX + padding, labelY + padding);
    });
}

// --- Review Hub Logic ---
const navDashboard = document.getElementById('nav-dashboard');
const navReview = document.getElementById('nav-review');
const dashboardView = document.getElementById('dashboard-view');
const reviewView = document.getElementById('review-view');
const reviewImage = document.getElementById('reviewImage');
const reviewCanvas = document.getElementById('reviewCanvas');
const reviewCtx = reviewCanvas.getContext('2d');
const reviewFilename = document.getElementById('reviewFilename');
const btnVerify = document.getElementById('btnVerify');
const btnWrong = document.getElementById('btnWrong');
const correctionPanel = document.getElementById('correctionPanel');
const btnNextImage = document.getElementById('btnNextImage');

// Debug Trigger Logic
const btnDebugTrigger = document.getElementById('btnDebugTrigger');
if (btnDebugTrigger) {
    btnDebugTrigger.addEventListener('click', async () => {
        if (!confirm("Start Debug Retrain Check? This will check HF for >10 images.")) return;

        try {
            btnDebugTrigger.textContent = "Checking...";
            btnDebugTrigger.disabled = true;

            const res = await fetch('/debug-trigger', { method: 'POST' });
            const data = await res.json();

            alert("Debug Result:\n" + JSON.stringify(data, null, 2));
            console.log("Debug Trigger Response:", data);
        } catch (e) {
            alert("Error triggering debug: " + e.message);
        } finally {
            btnDebugTrigger.textContent = "Force Retrain (Debug Check)";
            btnDebugTrigger.disabled = false;
        }
    });
}

let reviewQueue = [];
let currentReviewIndex = 0;

// Tab Switching
navDashboard.addEventListener('click', () => {
    switchTab('dashboard');
});

navReview.addEventListener('click', () => {
    switchTab('review');
    loadReviewData();
});

function switchTab(tab) {
    if (tab === 'dashboard') {
        dashboardView.classList.remove('hidden');
        reviewView.classList.add('hidden');
        navDashboard.classList.add('active');
        navReview.classList.remove('active');
    } else {
        dashboardView.classList.add('hidden');
        reviewView.classList.remove('hidden');
        navDashboard.classList.remove('active');
        navReview.classList.add('active');
    }
}

// Review Data Loading
async function loadReviewData() {
    try {
        reviewFilename.textContent = "Loading data...";
        const response = await fetch('/list-unverified?limit=20');
        const data = await response.json();

        if (data.images && data.images.length > 0) {
            reviewQueue = data.images;
            currentReviewIndex = 0;
            showReviewImage();
        } else {
            reviewFilename.textContent = "No images found for review.";
            reviewImage.src = "";
            btnVerify.disabled = true;
            btnWrong.disabled = true;
        }
    } catch (error) {
        console.error("Error loading review data:", error);
        reviewFilename.textContent = "Error loading data.";
    }
}

function showReviewImage() {
    if (currentReviewIndex >= reviewQueue.length) {
        reviewFilename.textContent = "All loaded images reviewed!";
        reviewImage.src = "";
        reviewCtx.clearRect(0, 0, reviewCanvas.width, reviewCanvas.height);
        btnVerify.disabled = true;
        btnWrong.disabled = true;
        return;
    }

    const path = reviewQueue[currentReviewIndex];
    reviewFilename.textContent = path.split('/').pop(); // Show just filename

    // Reset Canvas
    reviewCtx.clearRect(0, 0, reviewCanvas.width, reviewCanvas.height);

    // Use Proxy Endpoint to fetch secure image
    reviewImage.onload = async () => {
        // Set canvas to match image
        reviewCanvas.width = reviewImage.naturalWidth;
        reviewCanvas.height = reviewImage.naturalHeight;

        // Fetch and Draw Predictions
        await loadAndDrawPredictions(path);
    };

    // Trigger load
    reviewImage.src = `/proxy-image?path=${encodeURIComponent(path)}`;

    // Reset UI
    correctionPanel.classList.add('hidden');
    btnVerify.disabled = false;
    btnWrong.disabled = false;
}

async function loadAndDrawPredictions(imagePath) {
    try {
        const response = await fetch(`/get-prediction-data?path=${encodeURIComponent(imagePath)}`);
        if (!response.ok) return; // No predictions found

        const data = await response.json();
        if (data && data.detections) {
            drawReviewDetections(data.detections);
        }
    } catch (e) {
        console.error("Could not load prediction overlay", e);
    }
}

function drawReviewDetections(detections) {
    const imgWidth = reviewCanvas.width;
    const imgHeight = reviewCanvas.height;

    // Scale for readability
    const scale = Math.max(1, imgWidth / 1000);

    detections.forEach(det => {
        const [x1, y1, x2, y2] = det.box;
        const boxX = x1 * imgWidth;
        const boxY = y1 * imgHeight;
        const boxW = (x2 - x1) * imgWidth;
        const boxH = (y2 - y1) * imgHeight;

        // Use a distinct color for Review to differentiate
        const color = '#3b82f6'; // Blue

        const lineWidth = 6 * scale;
        const fontSize = Math.round(24 * scale);
        const padding = 8 * scale;

        // Draw Box
        reviewCtx.strokeStyle = color;
        reviewCtx.lineWidth = lineWidth;
        reviewCtx.strokeRect(boxX, boxY, boxW, boxH);

        // Draw Label
        reviewCtx.fillStyle = color;
        const text = `${det.class} ${Math.round(det.confidence * 100)}%`;
        reviewCtx.font = `bold ${fontSize}px "Inter", sans-serif`;
        const textMetrics = reviewCtx.measureText(text);
        const textWidth = textMetrics.width + (padding * 2);
        const textHeight = fontSize + (padding * 2);

        let labelY = boxY - textHeight;
        if (labelY < 0) labelY = boxY;

        reviewCtx.fillRect(boxX, labelY, textWidth, textHeight);
        reviewCtx.fillStyle = '#fff';
        reviewCtx.textBaseline = 'top';
        reviewCtx.fillText(text, boxX + padding, labelY + padding);
    });
}


// Actions
btnVerify.addEventListener('click', async () => {
    await submitReview('verified');
});

btnWrong.addEventListener('click', () => {
    correctionPanel.classList.remove('hidden'); // Show dropdown
});

btnSubmitCorrection.addEventListener('click', async () => {
    const label = document.getElementById('correctLabelSelect').value;
    await submitReview('correction', label);
});

btnNextImage.addEventListener('click', () => {
    currentReviewIndex++;
    showReviewImage();
});

async function submitReview(decision, label = null) {
    const path = reviewQueue[currentReviewIndex];
    const payload = {
        filename: path,
        decision: decision,
        label: label
    };

    try {
        btnVerify.disabled = true;

        const response = await fetch('/submit-review', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const result = await response.json();

        // Handle Success or Error
        if (response.ok && result.status === 'success') {
            // Move to next
            currentReviewIndex++;
            showReviewImage();
        } else {
            // Extract error message safely
            const msg = result.message || (result.detail ? JSON.stringify(result.detail) : "Unknown Error");
            alert('Error saving review: ' + msg);
            btnVerify.disabled = false;
        }

    } catch (error) {
        console.error("Review Submit Error:", error);
        alert("Network error submission failed.");
        btnVerify.disabled = false;
    }
}
