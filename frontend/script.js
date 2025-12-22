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
