const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const resultsArea = document.getElementById('resultsArea');
const sourceImage = document.getElementById('sourceImage');
const canvas = document.getElementById('detectionCanvas');
const ctx = canvas.getContext('2d');
const loader = document.getElementById('loader');
const resetBtn = document.getElementById('resetBtn');

const statusValue = document.getElementById('statusValue');

// Event Listeners
dropzone.addEventListener('click', () => fileInput.click());

dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
});

dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
});

dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFile(e.target.files[0]);
    }
});

resetBtn.addEventListener('click', resetApp);

// Functions
function resetApp() {
    resultsArea.classList.add('hidden');
    dropzone.classList.remove('hidden');
    fileInput.value = '';
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

async function handleFile(file) {
    if (!file.type.startsWith('image/')) return;

    // Show Image
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.src = e.target.result;
        img.onload = () => {
            // Resize logic
            const MAX_WIDTH = 1024;
            const MAX_HEIGHT = 1024;
            let width = img.width;
            let height = img.height;

            if (width > height) {
                if (width > MAX_WIDTH) {
                    height *= MAX_WIDTH / width;
                    width = MAX_WIDTH;
                }
            } else {
                if (height > MAX_HEIGHT) {
                    width *= MAX_HEIGHT / height;
                    height = MAX_HEIGHT;
                }
            }

            // Create resizing canvas
            const resizeCanvas = document.createElement('canvas');
            resizeCanvas.width = width;
            resizeCanvas.height = height;
            const ctxResize = resizeCanvas.getContext('2d');
            ctxResize.drawImage(img, 0, 0, width, height);

            // Update main canvas size
            sourceImage.src = resizeCanvas.toDataURL('image/jpeg');
            canvas.width = width;
            canvas.height = height;
            canvas.style.width = '100%';
            canvas.style.height = '100%';

            dropzone.classList.add('hidden');
            resultsArea.classList.remove('hidden');
            loader.classList.remove('hidden');

            // Send resized image as blob
            resizeCanvas.toBlob((blob) => {
                const resizedFile = new File([blob], file.name, { type: 'image/jpeg' });
                detectObjects(resizedFile);
            }, 'image/jpeg', 0.85);
        };
    };
    reader.readAsDataURL(file);
}

async function detectObjects(file) {
    const formData = new FormData();
    formData.append('file', file);

    const apiUrl = '/predict';

    try {
        const response = await fetch(`${apiUrl}`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('API Error');

        const data = await response.json();
        drawDetections(data.detections);
        updateMetrics(data.detections);

    } catch (error) {
        console.error(error);
        alert('Detection failed. Ensure the backend is running and the URL is correct.');
    } finally {
        loader.classList.add('hidden');
    }
}

function drawDetections(detections) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Line settings
    ctx.lineWidth = 4;
    ctx.font = 'bold 24px Inter';

    detections.forEach(det => {
        const [x1, y1, x2, y2] = det.bbox;
        const width = x2 - x1;
        const height = y2 - y1;
        const label = `${det.class} ${Math.round(det.confidence * 100)}%`;

        // Color based on class ID (simple hash)
        const hue = (det.class_id * 137) % 360;
        const color = `hsl(${hue}, 70%, 50%)`;

        // Draw Box
        ctx.strokeStyle = color;
        ctx.strokeRect(x1, y1, width, height);

        // Draw Label Background
        ctx.fillStyle = color;
        const textWidth = ctx.measureText(label).width;
        ctx.fillRect(x1, y1 - 30, textWidth + 10, 30);

        // Draw Label Text
        ctx.fillStyle = '#fff';
        ctx.fillText(label, x1 + 5, y1 - 7);
    });
}

function updateMetrics(detections) {
    document.getElementById('detectionCount').innerText = detections.length;

    if (detections.length > 0) {
        const avg = detections.reduce((acc, curr) => acc + curr.confidence, 0) / detections.length;
        document.getElementById('avgConf').innerText = `${Math.round(avg * 100)}%`;
    } else {
        document.getElementById('avgConf').innerText = '0%';
    }
}

// Initial Backend Check (Optional)
// fetch(apiUrlInput.value.replace('/predict', '/')).then(r => r.ok ? statusValue.innerText = 'Connected' : null);
