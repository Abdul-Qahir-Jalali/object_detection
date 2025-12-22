const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const resultsArea = document.getElementById('resultsArea');
const canvas = document.getElementById('detectionCanvas');
const ctx = canvas.getContext('2d');
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
    ctx.clearRect(0, 0, canvas.width, canvas.height);
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
        canvas.width = sourceImage.naturalWidth;
        canvas.height = sourceImage.naturalHeight;
        ctx.drawImage(sourceImage, 0, 0);

        // Resize if too big (Client-side optimization)
        let processedFile = file;
        if (file.size > 1024 * 1024 || sourceImage.naturalWidth > 1024) {
            processedFile = await resizeImage(file, 1024);
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
                }, file.type, 0.9);
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    });
}

function drawDetections(data) {
    loader.classList.add('hidden');
    const detections = data.detections;
    const imgWidth = canvas.width;
    const imgHeight = canvas.height;

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

        // Draw Box
        ctx.strokeStyle = color;
        ctx.lineWidth = 4;
        ctx.strokeRect(boxX, boxY, boxW, boxH);

        // Draw Label Background
        ctx.fillStyle = color;
        const text = `${det.class} ${Math.round(det.confidence * 100)}%`;
        const textWidth = ctx.measureText(text).width + 10;
        const textHeight = 24;
        ctx.fillRect(boxX, boxY - textHeight, textWidth, textHeight);

        // Draw Text
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 14px "Inter", sans-serif';
        ctx.fillText(text, boxX + 5, boxY - 7);
    });
}
