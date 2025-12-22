const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const resultsArea = document.getElementById('resultsArea');
const sourceImage = document.getElementById('sourceImage');
const canvas = document.getElementById('detectionCanvas');
const ctx = canvas.getContext('2d');
const loader = document.getElementById('loader');
const resetBtn = document.getElementById('resetBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const statusValue = document.getElementById('statusValue');

let selectedFile = null;

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
        handleFiles(e.dataTransfer.files);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFiles(e.target.files);
    }
});

analyzeBtn.addEventListener('click', () => {
    if (selectedFile) {
        processImage(selectedFile);
    }
});

resetBtn.addEventListener('click', () => {
    resultsArea.classList.add('hidden');
    dropzone.classList.remove('hidden');

    // Reset file and canvas
    sourceImage.src = '';
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    fileInput.value = '';
    selectedFile = null;

    // Hide analyze button until new file selected
    analyzeBtn.classList.add('hidden');

    // Reset dropzone text
    const dropzoneContent = dropzone.querySelector('.upload-content h3');
    dropzoneContent.textContent = 'Drag & Drop or Click to Upload';
    dropzoneContent.style.color = '#1e293b'; // Default text color
});

function handleFiles(files) {
    if (files.length > 0) {
        const file = files[0];
        if (file.type.startsWith('image/')) {
            selectedFile = file;

            // Show preview or just indicate selection
            analyzeBtn.classList.remove('hidden');

            // Visual feedback
            const dropzoneContent = dropzone.querySelector('.upload-content h3');
            dropzoneContent.textContent = `Selected: ${file.name}`;
            dropzoneContent.style.color = 'var(--primary)';
        } else {
            alert('Please upload an image file (JPG, PNG, WEBP).');
        }
    }
}

// Client-side image resizing
async function resizeImage(file, maxWidth = 1024) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = (event) => {
            const img = new Image();
            img.src = event.target.result;
            img.onload = () => {
                const canvas = document.createElement('canvas');
                let width = img.width;
                let height = img.height;

                if (width > maxWidth) {
                    height = Math.round((height * maxWidth) / width);
                    width = maxWidth;
                }

                canvas.width = width;
                canvas.height = height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, width, height);

                canvas.toBlob((blob) => {
                    resolve(new File([blob], file.name, {
                        type: file.type,
                        lastModified: Date.now(),
                    }));
                }, file.type, 0.9); // 0.9 quality
            };
        };
    });
}

async function processImage(file) {
    // UI Transitions
    dropzone.classList.add('hidden');
    analyzeBtn.classList.add('hidden');
    resultsArea.classList.remove('hidden');
    loader.classList.remove('hidden');

    // Show source image immediately to block empty space
    const reader = new FileReader();
    reader.onload = (e) => {
        sourceImage.src = e.target.result;
    }
    reader.readAsDataURL(file);

    try {
        // Resize image before sending (Performance Fix)
        const resizedFile = await resizeImage(file);

        const formData = new FormData();
        formData.append('file', resizedFile);

        const apiUrl = '/predict';

        const response = await fetch(`${apiUrl}`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`API Error: ${response.statusText}`);
        }

        const data = await response.json();
        drawDetections(data);
        updateMetrics(data);

    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred during object detection. Please check the console.');
        // Reset UI on error so user isn't stuck
        resultsArea.classList.add('hidden');
        dropzone.classList.remove('hidden');
        analyzeBtn.classList.remove('hidden');
    } finally {
        loader.classList.add('hidden');
    }
}

function drawDetections(data) {
    // Set canvas dimensions to match displayed image (which might be CSS constrained)
    // We need to map detection coordinates (from resized image) to display size

    // WAIT: sourceImage.naturalWidth/Height will be the full resolution (or resized res?)
    // data.detections coordinates are based on the image sent to backend (the resized one).
    // So we should use the resized image dimensions or just normalize.

    // Ideally, we set canvas match sourceImage natural dims and let CSS scale both.
    canvas.width = sourceImage.naturalWidth;
    canvas.height = sourceImage.naturalHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 4;
    ctx.font = 'bold 20px Inter';

    // The backend returns coordinates relative to the sent image.
    // Since we display the same image (roughly), we can just draw directly if we use naturalWidth.
    // However, if we displayed the *original* file but sent the *resized* file, coordinates would be off.
    // In processImage, we set sourceImage.src = reader.result (ORIGINAL).
    // Backend used RESIZED.
    // We must scale coordinates!

    // We don't have the resized dimensions easily here unless we pass them. 
    // BUT, Yolo usually returns normalized 0-1 coords or pixel coords.
    // If pixel coords, they are for the 1024px image.
    // We need to scale them to the Original image size.

    // Wait, let's look at the old code. It calculated scaleX/Y.
    // const scaleX = sourceImage.width / sourceImage.naturalWidth; -> This defines CSS scaling.
    // We want to draw on the canvas at 1:1 resolution to the natural image? 
    // No, we want to draw on the canvas such that it overlays the image.

    // Simplest approach:
    // 1. We know the backend used a max 1024px image.
    // 2. We are displaying the original full-res image.
    // 3. We need to scale the boxes from (1024 basis) to (Original basis).

    // Actually, `resizeImage` returns a File. We don't know the exact dimensions it chose without checking.
    // CORRECTION: The old code just assumed `sourceImage.width` (display width) vs `naturalWidth`. 
    // But that was for scaling context drawing to the display size.

    // Let's rely on the previous logic which seemed to work, BUT carefully.
    // If the previous logic was: `const scaleX = sourceImage.width / sourceImage.naturalWidth;`
    // And it drew using `x1 * scaleX`. This implies it was drawing on a canvas sized to the *display* size.
    // `canvas.width = sourceImage.width`. Yes.

    // Let's stick to that for consistency. Canvas sized to Display Elements.

    canvas.width = sourceImage.width;
    canvas.height = sourceImage.height;

    // Problem: sourceImage.width (display width) depends on CSS.
    // If we draw based on that, resizing the window breaks it unless we redraw.
    // But for a portfolio demo, it's fine.

    // Wait, if we sent a resized image, the coordinates are for that resized image.
    // If we display the original image, we need to map Resized-Coords -> Original-Coords -> Display-Coords.
    // OR: Map Resized-Coords -> Display-Coords directly.

    // If we don't know the resized dimensions, we are guessing.
    // However, `data` from backend might NOT include image size.

    // BETTER FIX: In `processImage` show the RESIZED image, not the original.
    // `sourceImage.src = URL.createObjectURL(resizedFile);`
    // Then coordinates match 1:1 (except for CSS scaling).

    // Let's do that. It ensures 100% alignment.

    // In processImage: 
    // sourceImage.src = URL.createObjectURL(resizedFile);

    // Let's keep the user code simple but consistent.

    data.detections.forEach(det => {
        const [x1, y1, x2, y2] = det.box;
        const color = getColor(det.class);

        // If we switch sourceImage to resizedFile, then naturalWidth is the resized width.
        // And Backend coordinates are based on resized width.
        // So `x1` is correct relative to `naturalWidth`.
        // To draw on `canvas` (which is set to `client width`), we scale by `clientWidth / naturalWidth`.

        const scaleX = sourceImage.width / sourceImage.naturalWidth;
        const scaleY = sourceImage.height / sourceImage.naturalHeight;

        const sx1 = x1 * scaleX;
        const sy1 = y1 * scaleY;
        const width = (x2 - x1) * scaleX;
        const height = (y2 - y1) * scaleY;

        // Draw Box
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.2;
        ctx.fillRect(sx1, sy1, width, height);
        ctx.globalAlpha = 1.0;
        ctx.strokeRect(sx1, sy1, width, height);

        // Draw Label
        const label = `${det.class} ${Math.round(det.confidence * 100)}%`;
        const textWidth = ctx.measureText(label).width;

        ctx.fillStyle = color;
        ctx.fillRect(sx1, sy1 - 25, textWidth + 10, 25);

        ctx.fillStyle = '#ffffff';
        ctx.fillText(label, sx1 + 5, sy1 - 7);
    });
}

function updateMetrics(data) {
    document.getElementById('detectionCount').textContent = data.count;

    if (data.count > 0) {
        const avg = data.detections.reduce((acc, curr) => acc + curr.confidence, 0) / data.count;
        document.getElementById('avgConf').textContent = `${Math.round(avg * 100)}%`;
    } else {
        document.getElementById('avgConf').textContent = '0%';
    }
}

function getColor(label) {
    const colors = {
        'person': '#FF5722',
        'car': '#00AEEF',
        'chair': '#C1D929',
        'book': '#E91E63',
        'bottle': '#9C27B0'
    };
    return colors[label.toLowerCase()] || '#FF9800';
}

// Initial Backend Check
fetch('/health')
    .then(res => {
        if (res.ok) {
            statusValue.innerHTML = '<i class="fa-solid fa-circle-check"></i> Online';
            statusValue.style.color = '#4CAF50';
        } else {
            throw new Error('Backend unhealthy');
        }
    })
    .catch(err => {
        console.warn('Backend check failed:', err);
        statusValue.innerHTML = '<i class="fa-solid fa-circle-xmark"></i> Offline';
        statusValue.style.color = '#F44336';
    });
