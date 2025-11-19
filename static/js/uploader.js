// Uploader specific functionality

const fileInput = document.getElementById('fileInput');
const processBtn = document.getElementById('processBtn');
const progressSection = document.getElementById('progressSection');
const progressBar = document.getElementById('progressBar');
const progressText = document.getElementById('progressText');
const scaleFactor = document.getElementById('scaleFactor');

if (processBtn) {
    processBtn.addEventListener('click', () => {
        if (fileInput.files.length > 0) {
            uploadAndProcess(fileInput.files[0]);
        } else {
            alert('Please select a file first!');
        }
    });
}

function uploadAndProcess(file) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('scale', scaleFactor.value);

    progressSection.style.display = 'block';

    fetch('/api/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log('✅ Uploaded:', data);
        // Simulate progress
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 30;
            if (progress >= 100) progress = 100;
            progressBar.value = progress;
            progressText.textContent = Math.round(progress) + '%';
            if (progress >= 100) {
                clearInterval(interval);
                alert('✅ Processing complete!');
            }
        }, 500);
    })
    .catch(error => {
        console.error('❌ Error:', error);
        alert('Processing failed!');
    });
}
