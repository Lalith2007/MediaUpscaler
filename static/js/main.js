// Main application JavaScript

document.addEventListener('DOMContentLoaded', function() {
    console.log('🎬 Ultimate Editing Suite loaded!');
    
    // Initialize dropzone areas
    const dropZones = document.querySelectorAll('.upload-zone');
    dropZones.forEach(zone => {
        zone.addEventListener('dragover', handleDragOver);
        zone.addEventListener('drop', handleDrop);
        zone.addEventListener('click', () => {
            const input = zone.querySelector('input[type="file"]');
            if (input) input.click();
        });
    });
});

function handleDragOver(e) {
    e.preventDefault();
    this.style.background = 'rgba(255, 255, 255, 0.3)';
}

function handleDrop(e) {
    e.preventDefault();
    this.style.background = 'rgba(255, 255, 255, 0.1)';
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        uploadFile(files[0]);
    }
}

function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    fetch('/api/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log('✅ File uploaded:', data);
        alert('File uploaded successfully!');
    })
    .catch(error => {
        console.error('❌ Upload failed:', error);
        alert('Upload failed!');
    });
}
