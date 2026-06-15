const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const controls = document.getElementById('controls');

dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) executeBimAutomationPipeline(file);
});

async function executeBimAutomationPipeline(file) {
    dropZone.innerHTML = `<p style="color: #10b981;">Processing 3D Data Layers...</p>`;
    
    const payload = new FormData();
    payload.append("file", file);
    
    try {
        // Post binary assets straight to your Python application layer server
        const response = await fetch("https://bim-ai-engine-production.up.railway.app/process-bim-model", {
            method: "POST",
            body: payload
        });
        
        const data = await response.json();
        
        if (data.status === "Success") {
            dropZone.innerHTML = `<p style="color: #10b981;">✓ Compilation Successful</p>`;
            controls.style.display = "flex";
            
            // Map return file system directory storage nodes to actions
            document.getElementById('download-dxf').href = `https://bim-ai-engine-production.up.railway.app/${data.dxf_blueprint}`;
            document.getElementById('download-csv').href = `https://bim-ai-engine-production.up.railway.app/${data.csv_schedule}`;
        } else {
            dropZone.innerHTML = `<p style="color: #ef4444;">Pipeline Execution Error</p>`;
        }
    } catch (err) {
        dropZone.innerHTML = `<p style="color: #ef4444;">Connection Failed</p>`;
        console.error(err);
    }
}