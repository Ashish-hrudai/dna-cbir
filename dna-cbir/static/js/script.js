// Global variables
let currentQueryFile = null;
let currentAnalyzeFile = null;
let currentCompareFile = null;

// Tab switching
function switchTab(tabName) {
    // Hide all tabs
    const tabs = document.querySelectorAll('.tab-content');
    tabs.forEach(tab => tab.classList.remove('active'));
    
    // Remove active class from all buttons
    const buttons = document.querySelectorAll('.tab-btn');
    buttons.forEach(btn => btn.classList.remove('active'));
    
    // Show selected tab
    document.getElementById(`${tabName}-tab`).classList.add('active');
    
    // Add active class to clicked button
    event.target.classList.add('active');
}

// Handle query image upload
async function handleQueryUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            currentQueryFile = data.filename;
            
            // Show preview
            const preview = document.getElementById('query-preview');
            const img = document.getElementById('query-img');
            img.src = `/uploads/${data.filename}`;
            preview.style.display = 'block';
            
            // Enable search button
            document.getElementById('search-btn').disabled = false;
        } else {
            alert('Error uploading file: ' + data.error);
        }
    } catch (error) {
        console.error('Upload error:', error);
        alert('Error uploading file');
    }
}

// Perform image search
async function performSearch() {
    if (!currentQueryFile) {
        alert('Please upload a query image first');
        return;
    }
    
    const modelType = document.getElementById('model-type').value;
    const similarityMetric = document.getElementById('similarity-metric').value;
    const topK = parseInt(document.getElementById('top-k').value);
    const useMultiScale = document.getElementById('multi-scale').checked;
    
    // Show loading
    document.getElementById('loading').style.display = 'block';
    document.getElementById('search-results').style.display = 'none';
    
    try {
        const response = await fetch('/api/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query_file: currentQueryFile,
                model_type: modelType,
                k: topK,
                use_multi_scale: useMultiScale,
                similarity_metric: similarityMetric
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displaySearchResults(data.results);
        } else {
            alert('Search error: ' + data.error);
        }
    } catch (error) {
        console.error('Search error:', error);
        alert('Error performing search');
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
}

// Display search results
function displaySearchResults(results) {
    const resultsContainer = document.getElementById('search-results');
    const resultsGrid = document.getElementById('results-grid');
    
    resultsGrid.innerHTML = '';
    
    results.forEach((result, index) => {
        const resultItem = document.createElement('div');
        resultItem.className = 'result-item';
        
        resultItem.innerHTML = `
            <img src="data:image/png;base64,${result.image}" alt="${result.filename}">
            <div class="result-info">
                <h4>Rank ${index + 1}</h4>
                <span class="result-score">Score: ${result.score.toFixed(4)}</span>
                <p style="font-size: 0.8em; margin-top: 5px; color: #7f8c8d;">${result.filename}</p>
            </div>
        `;
        
        resultsGrid.appendChild(resultItem);
    });
    
    resultsContainer.style.display = 'block';
}

// Handle analyze image upload
async function handleAnalyzeUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            currentAnalyzeFile = data.filename;
            
            // Show preview
            const preview = document.getElementById('analyze-preview');
            const img = document.getElementById('analyze-img');
            img.src = `/uploads/${data.filename}`;
            preview.style.display = 'block';
            
            // Enable analyze button
            document.getElementById('analyze-btn').disabled = false;
        } else {
            alert('Error uploading file: ' + data.error);
        }
    } catch (error) {
        console.error('Upload error:', error);
        alert('Error uploading file');
    }
}

// Analyze image
async function analyzeImage() {
    if (!currentAnalyzeFile) {
        alert('Please upload an image first');
        return;
    }
    
    // Show loading
    document.getElementById('loading').style.display = 'block';
    document.getElementById('analysis-results').style.display = 'none';
    
    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                filename: currentAnalyzeFile
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayAnalysisResults(data);
        } else {
            alert('Analysis error: ' + data.error);
        }
    } catch (error) {
        console.error('Analysis error:', error);
        alert('Error analyzing image');
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
}

// Display analysis results
function displayAnalysisResults(data) {
    const statsDiv = document.getElementById('dna-stats');
    const sequenceDiv = document.getElementById('dna-sequence');
    
    const stats = data.dna_stats;
    const gcSatisfied = stats.gc_constraint_satisfied ? 
        '<span class="gc-satisfied">✓ Satisfied</span>' : 
        '<span class="gc-not-satisfied">✗ Not Satisfied</span>';
    
    statsDiv.innerHTML = `
        <div class="stat-item">
            <span class="stat-label">Number of Channels:</span>
            <span class="stat-value">${stats.num_channels}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">DNA Length per Channel:</span>
            <span class="stat-value">${stats.dna_length_per_channel} nucleotides</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Feature Vector Length:</span>
            <span class="stat-value">${stats.feature_vector_length}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Multi-Scale Feature Length:</span>
            <span class="stat-value">${stats.multi_scale_feature_length}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">GC Content:</span>
            <span class="stat-value">${stats.gc_content_percentage}%</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">GC Constraint (40-60%):</span>
            ${gcSatisfied}
        </div>
        <div class="stat-item">
            <span class="stat-label">Image Shape:</span>
            <span class="stat-value">${data.image_shape.join(' × ')}</span>
        </div>
    `;
    
    sequenceDiv.innerHTML = `
        <h4 style="color: #ecf0f1; margin-bottom: 10px;">Sample DNA Sequence (first 100 nucleotides):</h4>
        <code>${data.sample_dna_sequence}</code>
    `;
    
    document.getElementById('analysis-results').style.display = 'block';
}

// Handle compare image upload
async function handleCompareUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            currentCompareFile = data.filename;
            
            // Show preview
            const preview = document.getElementById('compare-preview');
            const img = document.getElementById('compare-img');
            img.src = `/uploads/${data.filename}`;
            preview.style.display = 'block';
            
            // Enable compare button
            document.getElementById('compare-btn').disabled = false;
        } else {
            alert('Error uploading file: ' + data.error);
        }
    } catch (error) {
        console.error('Upload error:', error);
        alert('Error uploading file');
    }
}

// Compare similarity metrics
async function compareMetrics() {
    if (!currentCompareFile) {
        alert('Please upload a query image first');
        return;
    }
    
    const topK = parseInt(document.getElementById('compare-k').value);
    
    // Show loading
    document.getElementById('loading').style.display = 'block';
    document.getElementById('comparison-results').style.display = 'none';
    
    try {
        const response = await fetch('/api/compare-metrics', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query_file: currentCompareFile,
                k: topK
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayComparisonResults(data.metric_comparisons);
        } else {
            alert('Comparison error: ' + data.error);
        }
    } catch (error) {
        console.error('Comparison error:', error);
        alert('Error comparing metrics');
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
}

// Display comparison results
function displayComparisonResults(comparisons) {
    const resultsDiv = document.getElementById('comparison-results');
    resultsDiv.innerHTML = '';
    
    const metrics = ['euclidean', 'cosine', 'chi_square', 'weighted'];
    const metricNames = {
        'euclidean': 'Euclidean Distance',
        'cosine': 'Cosine Similarity',
        'chi_square': 'Chi-Square Distance',
        'weighted': 'Weighted Multi-Metric'
    };
    
    metrics.forEach(metric => {
        const results = comparisons[metric];
        
        const metricDiv = document.createElement('div');
        metricDiv.className = 'metric-comparison';
        
        let resultsHTML = `<h4>${metricNames[metric]}</h4><div class="comparison-grid">`;
        
        results.forEach((result, index) => {
            resultsHTML += `
                <div class="comparison-item">
                    <p><strong>Rank ${index + 1}</strong></p>
                    <p style="font-size: 0.85em;">${result.filename}</p>
                    <p style="color: #3498db; font-weight: 600;">Score: ${result.score.toFixed(4)}</p>
                </div>
            `;
        });
        
        resultsHTML += '</div>';
        metricDiv.innerHTML = resultsHTML;
        resultsDiv.appendChild(metricDiv);
    });
    
    resultsDiv.style.display = 'block';
}

// Handle database upload
async function handleDatabaseUpload(event) {
    const files = event.target.files;
    if (files.length === 0) return;
    
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        formData.append('files[]', files[i]);
    }
    
    try {
        const response = await fetch('/api/database/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            alert(`Successfully uploaded ${data.total_uploaded} images`);
            loadDatabaseImages();
        } else {
            alert('Error uploading files');
        }
    } catch (error) {
        console.error('Upload error:', error);
        alert('Error uploading files');
    }
}

// Load database images
async function loadDatabaseImages() {
    try {
        const response = await fetch('/api/database/list');
        const data = await response.json();
        
        if (data.success) {
            displayDatabaseImages(data.images);
            document.getElementById('db-count').textContent = data.total;
        }
    } catch (error) {
        console.error('Error loading database:', error);
    }
}

// Display database images
function displayDatabaseImages(images) {
    const grid = document.getElementById('database-grid');
    grid.innerHTML = '';
    
    images.forEach(image => {
        const item = document.createElement('div');
        item.className = 'result-item';
        
        item.innerHTML = `
            <img src="/database/${image.filename}" alt="${image.filename}">
            <div class="result-info">
                <h4>${image.filename}</h4>
            </div>
        `;
        
        grid.appendChild(item);
    });
}

// Load database on page load
window.addEventListener('load', () => {
    loadDatabaseImages();
    checkModel2Status();
});

// Check Model II status
async function checkModel2Status() {
    try {
        const response = await fetch('/api/model2/status');
        const data = await response.json();
        
        if (data.success) {
            const statusText = data.trained ? 
                '✓ Model II is trained and ready' : 
                '✗ Model II not trained yet';
            
            document.getElementById('model2-status').textContent = statusText;
            document.getElementById('model2-status').style.color = 
                data.trained ? '#27ae60' : '#e74c3c';
            
            // Enable/disable option
            const option = document.getElementById('model2-option');
            if (data.trained) {
                option.disabled = false;
            } else {
                option.disabled = true;
            }
            
            // Update Model II tab
            updateModel2Tab(data);
        }
    } catch (error) {
        console.error('Error checking Model II status:', error);
    }
}

// Update Model II tab with status
function updateModel2Tab(statusData) {
    const statusDiv = document.getElementById('model2-current-status');
    
    if (statusData.trained) {
        statusDiv.innerHTML = `
            <p style="color: #27ae60; font-weight: 600;">✓ Model II is trained and ready for use</p>
            <p>You can use class-based retrieval in the Image Search tab</p>
        `;
    } else {
        statusDiv.innerHTML = `
            <p style="color: #e74c3c; font-weight: 600;">✗ Model II not trained</p>
            <p>Train a model below to enable class-based retrieval</p>
        `;
    }
    
    // Show available models
    const modelsList = document.getElementById('available-models-list');
    if (statusData.available_models && statusData.available_models.length > 0) {
        let modelsHTML = '<ul>';
        statusData.available_models.forEach(model => {
            modelsHTML += `
                <li>
                    <strong>${model}</strong>
                    <button onclick="loadModel2('models/${model}')" style="margin-left: 10px;">Load</button>
                </li>
            `;
        });
        modelsHTML += '</ul>';
        modelsList.innerHTML = modelsHTML;
    } else {
        modelsList.innerHTML = '<p>No pre-trained models available</p>';
    }
}

// Train Model II
async function trainModel2() {
    const architecture = document.getElementById('train-architecture').value;
    const epochs = parseInt(document.getElementById('train-epochs').value);
    const batchSize = parseInt(document.getElementById('train-batch').value);
    const learningRate = parseFloat(document.getElementById('train-lr').value);
    
    if (!confirm(`Train Model II with ${architecture} for ${epochs} epochs?\nThis may take several minutes.`)) {
        return;
    }
    
    const progressDiv = document.getElementById('training-progress');
    const progressInfo = document.getElementById('progress-info');
    
    progressDiv.style.display = 'block';
    progressInfo.innerHTML = '<p>Preparing dataset...</p>';
    
    try {
        // Call backend to prepare dataset for training
        const prepResponse = await fetch('/api/model2/prepare-dataset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                source_dir: 'datasets/demo_dataset',
                target_size: [224, 224],
                split_ratio: 0.8
            })
        });

        const prepData = await prepResponse.json();

        if (!prepResponse.ok || !prepData.train_dir) {
            progressInfo.innerHTML += `<p style="color: #e74c3c;">✗ Dataset preparation failed: ${prepData.error || 'Unknown error'}</p>`;
            alert('Dataset preparation failed: ' + (prepData.error || 'Unknown error'));
            return;
        }

        progressInfo.innerHTML += '<p>✓ Dataset prepared</p><p>Starting training...</p>';

        const trainDir = prepData.train_dir;

        const response = await fetch('/api/model2/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                train_dir: trainDir,
                architecture: architecture,
                epochs: epochs,
                batch_size: batchSize,
                learning_rate: learningRate
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            progressInfo.innerHTML += `
                <p style="color: #27ae60;">✓ Training completed!</p>
                <p>Model saved to: ${data.model_path}</p>
                <p>Number of classes: ${data.num_classes}</p>
            `;
            
            if (data.history && data.history.accuracy) {
                const finalAcc = data.history.accuracy[data.history.accuracy.length - 1];
                progressInfo.innerHTML += `<p>Final accuracy: ${(finalAcc * 100).toFixed(2)}%</p>`;
            }
            
            // Refresh status
            setTimeout(() => {
                checkModel2Status();
                alert('Model II training completed! You can now use class-based retrieval.');
            }, 1000);
        } else {
            progressInfo.innerHTML += `<p style="color: #e74c3c;">✗ Training failed: ${data.error}</p>`;
            alert('Training failed: ' + data.error);
        }
    } catch (error) {
        console.error('Training error:', error);
        progressInfo.innerHTML += `<p style="color: #e74c3c;">✗ Error: ${error.message}</p>`;
        alert('Training error: ' + error.message);
    }
}

// Load pre-trained Model II
async function loadModel2(modelPath) {
    try {
        const response = await fetch('/api/model2/load', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model_path: modelPath
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            alert('Model II loaded successfully!');
            checkModel2Status();
        } else {
            alert('Error loading model: ' + data.error);
        }
    } catch (error) {
        console.error('Load error:', error);
        alert('Error loading model: ' + error.message);
    }
}