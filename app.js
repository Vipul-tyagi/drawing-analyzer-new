// Children's Drawing Analysis System - Frontend JavaScript
class DrawingAnalysisApp {
    constructor() {
        this.selectedFile = null;
        this.analysisResults = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.checkSystemStatus();
        this.setupDragAndDrop();
    }

    setupEventListeners() {
        // File input
        document.getElementById('fileInput').addEventListener('change', (e) => {
            this.handleFileSelect(e.target.files[0]);
        });

        // Generate video checkbox
        document.getElementById('generateVideo').addEventListener('change', (e) => {
            const videoStyleGroup = document.getElementById('videoStyleGroup');
            videoStyleGroup.style.display = e.target.checked ? 'block' : 'none';
        });

        // Analyze button
        document.getElementById('analyzeBtn').addEventListener('click', () => {
            this.startAnalysis();
        });
    }

    setupDragAndDrop() {
        const uploadArea = document.querySelector('.upload-area');
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileSelect(files[0]);
            }
        });
    }

    handleFileSelect(file) {
        if (!file) return;

        // Validate file type
        const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/tiff'];
        if (!validTypes.includes(file.type)) {
            alert('Please select a valid image file (PNG, JPG, JPEG, BMP, TIFF)');
            return;
        }

        // Validate file size (10MB max)
        if (file.size > 10 * 1024 * 1024) {
            alert('File size must be less than 10MB');
            return;
        }

        this.selectedFile = file;
        this.showPreview(file);
        document.getElementById('analyzeBtn').disabled = false;
    }

    showPreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            document.getElementById('previewImage').src = e.target.result;
            document.getElementById('fileName').textContent = file.name;
            document.getElementById('previewSection').style.display = 'block';
        };
        reader.readAsDataURL(file);
    }

    async checkSystemStatus() {
        // Check if Python backend is available
        try {
            const response = await fetch('/api/status');
            if (response.ok) {
                document.getElementById('pythonStatus').className = 'status-available';
                document.getElementById('pythonStatus').textContent = '✅';
                
                const status = await response.json();
                if (status.ai_components) {
                    document.getElementById('aiStatus').className = 'status-available';
                    document.getElementById('aiStatus').textContent = '✅';
                }
            }
        } catch (error) {
            console.log('Python backend not available, using demo mode');
            this.setupDemoMode();
        }
    }

    setupDemoMode() {
        // If Python backend is not available, show demo functionality
        console.log('Running in demo mode');
    }

    async startAnalysis() {
        if (!this.selectedFile) {
            alert('Please select an image first');
            return;
        }

        // Show loading
        document.getElementById('loadingSection').style.display = 'block';
        document.getElementById('analyzeBtn').disabled = true;

        // Get form data
        const formData = new FormData();
        formData.append('image', this.selectedFile);
        formData.append('childAge', document.getElementById('childAge').value);
        formData.append('drawingContext', document.getElementById('drawingContext').value);
        formData.append('analysisType', document.getElementById('analysisType').value);
        formData.append('generatePdf', document.getElementById('generatePdf').checked);
        formData.append('generateVideo', document.getElementById('generateVideo').checked);
        formData.append('videoStyle', document.getElementById('videoStyle').value);

        try {
            // Try to send to Python backend
            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const results = await response.json();
                this.displayResults(results);
            } else {
                throw new Error('Backend analysis failed');
            }
        } catch (error) {
            console.log('Backend not available, using demo analysis');
            this.runDemoAnalysis();
        }

        // Hide loading
        document.getElementById('loadingSection').style.display = 'none';
        document.getElementById('analyzeBtn').disabled = false;
    }

    runDemoAnalysis() {
        // Simulate analysis with demo data
        const childAge = document.getElementById('childAge').value;
        const drawingContext = document.getElementById('drawingContext').value;

        const demoResults = {
            input_info: {
                child_age: parseInt(childAge),
                age_group: this.getAgeGroup(parseInt(childAge)),
                drawing_context: drawingContext
            },
            traditional_analysis: {
                blip_description: `A colorful ${drawingContext.toLowerCase()} by a ${childAge}-year-old child`,
                color_analysis: {
                    dominant_color: 'Mixed colors',
                    color_diversity: 8,
                    brightness_level: 150,
                    color_richness: 'Rich'
                },
                shape_analysis: {
                    total_shapes: 5,
                    complexity_level: 'Medium',
                    drawing_coverage: 0.35,
                    detail_level: 'Good'
                },
                spatial_analysis: {
                    spatial_balance: 'Balanced',
                    drawing_style: 'Center-focused'
                },
                emotional_indicators: {
                    overall_mood: 'positive',
                    tone: 'bright_positive',
                    positive_words_found: 2,
                    negative_words_found: 0
                },
                developmental_assessment: {
                    level: 'age_appropriate',
                    actual_shapes: 5,
                    expected_shapes: { min_shapes: 3, max_shapes: 8 }
                }
            },
            confidence_scores: {
                traditional_ml: 0.85,
                llm_average: 0.0,
                overall: 0.85
            },
            summary: {
                ai_description: `A colorful ${drawingContext.toLowerCase()} by a ${childAge}-year-old child`,
                analysis_quality: 'Demo Mode',
                total_analyses: 1,
                available_providers: []
            },
            demo_mode: true
        };

        this.displayResults(demoResults);
    }

    getAgeGroup(age) {
        if (age < 4) return "Toddler (2-3 years)";
        if (age < 7) return "Preschool (4-6 years)";
        if (age < 12) return "School Age (7-11 years)";
        return "Adolescent (12+ years)";
    }

    displayResults(results) {
        this.analysisResults = results;
        
        // Show results section
        document.getElementById('resultsSection').style.display = 'block';
        
        // Populate overview
        this.populateOverview(results);
        
        // Populate detailed analysis
        this.populateDetailed(results);
        
        // Populate recommendations
        this.populateRecommendations(results);
        
        // Populate reports
        this.populateReports(results);
        
        // Scroll to results
        document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
    }

    populateOverview(results) {
        const overviewContent = document.getElementById('overviewContent');
        
        const childAge = results.input_info?.child_age || 'Unknown';
        const confidence = results.confidence_scores?.overall || 0;
        const analysisCount = (results.llm_analyses?.length || 0) + 1;
        const quality = results.summary?.analysis_quality || 'Good';
        
        overviewContent.innerHTML = `
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px;">
                <div class="metric-card">
                    <div class="metric-value">👶</div>
                    <div class="metric-label">Child Age</div>
                    <div style="font-size: 1.2rem; font-weight: bold; margin-top: 5px;">${childAge} years</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">🎯</div>
                    <div class="metric-label">Confidence</div>
                    <div style="font-size: 1.2rem; font-weight: bold; margin-top: 5px;">${(confidence * 100).toFixed(1)}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">🤖</div>
                    <div class="metric-label">AI Models</div>
                    <div style="font-size: 1.2rem; font-weight: bold; margin-top: 5px;">${analysisCount} analyses</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">⭐</div>
                    <div class="metric-label">Quality</div>
                    <div style="font-size: 1.2rem; font-weight: bold; margin-top: 5px;">${quality}</div>
                </div>
            </div>
            
            ${results.demo_mode ? `
                <div class="analysis-card" style="background: #fff3cd; border-left-color: #ffc107;">
                    <h3>🧪 Demo Mode</h3>
                    <p>This is a demonstration using simulated analysis results. For full AI-powered analysis, please set up the Python backend components.</p>
                </div>
            ` : ''}
            
            <div class="analysis-card">
                <h3>🤖 AI Description</h3>
                <p>"${results.traditional_analysis?.blip_description || 'A child\'s drawing'}"</p>
            </div>
            
            <div class="analysis-card">
                <h3>📈 Developmental Level</h3>
                <p style="color: green; font-weight: bold; font-size: 1.2em;">
                    ${(results.traditional_analysis?.developmental_assessment?.level || 'unknown').replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </p>
            </div>
            
            <div class="analysis-card">
                <h3>😊 Emotional Mood</h3>
                <p style="color: ${this.getMoodColor(results.traditional_analysis?.emotional_indicators?.overall_mood)}; font-weight: bold; font-size: 1.2em;">
                    ${(results.traditional_analysis?.emotional_indicators?.overall_mood || 'neutral').replace(/\b\w/g, l => l.toUpperCase())}
                </p>
            </div>
        `;
    }

    getMoodColor(mood) {
        switch(mood) {
            case 'positive': return 'green';
            case 'concerning': return 'red';
            default: return 'blue';
        }
    }

    populateDetailed(results) {
        const detailedContent = document.getElementById('detailedContent');
        
        const traditional = results.traditional_analysis || {};
        const colorAnalysis = traditional.color_analysis || {};
        const shapeAnalysis = traditional.shape_analysis || {};
        const spatialAnalysis = traditional.spatial_analysis || {};
        
        detailedContent.innerHTML = `
            <div class="analysis-card">
                <h3>🎨 Color Analysis</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px;">
                    <div>
                        <strong>Dominant Color:</strong><br>
                        ${colorAnalysis.dominant_color || 'Unknown'}
                    </div>
                    <div>
                        <strong>Color Diversity:</strong><br>
                        ${colorAnalysis.color_diversity || 0}
                    </div>
                    <div>
                        <strong>Brightness:</strong><br>
                        ${colorAnalysis.brightness_level || 0}/255
                    </div>
                    <div>
                        <strong>Richness:</strong><br>
                        ${colorAnalysis.color_richness || 'Unknown'}
                    </div>
                </div>
            </div>
            
            <div class="analysis-card">
                <h3>🔷 Shape Analysis</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px;">
                    <div>
                        <strong>Total Shapes:</strong><br>
                        ${shapeAnalysis.total_shapes || 0}
                    </div>
                    <div>
                        <strong>Complexity:</strong><br>
                        ${shapeAnalysis.complexity_level || 'Unknown'}
                    </div>
                    <div>
                        <strong>Coverage:</strong><br>
                        ${((shapeAnalysis.drawing_coverage || 0) * 100).toFixed(1)}%
                    </div>
                    <div>
                        <strong>Detail Level:</strong><br>
                        ${shapeAnalysis.detail_level || 'Unknown'}
                    </div>
                </div>
            </div>
            
            <div class="analysis-card">
                <h3>📐 Spatial Analysis</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px;">
                    <div>
                        <strong>Balance:</strong><br>
                        ${spatialAnalysis.spatial_balance || 'Unknown'}
                    </div>
                    <div>
                        <strong>Style:</strong><br>
                        ${spatialAnalysis.drawing_style || 'Unknown'}
                    </div>
                </div>
            </div>
            
            ${results.llm_analyses && results.llm_analyses.length > 0 ? `
                <div class="analysis-card">
                    <h3>🤖 AI Expert Analyses</h3>
                    ${results.llm_analyses.map(analysis => `
                        <div style="margin-bottom: 20px; padding: 15px; background: white; border-radius: 8px;">
                            <strong>${analysis.provider.toUpperCase()} Analysis:</strong>
                            <em style="color: #666;">(Confidence: ${(analysis.confidence * 100).toFixed(1)}%)</em>
                            <p style="margin-top: 10px;">${analysis.analysis}</p>
                        </div>
                    `).join('')}
                </div>
            ` : ''}
        `;
    }

    populateRecommendations(results) {
        const recommendationsContent = document.getElementById('recommendationsContent');
        
        // Generate age-appropriate recommendations
        const childAge = results.input_info?.child_age || 6;
        const devLevel = results.traditional_analysis?.developmental_assessment?.level || 'age_appropriate';
        const mood = results.traditional_analysis?.emotional_indicators?.overall_mood || 'neutral';
        
        const recommendations = this.generateRecommendations(childAge, devLevel, mood);
        
        recommendationsContent.innerHTML = `
            <div class="analysis-card">
                <h3>🚨 Immediate Actions</h3>
                ${recommendations.immediate.map(action => `
                    <div class="recommendation-item">• ${action}</div>
                `).join('')}
            </div>
            
            <div class="analysis-card">
                <h3>📅 Short-term Goals (1-3 months)</h3>
                ${recommendations.shortTerm.map(goal => `
                    <div class="recommendation-item">• ${goal}</div>
                `).join('')}
            </div>
            
            <div class="analysis-card">
                <h3>🎯 Long-term Development (3-12 months)</h3>
                ${recommendations.longTerm.map(goal => `
                    <div class="recommendation-item">• ${goal}</div>
                `).join('')}
            </div>
            
            <div class="analysis-card">
                <h3>🎨 Recommended Materials & Activities</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                    <div>
                        <strong>Materials:</strong>
                        <ul style="margin-top: 10px;">
                            ${recommendations.materials.map(material => `<li>${material}</li>`).join('')}
                        </ul>
                    </div>
                    <div>
                        <strong>Activities:</strong>
                        <ul style="margin-top: 10px;">
                            ${recommendations.activities.map(activity => `<li>${activity}</li>`).join('')}
                        </ul>
                    </div>
                </div>
            </div>
            
            ${recommendations.warnings.length > 0 ? `
                <div class="analysis-card">
                    <h3>⚠️ When to Seek Professional Help</h3>
                    ${recommendations.warnings.map(warning => `
                        <div class="recommendation-item warning-item">• ${warning}</div>
                    `).join('')}
                </div>
            ` : ''}
        `;
    }

    generateRecommendations(childAge, devLevel, mood) {
        const recommendations = {
            immediate: [],
            shortTerm: [],
            longTerm: [],
            materials: [],
            activities: [],
            warnings: []
        };

        // Age-specific recommendations
        if (childAge < 4) {
            recommendations.immediate.push("Encourage more drawing time - great for developing fine motor skills!");
            recommendations.immediate.push("Try finger paints or chunky crayons for easier grip");
            recommendations.materials.push("Chunky crayons", "Finger paints", "Large paper");
            recommendations.activities.push("Finger painting", "Large scribbling motions", "Color exploration");
        } else if (childAge < 7) {
            recommendations.immediate.push("Ask the child to tell stories about their drawings");
            recommendations.immediate.push("Provide various art materials to explore creativity");
            recommendations.materials.push("Crayons", "Markers", "Colored pencils", "Stickers");
            recommendations.activities.push("Story illustration", "Art games", "Creative play");
        } else if (childAge < 12) {
            recommendations.immediate.push("Encourage drawing from observation (flowers, pets, etc.)");
            recommendations.immediate.push("Consider art classes if the child shows strong interest");
            recommendations.materials.push("Sketch pads", "Watercolors", "Drawing pencils", "Erasers");
            recommendations.activities.push("Nature drawing", "Portrait practice", "Art challenges");
        } else {
            recommendations.immediate.push("Support artistic expression as a healthy outlet");
            recommendations.immediate.push("Discuss the meaning behind their artwork");
            recommendations.materials.push("Professional art supplies", "Digital art tools", "Canvas");
            recommendations.activities.push("Advanced techniques", "Art portfolio development", "Creative projects");
        }

        // Development level specific
        if (devLevel === 'below_expected') {
            recommendations.immediate.push("Increase drawing and creative activities in daily routine");
            recommendations.immediate.push("Celebrate all artistic attempts to build confidence");
            recommendations.shortTerm.push("Focus on basic shape recognition and drawing");
            recommendations.warnings.push("Consider developmental assessment if skills don't improve");
        } else if (devLevel === 'above_expected') {
            recommendations.immediate.push("Provide more challenging artistic activities");
            recommendations.immediate.push("Consider enrolling in age-appropriate art classes");
            recommendations.shortTerm.push("Explore advanced artistic techniques");
        }

        // Mood specific
        if (mood === 'concerning') {
            recommendations.immediate.push("⚠️ Consider talking with the child about their feelings");
            recommendations.warnings.push("If concerns persist, consult with a counselor or teacher");
            recommendations.warnings.push("Monitor for changes in behavior or mood");
        } else if (mood === 'positive') {
            recommendations.immediate.push("✨ Great emotional expression! Keep encouraging creativity");
        }

        // General recommendations
        recommendations.shortTerm.push("Maintain regular art time in daily routine");
        recommendations.shortTerm.push("Create a dedicated art space in your home");
        recommendations.longTerm.push("Document artistic progress with photos");
        recommendations.longTerm.push("Encourage exploration of different artistic mediums");

        return recommendations;
    }

    populateReports(results) {
        const reportsContent = document.getElementById('reportsContent');
        
        reportsContent.innerHTML = `
            <div class="analysis-card">
                <h3>📄 Available Reports</h3>
                
                ${results.demo_mode ? `
                    <div class="recommendation-item warning-item">
                        <strong>Demo Mode:</strong> PDF and video generation require the full Python backend setup.
                    </div>
                ` : ''}
                
                <div class="download-section">
                    <button class="download-btn" onclick="app.exportJSON()">
                        📊 Download Analysis Data (JSON)
                    </button>
                    
                    <button class="download-btn" onclick="app.generateReport()" ${results.demo_mode ? 'disabled' : ''}>
                        📄 Generate PDF Report
                    </button>
                    
                    <button class="download-btn" onclick="app.generateVideo()" ${results.demo_mode ? 'disabled' : ''}>
                        🎬 Generate Memory Video
                    </button>
                </div>
                
                <div style="margin-top: 30px;">
                    <h4>🎨 Available Animation Styles:</h4>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 15px;">
                        <div>
                            <strong>• Intelligent:</strong> AI-powered component animation<br>
                            <strong>• Elements:</strong> Individual element animations<br>
                            <strong>• Particle:</strong> Particle effect animations
                        </div>
                        <div>
                            <strong>• Floating:</strong> Floating and orbiting effects<br>
                            <strong>• Animated:</strong> Standard animation effects
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="analysis-card">
                <h3>🔧 Raw Analysis Results (Debug)</h3>
                <details>
                    <summary>Click to view raw data</summary>
                    <pre style="background: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto; margin-top: 10px;">
${JSON.stringify(results, null, 2)}
                    </pre>
                </details>
            </div>
        `;
    }

    exportJSON() {
        if (!this.analysisResults) return;
        
        const dataStr = JSON.stringify(this.analysisResults, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = `drawing_analysis_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`;
        link.click();
    }

    async generateReport() {
        if (this.analysisResults?.demo_mode) {
            alert('PDF generation requires the full Python backend setup.');
            return;
        }
        
        try {
            const response = await fetch('/api/generate-pdf', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(this.analysisResults)
            });
            
            if (response.ok) {
                const blob = await response.blob();
                const link = document.createElement('a');
                link.href = URL.createObjectURL(blob);
                link.download = `drawing_analysis_report_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.pdf`;
                link.click();
            } else {
                throw new Error('PDF generation failed');
            }
        } catch (error) {
            alert('PDF generation is not available. Please set up the Python backend.');
        }
    }

    async generateVideo() {
        if (this.analysisResults?.demo_mode) {
            alert('Video generation requires the full Python backend setup.');
            return;
        }
        
        try {
            const response = await fetch('/api/generate-video', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    results: this.analysisResults,
                    style: document.getElementById('videoStyle').value
                })
            });
            
            if (response.ok) {
                const blob = await response.blob();
                const link = document.createElement('a');
                link.href = URL.createObjectURL(blob);
                link.download = `memory_video_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.mp4`;
                link.click();
            } else {
                throw new Error('Video generation failed');
            }
        } catch (error) {
            alert('Video generation is not available. Please set up the Python backend.');
        }
    }
}

function showTab(tabName) {
    // Hide all tab contents
    const tabContents = document.querySelectorAll('.tab-content');
    tabContents.forEach(content => content.classList.remove('active'));
    
    // Remove active class from all tab buttons
    const tabButtons = document.querySelectorAll('.tab-button');
    tabButtons.forEach(button => button.classList.remove('active'));
    
    // Show selected tab content
    document.getElementById(tabName).classList.add('active');
    
    // Add active class to clicked button
    event.target.classList.add('active');
}

// Initialize the app when the page loads
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new DrawingAnalysisApp();
});