const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

function installRequirements() {
    console.log("📦 Installing required packages...");
    try {
        // Check if requirements.txt exists
        if (fs.existsSync('requirements.txt')) {
            console.log("Found requirements.txt, installing Python packages...");
            execSync('python3 -m pip install -r requirements.txt', { stdio: 'inherit' });
            console.log("✅ Python packages installed successfully!");
        } else {
            console.log("⚠️  requirements.txt not found, skipping Python package installation");
        }
    } catch (error) {
        console.error("❌ Error installing Python packages:", error.message);
        console.log("💡 You may need to install packages manually later");
    }
}

function setupEnvironment() {
    console.log("🔧 Setting up environment file...");
    const envFile = path.join(__dirname, '.env');
    
    if (!fs.existsSync(envFile)) {
        console.log("📝 Creating .env file...");
        const envContent = `# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Perplexity Configuration  
PERPLEXITY_API_KEY=your_perplexity_api_key_here

# Optional: Other providers
ANTHROPIC_API_KEY=your_anthropic_key_here
`;
        fs.writeFileSync(envFile, envContent);
        console.log("✅ .env file created successfully!");
        console.log("📝 Please edit .env file with your actual API keys");
    } else {
        console.log("✅ .env file already exists");
    }
}

function checkPythonInstallation() {
    console.log("🐍 Checking Python installation...");
    try {
        const pythonVersion = execSync('python3 --version', { encoding: 'utf8' });
        console.log(`✅ Python found: ${pythonVersion.trim()}`);
        return true;
    } catch (error) {
        console.log("⚠️  Python3 not found, trying python...");
        try {
            const pythonVersion = execSync('python --version', { encoding: 'utf8' });
            console.log(`✅ Python found: ${pythonVersion.trim()}`);
            return true;
        } catch (error2) {
            console.error("❌ Python not found. Please install Python to use this application.");
            return false;
        }
    }
}

function main() {
    console.log("🚀 Setting up Children's Drawing Analysis System...");
    
    const pythonAvailable = checkPythonInstallation();
    
    if (pythonAvailable) {
        installRequirements();
    }
    
    setupEnvironment();
    
    console.log("\n✅ Setup complete!");
    console.log("\n📋 Next steps:");
    console.log("1. Edit .env file with your API keys");
    if (pythonAvailable) {
        console.log("2. Run: streamlit run app.py");
    } else {
        console.log("2. Install Python and then run: streamlit run app.py");
    }
}

if (require.main === module) {
    main();
}

module.exports = { main, setupEnvironment, installRequirements, checkPythonInstallation };