# AgriLens Automated Deployment Helper for Windows

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "🌱 Welcome to the AgriLens Automated Deployment Helper! 🌱" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
Write-Host ""

# Step 1: Prompt user for their GitHub Repository URL
Write-Host "To deploy your app permanently with 100% stability, please create a new repository on GitHub."
Write-Host "1. Go to https://github.com/new"
Write-Host "2. Name it 'agrilens' (leave it Public, do NOT add a README or License)"
Write-Host "3. Click 'Create repository'"
Write-Host ""
$repoUrl = Read-Host "Paste your GitHub Repository URL here (e.g., https://github.com/your-username/agrilens.git)"

if ([string]::IsNullOrWhiteSpace($repoUrl)) {
    Write-Error "Repository URL cannot be empty. Please run this script again."
    exit
}

Write-Host ""
Write-Host "🚀 Initializing deployment..." -ForegroundColor Cyan

# Configure local git identity if not set globally
$gitUser = git config --global user.name
if ([string]::IsNullOrWhiteSpace($gitUser)) {
    git config user.name "AgriLens User"
    git config user.email "deploy@agrilens.com"
}

# Initialize Git Repository
if (-not (Test-Path .git)) {
    Write-Host "Initializing new Git repository..." -ForegroundColor Yellow
    git init
} else {
    Write-Host "Git repository already initialized." -ForegroundColor Yellow
}

# Add Remote Origin
Write-Host "Linking your local folder to GitHub remote..." -ForegroundColor Yellow
git remote remove origin 2>$null
git remote add origin $repoUrl

# Create .gitignore to avoid uploading virtual environments
Write-Host "Creating .gitignore file..." -ForegroundColor Yellow
$gitignoreContent = @"
venv/
.venv/
__pycache__/
*.pyc
.gemini/
scratch/
C:/
*.pdf
.system_generated/
"@
Set-Content -Path .gitignore -Value $gitignoreContent

# Stage and Commit files
Write-Host "Staging files and committing changes..." -ForegroundColor Yellow
git add .
git commit -m "Deploy AgriLens Premium Dashboard"

# Rename branch to main
git branch -M main

# Push to GitHub
Write-Host ""
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "PULSING CODE TO GITHUB... Please authorize git if prompted." -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""

git push -u origin main --force

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "==========================================================" -ForegroundColor Green
    Write-Host "🎉 SUCCESS! Your code has been pushed to GitHub perfectly! 🎉" -ForegroundColor Green
    Write-Host "==========================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Now let's launch your app live in the cloud:"
    Write-Host "1. Go to: https://share.streamlit.io/"
    Write-Host "2. Log in with your GitHub account."
    Write-Host "3. Click 'New app' (or 'Create app')."
    Write-Host "4. Select repository: your-username/agrilens"
    Write-Host "5. Main file path: streamlit_app.py"
    Write-Host "6. Click 'Deploy!'"
    Write-Host ""
    Write-Host "Your app will be live globally on a high-speed cloud server in under 2 minutes!" -ForegroundColor Green
} else {
    Write-Error "Failed to push code to GitHub. Please check your GitHub permissions, make sure you created the repo, and try again."
}

Write-Host ""
Read-Host "Press Enter to exit..."
