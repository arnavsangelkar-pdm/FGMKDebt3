#!/bin/bash

# Build script specifically for Render deployment
echo "🚀 Starting Render build process..."

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r rag_app/requirements.txt

# Navigate to frontend directory
echo "📁 Navigating to frontend directory..."
cd rag-frontend

# Install Node dependencies
echo "📦 Installing Node dependencies..."
npm install

# Build the frontend
echo "🏗️ Building frontend..."
npm run build

# Check if build was successful
if [ ! -d "build" ]; then
    echo "❌ Frontend build failed!"
    exit 1
fi

# Create static directory in backend
echo "📁 Creating static directory..."
mkdir -p ../rag_app/static

# Copy frontend build to backend
echo "📋 Copying frontend build to backend..."
cp -r build/* ../rag_app/static/

echo "✅ Build completed successfully!"
echo "📋 Frontend files copied to rag_app/static/"
