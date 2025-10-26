#!/bin/bash

# Build script for Render deployment
echo "=== RENDER BUILD SCRIPT ==="
echo "Current directory: $(pwd)"
echo "Listing contents:"
ls -la

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r rag_app/requirements.txt

# Find frontend directory
echo "Looking for frontend directory..."
if [ -d "rag-frontend" ]; then
    FRONTEND_DIR="rag-frontend"
    echo "Found rag-frontend directory"
elif [ -d "src/rag-frontend" ]; then
    FRONTEND_DIR="src/rag-frontend"
    echo "Found src/rag-frontend directory"
else
    echo "ERROR: Could not find frontend directory"
    echo "Available directories:"
    find . -type d -name "*frontend*" -o -name "*react*" -o -name "*src*"
    exit 1
fi

echo "Using frontend directory: $FRONTEND_DIR"
cd $FRONTEND_DIR

echo "Current directory: $(pwd)"
echo "Listing frontend contents:"
ls -la

# Check for required files
echo "Checking for required files..."
if [ ! -f "package.json" ]; then
    echo "ERROR: package.json not found"
    exit 1
fi

if [ ! -d "public" ]; then
    echo "ERROR: public directory not found"
    exit 1
fi

if [ ! -f "public/index.html" ]; then
    echo "ERROR: public/index.html not found"
    exit 1
fi

echo "All required files found"

# Install Node dependencies
echo "Installing Node dependencies..."
npm install

# Build frontend
echo "Building frontend..."
npm run build

# Check if build was successful
if [ ! -d "build" ]; then
    echo "ERROR: Build failed - build directory not created"
    exit 1
fi

echo "Build successful, checking build contents:"
ls -la build/

# Create static directory in backend
echo "Creating static directory in backend..."
mkdir -p ../rag_app/static

# Copy build files
echo "Copying build files to backend..."
cp -r build/* ../rag_app/static/

echo "Verifying copy:"
ls -la ../rag_app/static/

echo "=== BUILD COMPLETE ==="