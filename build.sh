#!/bin/bash

# Build script for full-stack deployment
echo "🚀 Building Debt Agreement Analysis for deployment..."

# Build frontend
echo "📦 Building frontend..."
cd rag-frontend
npm install
npm run build

# Create static directory in backend
echo "📁 Setting up static files..."
cd ../rag_app
mkdir -p static

# Copy frontend build to backend static directory
echo "📋 Copying frontend build to backend..."
cp -r ../rag-frontend/build/* static/

echo "✅ Build completed successfully!"
echo "📋 Frontend files copied to rag_app/static/"
echo "🚀 Ready for deployment!"
