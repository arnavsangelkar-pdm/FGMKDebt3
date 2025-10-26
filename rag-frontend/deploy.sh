#!/bin/bash

# Deployment script for Debt Agreement Analysis Frontend
# This script helps prepare and deploy the frontend to Render

echo "🚀 Preparing Debt Agreement Analysis Frontend for deployment..."

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Error: Please run this script from the frontend directory"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
npm install

# Run type check
echo "🔍 Running type check..."
npm run type-check

# Run linting
echo "🧹 Running linter..."
npm run lint

# Build the project
echo "🏗️ Building project..."
npm run build

# Check if build was successful
if [ ! -d "build" ]; then
    echo "❌ Error: Build failed. Please check the errors above."
    exit 1
fi

echo "✅ Build completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Commit your changes: git add . && git commit -m 'Deploy to Render'"
echo "2. Push to repository: git push origin main"
echo "3. Go to https://dashboard.render.com"
echo "4. Create a new Static Site"
echo "5. Connect your repository"
echo "6. Set environment variables:"
echo "   - REACT_APP_API_URL=https://your-backend-api.onrender.com"
echo "   - REACT_APP_APP_NAME=Debt Agreement Analysis"
echo "7. Deploy!"
echo ""
echo "📖 For detailed instructions, see DEPLOYMENT.md"
