# Deployment Guide for Debt Agreement Analysis Frontend

This guide will help you deploy the Debt Agreement Analysis frontend to Render.

## Prerequisites

1. A Render account (free tier available)
2. Your backend API deployed and accessible
3. Git repository with your code

## Step 1: Prepare Your Repository

1. **Commit all changes to your repository:**
   ```bash
   git add .
   git commit -m "Prepare for Render deployment"
   git push origin main
   ```

2. **Ensure your backend API is deployed and accessible** (you'll need the URL for the next step)

## Step 2: Deploy to Render

### Option A: Deploy via Render Dashboard

1. **Go to [Render Dashboard](https://dashboard.render.com)**
2. **Click "New +" and select "Static Site"**
3. **Connect your repository:**
   - Choose your Git provider (GitHub, GitLab, etc.)
   - Select your repository
   - Choose the branch (usually `main`)

4. **Configure the deployment:**
   - **Name:** `debt-agreement-analysis-frontend`
   - **Build Command:** `npm install && npm run build`
   - **Publish Directory:** `build`
   - **Node Version:** `18` (or latest LTS)

5. **Add Environment Variables:**
   - Click "Advanced" → "Environment Variables"
   - Add: `REACT_APP_API_URL` = `https://your-backend-api.onrender.com`
   - Add: `REACT_APP_APP_NAME` = `Debt Agreement Analysis`

6. **Click "Create Static Site"**

### Option B: Deploy via render.yaml (Recommended)

1. **The render.yaml file is already configured in your repository**
2. **Go to [Render Dashboard](https://dashboard.render.com)**
3. **Click "New +" and select "Blueprint"**
4. **Connect your repository and select the branch**
5. **Render will automatically detect the render.yaml file and configure the deployment**

## Step 3: Configure Your Backend API URL

1. **After deployment, go to your service settings**
2. **Navigate to Environment Variables**
3. **Update `REACT_APP_API_URL` to point to your actual backend API URL**
4. **Redeploy the service**

## Step 4: Custom Domain (Optional)

1. **Go to your service settings**
2. **Click "Custom Domains"**
3. **Add your domain and follow the DNS configuration instructions**

## Step 5: Verify Deployment

1. **Visit your deployed URL**
2. **Test the upload functionality**
3. **Test the query functionality**
4. **Ensure all features work correctly**

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `REACT_APP_API_URL` | Backend API URL | `https://your-backend-api.onrender.com` |
| `REACT_APP_APP_NAME` | Application name | `Debt Agreement Analysis` |
| `NODE_ENV` | Environment | `production` |

## Troubleshooting

### Build Failures
- Check the build logs in Render dashboard
- Ensure all dependencies are in package.json
- Verify Node.js version compatibility

### API Connection Issues
- Verify the backend API URL is correct
- Check CORS settings on your backend
- Ensure the backend is accessible from the internet

### Static File Issues
- Verify the build directory is set to `build`
- Check that all static assets are included in the build

## Performance Optimization

1. **Enable gzip compression** (handled by Render)
2. **Set up CDN** (Render provides this automatically)
3. **Optimize images** before uploading
4. **Use production build** (handled automatically)

## Security Considerations

1. **Environment variables are secure** on Render
2. **HTTPS is enabled by default**
3. **Security headers are configured** in render.yaml
4. **CORS is handled** by your backend API

## Monitoring

1. **Check deployment logs** in Render dashboard
2. **Monitor performance** through Render metrics
3. **Set up alerts** for deployment failures

## Cost

- **Free tier:** 750 hours/month
- **Paid plans:** Start at $7/month for more resources
- **Custom domains:** Free on paid plans

## Support

- **Render Documentation:** https://render.com/docs
- **Render Support:** Available through dashboard
- **Community:** Render Discord and forums

---

Your Debt Agreement Analysis frontend should now be live and accessible to your clients!
