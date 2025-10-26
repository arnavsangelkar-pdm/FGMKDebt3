# 🚀 Render Deployment Checklist

## Pre-Deployment Checklist

- [ ] **Backend API is deployed and accessible**
  - [ ] Backend is running on Render or another service
  - [ ] API endpoints are working (`/ingest`, `/query`)
  - [ ] CORS is configured to allow frontend domain
  - [ ] You have the backend API URL

- [ ] **Code is ready**
  - [ ] All changes committed to git
  - [ ] Code pushed to repository (GitHub/GitLab)
  - [ ] No console errors in development
  - [ ] Build runs successfully locally (`npm run build`)

- [ ] **Environment variables prepared**
  - [ ] `REACT_APP_API_URL` = Your backend API URL
  - [ ] `REACT_APP_APP_NAME` = "Debt Agreement Analysis"

## Deployment Steps

### Option 1: Quick Deploy (Recommended)

1. **Run the deployment script:**
   ```bash
   ./deploy.sh
   ```

2. **Follow the script's instructions to deploy via Render dashboard**

### Option 2: Manual Deploy

1. **Go to [Render Dashboard](https://dashboard.render.com)**
2. **Click "New +" → "Static Site"**
3. **Connect your repository**
4. **Configure:**
   - Name: `debt-agreement-analysis-frontend`
   - Build Command: `npm install && npm run build`
   - Publish Directory: `build`
5. **Add environment variables**
6. **Deploy!**

## Post-Deployment Checklist

- [ ] **Frontend is accessible**
  - [ ] Site loads without errors
  - [ ] No console errors in browser
  - [ ] All pages render correctly

- [ ] **Functionality works**
  - [ ] Document upload works
  - [ ] Document query works
  - [ ] Error handling works
  - [ ] Loading states work

- [ ] **Performance is good**
  - [ ] Site loads quickly
  - [ ] Images and assets load
  - [ ] No broken links

## Troubleshooting

### Common Issues

1. **Build fails:**
   - Check Node.js version (use 18+)
   - Check build logs in Render dashboard
   - Ensure all dependencies are in package.json

2. **API connection fails:**
   - Verify `REACT_APP_API_URL` is correct
   - Check backend CORS settings
   - Ensure backend is accessible

3. **Static files not loading:**
   - Verify publish directory is set to `build`
   - Check that build completed successfully

### Getting Help

- Check Render logs in dashboard
- Review DEPLOYMENT.md for detailed instructions
- Check browser console for errors
- Verify environment variables are set correctly

## Success! 🎉

Once deployed, your Debt Agreement Analysis frontend will be live and ready for clients to use!
