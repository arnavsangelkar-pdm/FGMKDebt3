# DocuMind AI - Frontend

A modern, professional React frontend for the DocuMind AI document analysis platform. This application provides an intuitive interface for uploading PDF documents and querying them using AI-powered natural language processing.

## ✨ Features

- **Modern Design System**: Clean, professional UI with consistent typography and spacing
- **Document Upload**: Drag-and-drop PDF upload with progress indicators
- **AI-Powered Queries**: Natural language question answering with source citations
- **Responsive Design**: Optimized for desktop, tablet, and mobile devices
- **Real-time Feedback**: Loading states, error handling, and success notifications
- **Accessibility**: WCAG compliant with keyboard navigation and screen reader support

## 🚀 Quick Start

### Prerequisites

- Node.js 16+ and npm
- Backend API running on `http://localhost:8000`

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm start
```

The application will be available at `http://localhost:3000`.

### Production Build

```bash
# Create production build
npm run build

# Serve the build (requires a static server)
npx serve -s build
```

## 🎨 Design System

The application uses a modern design system with:

- **Colors**: Professional blue palette with semantic color tokens
- **Typography**: Inter font family for optimal readability
- **Spacing**: Consistent 8px grid system
- **Components**: Reusable UI components with consistent styling
- **Animations**: Subtle transitions and micro-interactions

## 📱 Responsive Breakpoints

- **Mobile**: < 480px
- **Tablet**: 480px - 768px
- **Desktop**: > 768px

## 🔧 Configuration

The application connects to the backend API at `http://localhost:8000`. To change this:

1. Update the API URLs in the component files
2. Or set up environment variables for different environments

## 🏗️ Architecture

```
src/
├── components/          # Reusable UI components
│   ├── DocumentUpload.tsx
│   └── DocumentQuery.tsx
├── App.tsx             # Main application component
├── App.css             # Global styles and design system
└── index.css           # Base styles and font imports
```

## 🎯 Key Components

### DocumentUpload
- Drag-and-drop file upload
- File validation (PDF only, 250MB max)
- Progress indicators and status messages
- Feature highlights for user guidance

### DocumentQuery
- Natural language question input
- Real-time query processing
- Results display with citations
- Clear and retry functionality

## 🚀 Deployment

### Build for Production

```bash
npm run build
```

### Deploy to Static Hosting

The build folder can be deployed to any static hosting service:

- **Vercel**: `vercel --prod`
- **Netlify**: Drag and drop the build folder
- **AWS S3**: Upload build folder contents
- **GitHub Pages**: Use GitHub Actions

### Environment Variables

Create a `.env` file for environment-specific configuration:

```env
REACT_APP_API_URL=http://localhost:8000
REACT_APP_APP_NAME=DocuMind AI
```

## 🔒 Security Considerations

- File upload validation on both client and server
- XSS protection through React's built-in sanitization
- HTTPS recommended for production deployments
- Content Security Policy headers for additional protection

## 📊 Performance

- Optimized bundle size with code splitting
- Lazy loading for better initial load times
- Efficient re-renders with React best practices
- Image optimization and compression

## 🧪 Testing

```bash
# Run tests
npm test

# Run tests with coverage
npm test -- --coverage
```

## 📝 License

This project is part of the DocuMind AI platform. All rights reserved.

## 🤝 Contributing

1. Follow the established design system
2. Maintain responsive design principles
3. Ensure accessibility compliance
4. Write clean, maintainable code
5. Test across different devices and browsers

---

Built with ❤️ using React, TypeScript, and modern web technologies.