import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import ErrorBoundaryDebug from './ErrorBoundaryDebug';
import './index.css';

console.log('Main.tsx loading...');
const root = document.getElementById('root');
console.log('Root element:', root);

if (root) {
  console.log('Creating React root...');
  ReactDOM.createRoot(root).render(
    <React.StrictMode>
      <ErrorBoundaryDebug>
        <App />
      </ErrorBoundaryDebug>
    </React.StrictMode>
  );
  console.log('React app rendered');
  
  // Remove loading screen after React mounts
  setTimeout(() => {
    const loadingScreen = document.getElementById('loading-screen');
    if (loadingScreen) {
      console.log('Removing loading screen...');
      loadingScreen.style.display = 'none';
      loadingScreen.remove();
    } else {
      console.log('Loading screen not found');
    }
  }, 100);
  
  // Also try to remove it immediately
  const loadingScreen = document.getElementById('loading-screen');
  if (loadingScreen) {
    loadingScreen.style.opacity = '0.5';
  }
}