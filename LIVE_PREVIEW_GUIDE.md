# GoldenSignals Live Preview Guide

## Overview

The GoldenSignals Live Preview system allows you to see UI changes in real-time as you develop, similar to Android Studio or Xcode. This guide explains how to use the live preview features with your existing setup.

## Quick Start

Since you already have your frontend server running on port 3000, you can immediately start using the live preview:

### Option 1: Simple Browser Preview
1. Open `frontend-v2/scripts/live-preview.html` in your browser
2. This provides a dedicated preview environment with:
   - Device frame switching (Desktop/Tablet/Mobile)
   - Dark mode toggle
   - Grid overlay for alignment
   - Auto-refresh on file changes

### Option 2: Command Line
```bash
./start-live-preview.sh
```

### Option 3: VS Code/Cursor Integration
1. Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
2. Type "Tasks: Run Task"
3. Select "Start Live Preview"

## Features

### 🔄 Hot Module Replacement (HMR)
Your existing Vite setup already provides excellent HMR. The live preview enhances this with:
- Visual feedback when files change
- Preserved component state during updates
- Instant CSS updates

### 📱 Responsive Preview
Test your UI across different device sizes:
- **Desktop**: Full browser view
- **Tablet**: 768x1024px with device frame
- **Mobile**: 375x812px with device frame

### ⌨️ Keyboard Shortcuts
- `Cmd/Ctrl + R`: Refresh preview
- `Cmd/Ctrl + D`: Toggle dark mode
- `Cmd/Ctrl + G`: Toggle grid overlay

### 🔍 Component Inspector (DevTools)
In your main app, use these shortcuts:
- `Cmd/Ctrl + Shift + I`: Toggle component inspector
- `Cmd/Ctrl + Shift + S`: Toggle state visualization
- `Cmd/Ctrl + Shift + D`: Toggle all DevTools

## VS Code/Cursor Extensions

For the best experience, install these recommended extensions:

1. **Live Server** - Launch a local development server with live reload
   ```
   ext install ritwickdey.liveserver
   ```

2. **Browser Preview** - Browser preview inside VS Code
   ```
   ext install auchenberg.vscode-browser-preview
   ```

## Split Screen Setup

For optimal development workflow:

1. **Left Panel**: Your code editor
2. **Right Panel**: Live preview (either in browser or VS Code preview)
3. **Bottom Panel**: Terminal for logs

### In VS Code/Cursor:
1. Open your component file
2. Press `Cmd+\` (Mac) or `Ctrl+\` (Windows/Linux) to split editor
3. In the right panel, open the Browser Preview extension
4. Navigate to `http://localhost:3000`

## Advanced Features

### WebSocket Live Updates
The dev tools automatically connect to a WebSocket server for enhanced live updates:
- File change notifications
- Component update tracking
- Performance metrics
- Network request monitoring

### State Visualization
When developing, you can visualize your app's state in real-time:
```javascript
import { devTools } from '@/utils/devtools';

// In your component
devTools.updateState(currentState);
```

### Performance Monitoring
The live preview includes FPS monitoring and performance metrics:
- Real-time FPS counter
- Network request timing
- Component render performance

## Troubleshooting

### Preview not updating?
1. Check that your frontend server is running on port 3000
2. Ensure file auto-save is enabled in VS Code/Cursor
3. Try manually refreshing with `Cmd/Ctrl + R`

### WebSocket connection failed?
This is optional - the preview will still work with standard HMR. To enable WebSocket features:
1. Run the full dev server: `node frontend-v2/scripts/dev-server.js`
2. This starts additional services for enhanced live updates

### Performance issues?
1. Disable unnecessary DevTools features
2. Use production build for performance testing: `npm run build && npm run preview`
3. Check browser console for errors

## Tips for Best Experience

1. **Use Auto-Save**: Enable auto-save in VS Code for instant updates
   ```json
   "files.autoSave": "afterDelay",
   "files.autoSaveDelay": 1000
   ```

2. **Format on Save**: Keep code consistent
   ```json
   "editor.formatOnSave": true
   ```

3. **Multiple Monitors**: Put live preview on second monitor for best workflow

4. **Browser DevTools**: Use Chrome DevTools alongside for debugging:
   - Right-click in preview → Inspect
   - Use React Developer Tools extension

## Next Steps

1. Start developing with live preview open
2. Make changes to see instant updates
3. Use device frames to test responsive design
4. Enable DevTools for component inspection

Happy coding! 🚀 