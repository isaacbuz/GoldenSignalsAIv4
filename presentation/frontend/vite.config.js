import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/signal': {
        target: 'https://your-backend-domain.com', // TODO: change to your deployed FastAPI backend
        changeOrigin: true,
        secure: false,
      },
    },
  },
})
