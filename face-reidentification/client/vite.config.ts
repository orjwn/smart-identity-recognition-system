import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/kiosk": "http://127.0.0.1:8000",
      "/admin": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
      "/traveller": "http://127.0.0.1:8000"
    }
  }
});


