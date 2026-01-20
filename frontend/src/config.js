// API Base URL configuration
// - In production (Docker): empty string = same origin, nginx proxies /api to backend
// - In development: localhost:5001 for direct backend access
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '';
