

const nextConfig = {
  async rewrites() {
    return [];
  },
  env: {
    BACKEND_URL: process.env.BACKEND_URL ?? "http://localhost:8000",
  },
};

export default nextConfig;
