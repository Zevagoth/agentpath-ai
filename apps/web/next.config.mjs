/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // The API base URL is read in the browser, so it must be inlined at build time.
  // Defaults to the loopback API; nothing here talks to the network.
  env: {
    NEXT_PUBLIC_AGENTPATH_API_URL:
      process.env.NEXT_PUBLIC_AGENTPATH_API_URL ?? "http://127.0.0.1:8000",
  },
};

export default nextConfig;
