/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Produce a self-contained server bundle for the minimal Docker image.
  output: "standalone",
  // react-pdf / pdfjs-dist ship an ESM worker that needs to be treated as an external.
  experimental: {
    serverComponentsExternalPackages: ["pdfjs-dist"],
  },
  webpack: (config) => {
    // pdfjs-dist expects `canvas` to be available at runtime; stub it in the browser bundle.
    // The pdf.worker is served statically from /public/pdf.worker.min.mjs (not bundled).
    config.resolve.alias = {
      ...config.resolve.alias,
      canvas: false,
    };
    return config;
  },
};

export default nextConfig;
