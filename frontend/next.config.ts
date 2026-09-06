import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  // The dashboard is entirely client-side. A static export keeps the same
  // local development flow while producing files Amplify can host directly.
  output: 'export',
};

export default nextConfig;
