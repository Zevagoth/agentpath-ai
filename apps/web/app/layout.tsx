import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AgentPath AI",
  description:
    "Describe the case. Find the right path. Hybrid retrieval over a synthetic customer-service knowledge base.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  // The console chrome is left-to-right; individual fields flip direction based
  // on their own content (see lib/direction.ts).
  return (
    <html lang="en" dir="ltr">
      <body>{children}</body>
    </html>
  );
}
