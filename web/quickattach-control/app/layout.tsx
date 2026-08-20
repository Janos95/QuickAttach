import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "QuickAttach Robot Control",
  description: "Explore the SO-101 QuickAttach robot and radial tool fixture in Three.js.",
  other: { "codex-preview": "development" },
  icons: { icon: "/favicon.svg", shortcut: "/favicon.svg" },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
