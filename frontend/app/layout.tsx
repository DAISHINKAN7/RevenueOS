import type { Metadata } from "next";
import { Shell } from "@/components/ui/shell";
import "./globals.css";

export const metadata: Metadata = {
  title: "RevenueOS — Autonomous Revenue Recovery",
  description:
    "Detect revenue leakage, choose the highest-value intervention, and execute bounded recovery workflows.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="antialiased">
        <Shell>{children}</Shell>
      </body>
    </html>
  );
}