import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Varna Weather LSTM",
  description: "Hourly weather forecasting for Varna with an LSTM — train, compare, forecast.",
};

const NAV = [
  { href: "/", label: "Forecast" },
  { href: "/training", label: "Training" },
  { href: "/comparison", label: "Comparison" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="min-h-screen">
          <header className="border-b border-slate-800 bg-slate-900/70 backdrop-blur">
            <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
              <div className="flex items-center gap-3">
                <span className="text-lg font-bold tracking-tight">Varna Weather LSTM</span>
                <span className="hidden text-xs text-slate-500 sm:inline">
                  hourly ERA5 · PyTorch
                </span>
              </div>
              <nav className="flex gap-1">
                {NAV.map((item) => (
                  <Link
                    key={item.href}
                    href={item.href}
                    className="rounded-lg px-3 py-1.5 text-sm text-slate-300 hover:bg-slate-800 hover:text-white"
                  >
                    {item.label}
                  </Link>
                ))}
              </nav>
            </div>
          </header>
          <main className="mx-auto max-w-7xl px-6 py-8">{children}</main>
          <footer className="mx-auto max-w-7xl px-6 py-8 text-xs text-slate-500">
            Weather data by Open-Meteo (ECMWF ERA5), CC BY 4.0.
          </footer>
        </div>
      </body>
    </html>
  );
}
