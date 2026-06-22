import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Varna Weather LSTM",
  description: "Почасова прогноза на времето за Варна с LSTM — обучение, сравнение, прогноза.",
};

const NAV = [
  { href: "/", label: "Прогноза" },
  { href: "/training", label: "Обучение" },
  { href: "/comparison", label: "Сравнение" },
  { href: "/report", label: "Доклад" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="bg">
      <body>
        <div className="min-h-screen">
          <header className="border-b border-slate-800 bg-slate-900/70 backdrop-blur">
            <div className="mx-auto flex max-w-[1600px] items-center justify-between px-6 py-4">
              <div className="flex items-center gap-3">
                <span className="text-lg font-bold tracking-tight">Varna Weather LSTM</span>
                <span className="hidden text-xs text-slate-500 sm:inline">
                  почасови данни ERA5 · PyTorch
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
          <main className="mx-auto max-w-[1600px] px-6 py-8">{children}</main>
          <footer className="mx-auto max-w-[1600px] px-6 py-8 text-xs text-slate-500">
            Метеорологични данни от Open-Meteo (ECMWF ERA5), CC BY 4.0.
          </footer>
        </div>
      </body>
    </html>
  );
}
