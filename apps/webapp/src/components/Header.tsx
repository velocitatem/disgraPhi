import Link from "next/link";
import { getLocale } from "@/libs/locales";

export default function Header() {
  const { common } = getLocale('en');

  return (
    <header className="card mx-4 mt-4 sticky top-4 z-50">
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <Link href="/" className="text-2xl font-bold text-gradient hover:opacity-80 transition-opacity">
              {common.header.brand}
            </Link>
          </div>
          <nav className="flex items-center gap-8">
            <Link href="/" className="font-semibold hover:opacity-70 transition-opacity">
              {common.header.nav.home}
            </Link>
            <Link href="/data-prep" className="font-semibold hover:opacity-70 transition-opacity">
              Data Prep
            </Link>
            <Link href="/dashboard" className="font-semibold hover:opacity-70 transition-opacity">
              {common.header.nav.dashboard}
            </Link>
          </nav>
          <div>
            <Link
              href="/login"
              className="px-6 py-2 rounded-xl font-bold text-white shadow-magical"
              style={{background: 'linear-gradient(135deg, var(--castle-blue), var(--royal-blue))'}}
            >
              {common.header.actions.signIn}
            </Link>
          </div>
        </div>
      </div>
    </header>
  );
}
