import Link from "next/link";
import { getLocale } from "@/libs/locales";

export default function Footer() {
  const { common } = getLocale('en');

  return (
    <footer className="card mx-4 mb-4 mt-16">
      <div className="max-w-7xl mx-auto px-6 py-12">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-12 mb-8">
          <div>
            <h3 className="text-xl font-bold text-gradient mb-3">{common.footer.brand}</h3>
            <p className="opacity-80 text-sm">{common.footer.description}</p>
          </div>
          <div>
            <h4 className="font-bold mb-4">{common.footer.legal.title}</h4>
            <ul className="space-y-2">
              <li>
                <Link href="/privacy-policy" className="opacity-70 hover:opacity-100 transition-opacity text-sm">
                  {common.footer.legal.privacyPolicy}
                </Link>
              </li>
              <li>
                <Link href="/tos" className="opacity-70 hover:opacity-100 transition-opacity text-sm">
                  {common.footer.legal.termsOfService}
                </Link>
              </li>
            </ul>
          </div>
          <div>
            <h4 className="font-bold mb-4">{common.footer.company.title}</h4>
            <ul className="space-y-2">
              <li>
                <Link href="/blog" className="opacity-70 hover:opacity-100 transition-opacity text-sm">
                  {common.footer.company.blog}
                </Link>
              </li>
              <li>
                <Link href="/dashboard" className="opacity-70 hover:opacity-100 transition-opacity text-sm">
                  {common.footer.company.dashboard}
                </Link>
              </li>
            </ul>
          </div>
        </div>
        <div className="pt-8 border-t opacity-60 text-center text-sm" style={{borderColor: 'var(--border)'}}>
          <p>{common.footer.copyright}</p>
        </div>
      </div>
    </footer>
  );
}