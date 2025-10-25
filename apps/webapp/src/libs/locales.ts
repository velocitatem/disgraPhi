import commonEn from '@/locales/en/common.json';

// TODO: Add more languages as needed
const locales = {
  en: {
    common: commonEn
  }
};

export function getLocale(locale: string = 'en') {
  return locales[locale as keyof typeof locales] || locales.en;
}

export function t(key: string, locale: string = 'en') {
  const translations = getLocale(locale);
  const keys = key.split('.');

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let value: any = translations;
  for (const k of keys) {
    value = value?.[k];
  }

  return value || key;
}