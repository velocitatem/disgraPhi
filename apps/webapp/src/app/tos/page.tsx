export default function TermsOfServicePage() {
  return (
    <div className="container mx-auto p-8 max-w-4xl">
      <h1 className="text-3xl font-bold mb-6">Terms of Service</h1>
      <div className="prose">
        <p className="mb-4">Last updated: {new Date().toLocaleDateString()}</p>
        <h2 className="text-2xl font-semibold mt-6 mb-3">Acceptance of Terms</h2>
        <p className="mb-4">By accessing and using this service, you accept and agree to be bound by the terms and provision of this agreement.</p>
        <h2 className="text-2xl font-semibold mt-6 mb-3">Use License</h2>
        <p className="mb-4">Permission is granted to temporarily use this service for personal, non-commercial purposes.</p>
        <h2 className="text-2xl font-semibold mt-6 mb-3">Disclaimer</h2>
        <p className="mb-4">The materials on this service are provided on an &apos;as is&apos; basis. We make no warranties, expressed or implied.</p>
        <h2 className="text-2xl font-semibold mt-6 mb-3">Limitations</h2>
        <p className="mb-4">In no event shall we or our suppliers be liable for any damages arising out of the use or inability to use our service.</p>
      </div>
    </div>
  );
}
