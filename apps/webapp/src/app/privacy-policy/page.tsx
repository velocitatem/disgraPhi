export default function PrivacyPolicyPage() {
  return (
    <div className="container mx-auto p-8 max-w-4xl">
      <h1 className="text-3xl font-bold mb-6">Privacy Policy</h1>
      <div className="prose">
        <p className="mb-4">Last updated: {new Date().toLocaleDateString()}</p>
        <h2 className="text-2xl font-semibold mt-6 mb-3">Information Collection</h2>
        <p className="mb-4">We collect information that you provide directly to us.</p>
        <h2 className="text-2xl font-semibold mt-6 mb-3">Use of Information</h2>
        <p className="mb-4">We use the information we collect to provide and improve our services.</p>
        <h2 className="text-2xl font-semibold mt-6 mb-3">Data Security</h2>
        <p className="mb-4">We implement appropriate security measures to protect your information.</p>
        <h2 className="text-2xl font-semibold mt-6 mb-3">Contact Us</h2>
        <p className="mb-4">If you have questions about this privacy policy, please contact us.</p>
      </div>
    </div>
  );
}
