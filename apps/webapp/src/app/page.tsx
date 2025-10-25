import Link from "next/link";

export default function Home() {
  return (
    <div className="max-w-6xl mx-auto px-6 py-16">
      <div className="text-center mb-16">
        <h1 className="text-6xl font-bold mb-6 text-gradient">
          DisgraPhi
        </h1>
        <p className="text-xl opacity-80 mb-12 max-w-2xl mx-auto">
          Create personalized handwriting datasets for machine learning with our intuitive data preparation tool
        </p>
        <Link
          href="/data-prep"
          className="inline-block px-12 py-4 rounded-xl font-bold text-xl text-white shadow-magical"
          style={{background: 'linear-gradient(135deg, var(--castle-blue), var(--royal-blue))'}}
        >
          Get Started
        </Link>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-20">
        <div className="card p-8 text-center">
          <div className="text-4xl mb-4">📝</div>
          <h3 className="text-xl font-bold mb-3">Write Samples</h3>
          <p className="opacity-80">
            Write text samples by hand and upload photos of your handwriting
          </p>
        </div>

        <div className="card p-8 text-center">
          <div className="text-4xl mb-4">🎯</div>
          <h3 className="text-xl font-bold mb-3">Annotate Images</h3>
          <p className="opacity-80">
            Draw bounding boxes around your handwritten text with our interactive tool
          </p>
        </div>

        <div className="card p-8 text-center">
          <div className="text-4xl mb-4">📦</div>
          <h3 className="text-xl font-bold mb-3">Export Dataset</h3>
          <p className="opacity-80">
            Download a complete dataset with cropped images and ground truth labels
          </p>
        </div>
      </div>

      <div className="mt-20 card p-12">
        <h2 className="text-3xl font-bold mb-6 text-center">How It Works</h2>
        <ol className="space-y-4 max-w-2xl mx-auto">
          <li className="flex gap-4">
            <span className="font-bold text-xl" style={{color: 'var(--castle-blue)'}}>1.</span>
            <span>Select one or more text corpora from our pre-built collections</span>
          </li>
          <li className="flex gap-4">
            <span className="font-bold text-xl" style={{color: 'var(--castle-blue)'}}>2.</span>
            <span>Write each text sample by hand on paper and photograph it</span>
          </li>
          <li className="flex gap-4">
            <span className="font-bold text-xl" style={{color: 'var(--castle-blue)'}}>3.</span>
            <span>Upload your photos and draw bounding boxes around the text</span>
          </li>
          <li className="flex gap-4">
            <span className="font-bold text-xl" style={{color: 'var(--castle-blue)'}}>4.</span>
            <span>Download your complete dataset as a ZIP file with manifest</span>
          </li>
        </ol>
      </div>
    </div>
  );
}
