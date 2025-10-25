import { getLocale } from "@/libs/locales";

export default function Testimonials1() {
  const { common } = getLocale('en');

  return (
    <section>
      {/* TODO: Style this testimonials section when implementing in your project */}
      <div>
        <h2>{common.testimonials.title}</h2>
        <div>
          {common.testimonials.items.map((testimonial, index) => (
            <div key={index}>
              <p>&ldquo;{testimonial.content}&rdquo;</p>
              <div>
                <p>{testimonial.name}</p>
                <p>{testimonial.role}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}