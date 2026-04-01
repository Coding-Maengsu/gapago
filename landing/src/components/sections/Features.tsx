import Container from '@/components/ui/Container'
import SectionHeading from '@/components/ui/SectionHeading'
import { FEATURES } from '@/lib/constants'

export default function Features() {
  return (
    <section id="features" className="py-16 md:py-20">
      <Container>
        <SectionHeading
          title="핵심 기능"
          highlightText="핵심 기능"
          subtitle="연구자에게 필요한 모든 기능을 갖추고 있습니다."
        />
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {FEATURES.map((feature) => (
            <div
              key={feature.title}
              className="flex gap-4 rounded-xl border border-border bg-card p-6 hover:border-primary/30 transition-colors"
            >
              <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
                <feature.icon className="h-5 w-5 text-primary" />
              </div>
              <div>
                <h3 className="font-semibold mb-1">{feature.title}</h3>
                <p className="text-sm text-muted-foreground">{feature.description}</p>
              </div>
            </div>
          ))}
        </div>
      </Container>
    </section>
  )
}
