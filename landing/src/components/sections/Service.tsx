import Container from '@/components/ui/Container'
import SectionHeading from '@/components/ui/SectionHeading'
import { SERVICE_CARDS } from '@/lib/constants'

export default function Service() {
  return (
    <section className="py-24 md:py-32">
      <Container>
        <SectionHeading
          title="GAPAGO가 하는 일"
          highlightText="GAPAGO"
          subtitle="연구자가 가장 오래 걸리는 문헌 분석 과정을 AI로 혁신합니다."
        />
        <div className="grid md:grid-cols-3 gap-8">
          {SERVICE_CARDS.map((card) => (
            <div
              key={card.title}
              className="rounded-xl border border-border bg-card p-8 hover:border-primary/40 hover:shadow-[var(--shadow-glow)] transition-all duration-300"
            >
              <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
                <card.icon className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-lg font-semibold mb-2">{card.title}</h3>
              <p className="text-muted-foreground text-sm">{card.description}</p>
            </div>
          ))}
        </div>
      </Container>
    </section>
  )
}
