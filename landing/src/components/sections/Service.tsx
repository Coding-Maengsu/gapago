import Container from '@/components/ui/Container'
import SectionHeading from '@/components/ui/SectionHeading'
import { SERVICE_CARDS, EXTRA_FEATURES } from '@/lib/constants'

export default function Service() {
  return (
    <section id="service" className="py-16 md:py-20">
      <Container>
        <SectionHeading
          title="GAPAGO가 하는 일"
          highlightText="GAPAGO"
          subtitle="연구자가 가장 오래 걸리는 문헌 분석 과정을 AI로 혁신합니다."
        />
        {/* 핵심 서비스 (큰 카드 3개) */}
        <div className="grid md:grid-cols-3 gap-8 mb-12">
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
        {/* 추가 기능 (작은 카드 4개) */}
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {EXTRA_FEATURES.map((feature) => (
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
