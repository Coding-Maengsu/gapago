import Container from '@/components/ui/Container'
import SectionHeading from '@/components/ui/SectionHeading'
import { WORKFLOW_STEPS } from '@/lib/constants'

export default function Workflow() {
  return (
    <section id="workflow" className="py-24 md:py-32 bg-card/50">
      <Container>
        <SectionHeading
          title="4단계로 끝나는 GAP 분석"
          highlightText="GAP 분석"
          subtitle="복잡한 문헌 분석, 이제 몇 분이면 충분합니다."
        />
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-8">
          {WORKFLOW_STEPS.map((step, i) => (
            <div key={step.step} className="relative text-center">
              <span className="text-xs text-primary font-bold mb-3 block">{step.step}</span>
              <div className="h-16 w-16 rounded-2xl bg-primary/10 border border-primary/20 flex items-center justify-center mb-4 mx-auto">
                <step.icon className="h-7 w-7 text-primary" />
              </div>
              <h3 className="font-semibold mb-1">{step.title}</h3>
              <p className="text-sm text-muted-foreground">{step.description}</p>

              {/* 연결선 (마지막 스텝 제외, lg에서만 표시) */}
              {i < WORKFLOW_STEPS.length - 1 && (
                <div className="hidden lg:block absolute top-10 -right-4 w-8 h-px bg-border" />
              )}
            </div>
          ))}
        </div>
      </Container>
    </section>
  )
}
