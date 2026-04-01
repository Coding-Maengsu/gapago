import Container from '@/components/ui/Container'
import SectionHeading from '@/components/ui/SectionHeading'
import { EXAMPLES } from '@/lib/constants'

export default function Example() {
  return (
    <section id="examples" className="py-16 md:py-20">
      <Container>
        <SectionHeading
          title="이런 GAP을 찾아냅니다"
          highlightText="GAP"
          subtitle="실제 분석 예시를 확인해보세요."
        />
        <div className="grid md:grid-cols-3 gap-8">
          {EXAMPLES.map((example) => (
            <div key={example.query} className="rounded-xl border border-border bg-card p-8">
              <span className="inline-block rounded-full bg-primary/10 px-3 py-1 text-xs text-primary font-medium mb-6">
                {example.query}
              </span>
              <ul className="space-y-3">
                {example.gaps.map((gap) => (
                  <li key={gap} className="flex items-start gap-3">
                    <span className="h-1.5 w-1.5 rounded-full bg-primary mt-2 shrink-0" />
                    <span className="text-sm text-muted-foreground">{gap}</span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </Container>
    </section>
  )
}
