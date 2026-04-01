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
            <div key={example.query} className="rounded-xl border border-border bg-card p-6">
              {/* 키워드 배지 */}
              <span className="inline-block rounded-full bg-primary/10 px-4 py-1.5 text-base text-primary font-semibold mb-4">
                {example.query}
              </span>

              {/* 메타 정보 */}
              <div className="flex gap-4 text-xs text-muted-foreground mb-4">
                <span>논문 {example.paperCount}편 분석</span>
                <span>연구 축 {example.axes.length}개</span>
              </div>

              {/* 연구 축 태그 */}
              <div className="flex flex-wrap gap-1.5 mb-5">
                {example.axes.map((axis) => (
                  <span key={axis} className="text-xs bg-secondary px-2 py-0.5 rounded">
                    {axis}
                  </span>
                ))}
              </div>

              {/* GAP 목록 */}
              <div className="space-y-3">
                {example.gaps.map((gap) => (
                  <div key={gap.title} className="bg-background rounded-lg p-3 border border-border/50">
                    <p className="text-sm font-medium mb-1">{gap.title}</p>
                    <p className="text-xs text-muted-foreground mb-2">{gap.detail}</p>
                    <span className="text-xs text-primary">근거 논문 {gap.relatedPapers}편</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </Container>
    </section>
  )
}
