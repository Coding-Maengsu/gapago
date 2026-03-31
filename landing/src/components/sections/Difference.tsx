import { CheckCircle2, XCircle } from 'lucide-react'
import Container from '@/components/ui/Container'
import SectionHeading from '@/components/ui/SectionHeading'
import { COMPARISON_DATA } from '@/lib/constants'

export default function Difference() {
  return (
    <section id="difference" className="py-24 md:py-32 bg-card/50">
      <Container className="max-w-3xl">
        <SectionHeading
          title="왜 GAPAGO인가요?"
          highlightText="GAPAGO"
          subtitle="기존 논문 분석 도구와의 차이를 확인하세요."
        />
        <div className="rounded-xl border border-border overflow-hidden">
          {/* 헤더 */}
          <div className="bg-secondary/50 grid grid-cols-3 text-sm font-medium p-4">
            <span>기능</span>
            <span className="text-center">GAPAGO</span>
            <span className="text-center">기존 도구</span>
          </div>
          {/* 데이터 행 */}
          {COMPARISON_DATA.map((row) => (
            <div key={row.feature} className="grid grid-cols-3 p-4 border-t border-border items-center">
              <span className="text-sm">{row.feature}</span>
              <div className="flex justify-center">
                <CheckCircle2 className="h-5 w-5 text-primary" />
              </div>
              <div className="flex justify-center">
                {row.existing ? (
                  <CheckCircle2 className="h-5 w-5 text-muted-foreground/50" />
                ) : (
                  <XCircle className="h-5 w-5 text-muted-foreground/30" />
                )}
              </div>
            </div>
          ))}
        </div>
      </Container>
    </section>
  )
}
