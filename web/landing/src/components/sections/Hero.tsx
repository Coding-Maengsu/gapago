import { ArrowRight, ChevronDown } from 'lucide-react'
import Button from '@/components/ui/Button'

export default function Hero() {
  return (
    <section className="relative min-h-screen flex items-center justify-center pt-16 overflow-hidden">
      {/* 그라디언트 오버레이 (배경 이미지 대신) */}
      <div className="absolute inset-0 bg-gradient-to-b from-background/60 via-background/80 to-background" />

      {/* 콘텐츠 */}
      <div className="relative z-10 text-center px-4">
        {/* 배지 */}
        <span className="inline-block rounded-full border border-primary/30 bg-primary/10 px-4 py-1.5 text-xs text-primary font-medium animate-pulse-glow mb-8">
          AI 기반 연구 GAP 분석
        </span>

        {/* 제목 */}
        <h1 className="text-4xl md:text-6xl lg:text-7xl font-extrabold max-w-4xl leading-tight mb-6 mx-auto">
          논문 속 <span className="text-[#5469d4]">연구 GAP</span>을
          <br />찾아드립니다
        </h1>

        {/* 부제목 */}
        <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto mb-10">
          키워드로 관련 논문들을 분석하고, 연구 GAP을 도출합니다.
          <br className="hidden sm:block" />
          단순 요약이 아닌, 진짜 빈틈을 발견하고 연구를 시작하세요.
        </p>

        {/* CTA 버튼 */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
          <Button size="lg" glow href="/app">
            무료로 분석 시작 <ArrowRight className="h-5 w-5" />
          </Button>
        </div>
      </div>

      {/* 스크롤 유도 화살표 */}
      <a
        href="#workflow"
        className="absolute bottom-8 left-1/2 -translate-x-1/2 z-10 animate-bounce text-foreground/30 hover:text-foreground/60 transition-colors"
        aria-label="아래로 스크롤"
      >
        <ChevronDown className="h-8 w-8" />
      </a>
    </section>
  )
}
