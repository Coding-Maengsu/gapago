import { ArrowRight } from 'lucide-react'
import Container from '@/components/ui/Container'
import Button from '@/components/ui/Button'

export default function CTA() {
  return (
    <section className="py-24 md:py-32 bg-card/50">
      <Container className="max-w-2xl text-center">
        <h2 className="text-3xl md:text-4xl font-bold mb-4">
          지금 바로 <span className="text-gradient">Research Gap</span>을 찾아보세요
        </h2>
        <p className="text-lg text-muted-foreground mb-8">
          키워드 하나면 충분합니다. 복잡한 설정 없이 바로 시작하세요.
        </p>
        <Button size="lg" glow href="/app" className="text-base px-10">
          무료로 분석 시작 <ArrowRight className="h-5 w-5" />
        </Button>
      </Container>
    </section>
  )
}
