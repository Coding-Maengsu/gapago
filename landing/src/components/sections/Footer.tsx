import Container from '@/components/ui/Container'

export default function Footer() {
  return (
    <footer className="border-t border-border py-12">
      <Container className="flex flex-col md:flex-row items-center justify-between gap-4">
        <span className="text-lg font-bold text-[#5469d4]">GAPAGO</span>
        <p className="text-sm text-muted-foreground">&copy; 2026 GAPAGO. All rights reserved.</p>
        <div className="flex gap-4 text-sm text-muted-foreground">
          <a href="#" className="hover:text-foreground transition-colors">이용약관</a>
          <a href="#" className="hover:text-foreground transition-colors">개인정보처리방침</a>
        </div>
      </Container>
    </footer>
  )
}
