import Container from '@/components/ui/Container'

export default function Footer() {
  return (
    <footer className="border-t border-border py-12">
      <Container className="flex flex-col md:flex-row items-center justify-between gap-4">
        <a href="/" className="flex items-center gap-2">
          <img src="/new_logo.png" alt="GAPAGO" className="h-7 w-7 rounded-lg object-cover" />
          <span className="text-lg font-bold text-[#5469d4]">GAPAGO</span>
        </a>
        <p className="text-sm text-muted-foreground">&copy; 2026 GAPAGO. All rights reserved.</p>
      </Container>
    </footer>
  )
}
