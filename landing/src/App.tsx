import Header from '@/components/sections/Header'
import Hero from '@/components/sections/Hero'
import Service from '@/components/sections/Service'
import Workflow from '@/components/sections/Workflow'
import Features from '@/components/sections/Features'
import Difference from '@/components/sections/Difference'
import Example from '@/components/sections/Example'
import CTA from '@/components/sections/CTA'
import Footer from '@/components/sections/Footer'

export default function App() {
  return (
    <>
      <Header />
      <main>
        <Hero />
        <Service />
        <Workflow />
        <Features />
        <Difference />
        <Example />
        <CTA />
      </main>
      <Footer />
    </>
  )
}
