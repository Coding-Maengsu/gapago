import Header from '@/components/sections/Header'
import Hero from '@/components/sections/Hero'
import Workflow from '@/components/sections/Workflow'
import Service from '@/components/sections/Service'
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
        <Workflow />
        <Service />
        <Difference />
        <Example />
        <CTA />
      </main>
      <Footer />
    </>
  )
}
