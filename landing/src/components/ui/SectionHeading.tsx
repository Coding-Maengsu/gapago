type SectionHeadingProps = {
  title: string
  highlightText: string
  subtitle: string
}

export default function SectionHeading({ title, highlightText, subtitle }: SectionHeadingProps) {
  const parts = title.split(highlightText)

  return (
    <div className="text-center mb-16">
      <h2 className="text-3xl md:text-4xl font-bold mb-4">
        {parts[0]}
        <span className="text-gradient">{highlightText}</span>
        {parts[1] || ''}
      </h2>
      <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
        {subtitle}
      </p>
    </div>
  )
}
