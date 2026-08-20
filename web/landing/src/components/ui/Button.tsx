import { type ButtonHTMLAttributes, type AnchorHTMLAttributes } from 'react'

type Variant = 'primary' | 'outline'
type Size = 'sm' | 'lg'

type BaseProps = {
  variant?: Variant
  size?: Size
  glow?: boolean
  children: React.ReactNode
}

type ButtonAsButton = BaseProps & ButtonHTMLAttributes<HTMLButtonElement> & { href?: undefined }
type ButtonAsAnchor = BaseProps & AnchorHTMLAttributes<HTMLAnchorElement> & { href: string }
type ButtonProps = ButtonAsButton | ButtonAsAnchor

const variantStyles: Record<Variant, string> = {
  primary: 'bg-[#5469d4] hover:bg-[#4255b8] text-white',
  outline: 'border border-border bg-transparent hover:bg-secondary text-foreground',
}

const sizeStyles: Record<Size, string> = {
  sm: 'px-4 py-2 text-sm',
  lg: 'px-8 py-3 text-base',
}

export default function Button({ variant = 'primary', size = 'sm', glow, children, ...props }: ButtonProps) {
  const classes = [
    'inline-flex items-center justify-center gap-2 rounded-[var(--radius)] font-medium',
    'transition-colors duration-200 cursor-pointer',
    variantStyles[variant],
    sizeStyles[size],
    glow ? 'glow-border' : '',
    ('className' in props && props.className) || '',
  ].filter(Boolean).join(' ')

  if ('href' in props && props.href) {
    const { className: _, ...rest } = props as ButtonAsAnchor
    return <a className={classes} {...rest}>{children}</a>
  }

  const { className: _, ...rest } = props as ButtonAsButton
  return <button className={classes} {...rest}>{children}</button>
}
