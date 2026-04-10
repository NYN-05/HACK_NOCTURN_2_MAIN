import type { ButtonHTMLAttributes, PropsWithChildren } from 'react';

type ButtonVariant = 'primary' | 'secondary' | 'ghost';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
}

export function Button({ children, variant = 'primary', className = '', type = 'button', ...props }: PropsWithChildren<ButtonProps>) {
  return (
    <button type={type} className={`button button--${variant} ${className}`.trim()} {...props}>
      {children}
    </button>
  );
}
