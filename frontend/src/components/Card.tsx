import { forwardRef } from 'react';
import type { PropsWithChildren } from 'react';

interface CardProps {
  className?: string;
}

export const Card = forwardRef<HTMLElement, PropsWithChildren<CardProps>>(function Card({ children, className = '' }, ref) {
  return (
    <section ref={ref} className={`card ${className}`.trim()}>
      {children}
    </section>
  );
});