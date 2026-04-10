import React from 'react';
import { cn } from '../../lib/utils';

interface NoiseBackgroundProps {
      children: React.ReactNode;
      className?: string;
      containerClassName?: string;
      gradientColors?: string[];
}

export function NoiseBackground({
      children,
      className,
      containerClassName,
      gradientColors = ['rgb(255, 100, 150)', 'rgb(100, 150, 255)', 'rgb(255, 200, 100)'],
}: NoiseBackgroundProps) {
      const gradient = `linear-gradient(120deg, ${gradientColors.join(', ')})`;

      return (
            <div className={cn('relative overflow-hidden rounded-full', containerClassName)}>
                  <div className={cn('absolute inset-0 noise-gradient-shift', className)} style={{ backgroundImage: gradient }} />
                  <div className="absolute inset-0 noise-grain-overlay" />
                  <div className="relative z-10">{children}</div>
            </div>
      );
}
