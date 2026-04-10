import { CSSProperties } from 'react';

interface DottedGlowBackgroundProps {
      className?: string;
      opacity?: number;
      gap?: number;
      radius?: number;
      colorLightVar?: string;
      glowColorLightVar?: string;
      colorDarkVar?: string;
      glowColorDarkVar?: string;
      backgroundOpacity?: number;
      speedMin?: number;
      speedMax?: number;
      speedScale?: number;
}

export function DottedGlowBackground({
      className,
      opacity = 1,
      gap = 10,
      radius = 1.6,
      colorLightVar = '--color-neutral-500',
      glowColorLightVar = '--color-neutral-600',
      colorDarkVar = '--color-neutral-500',
      glowColorDarkVar = '--color-sky-800',
      backgroundOpacity = 0,
      speedMin = 0.3,
      speedMax = 1.6,
      speedScale = 1,
}: DottedGlowBackgroundProps) {
      const averageSpeed = (speedMin + speedMax) / 2;
      const normalizedSpeed = Math.max(0.2, averageSpeed * Math.max(speedScale, 0.2));
      const animationDuration = `${Math.max(8, 28 / normalizedSpeed)}s`;
      const effectiveDotColor = `var(${colorDarkVar}, var(${colorLightVar}, rgba(120,120,120,0.35)))`;
      const effectiveGlowColor = `var(${glowColorDarkVar}, var(${glowColorLightVar}, rgba(0,229,255,0.25)))`;

      const style: CSSProperties = {
            opacity,
            backgroundColor: `rgba(0, 0, 0, ${backgroundOpacity})`,
            backgroundImage: [
                  `radial-gradient(circle at center, ${effectiveDotColor} 0 ${radius}px, transparent ${radius + 0.1}px)`,
                  `radial-gradient(circle at center, ${effectiveGlowColor} 0 ${radius * 2.2}px, transparent ${radius * 3.2}px)`,
            ].join(','),
            backgroundSize: `${gap}px ${gap}px, ${gap * 2.2}px ${gap * 2.2}px`,
            backgroundPosition: '0 0, 6px 6px',
            animation: `dottedGlowDrift ${animationDuration} linear infinite`,
      };

      return <div className={className} style={style} aria-hidden="true" />;
}
