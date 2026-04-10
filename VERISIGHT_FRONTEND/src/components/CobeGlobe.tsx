import { useEffect, useRef } from 'react';
import createGlobe from 'cobe';

type RGB = [number, number, number];

const hexToRgbNormalized = (hex: string): RGB => {
      let r = 0;
      let g = 0;
      let b = 0;

      const cleanHex = hex.startsWith('#') ? hex.slice(1) : hex;

      if (cleanHex.length === 3) {
            r = parseInt(cleanHex[0] + cleanHex[0], 16);
            g = parseInt(cleanHex[1] + cleanHex[1], 16);
            b = parseInt(cleanHex[2] + cleanHex[2], 16);
      } else if (cleanHex.length === 6) {
            r = parseInt(cleanHex.substring(0, 2), 16);
            g = parseInt(cleanHex.substring(2, 4), 16);
            b = parseInt(cleanHex.substring(4, 6), 16);
      } else {
            return [0, 0, 0];
      }

      return [r / 255, g / 255, b / 255];
};

type ColorProp = RGB | string;

interface CobeGlobeProps {
      className?: string;
      canvasClassName?: string;
      theta?: number;
      dark?: number;
      scale?: number;
      diffuse?: number;
      mapSamples?: number;
      mapBrightness?: number;
      baseColor?: ColorProp;
      markerColor?: ColorProp;
      glowColor?: ColorProp;
      autoRotateSpeed?: number;
      draggable?: boolean;
      ariaLabel?: string;
}

export function CobeGlobe({
      className,
      canvasClassName,
      theta = 0.25,
      dark = 0,
      scale = 1.1,
      diffuse = 1.2,
      mapSamples = 60000,
      mapBrightness = 10,
      baseColor = [0.4, 0.6509, 1],
      markerColor = [1, 0, 0],
      glowColor = [0.2745, 0.5765, 0.898],
      autoRotateSpeed = 0.003,
      draggable = true,
      ariaLabel = 'Rotating globe visualization',
}: CobeGlobeProps) {
      const canvasRef = useRef<HTMLCanvasElement>(null);
      const globeRef = useRef<{ destroy: () => void; update: (state: Record<string, unknown>) => void } | null>(null);
      const frameRef = useRef<number | null>(null);

      const phiRef = useRef(0);
      const thetaRef = useRef(theta);
      const isDraggingRef = useRef(false);
      const lastMouseXRef = useRef(0);
      const lastMouseYRef = useRef(0);

      useEffect(() => {
            thetaRef.current = theta;
      }, [theta]);

      useEffect(() => {
            const canvas = canvasRef.current;
            if (!canvas) return;

            const resolvedBaseColor: RGB = typeof baseColor === 'string' ? hexToRgbNormalized(baseColor) : baseColor;
            const resolvedMarkerColor: RGB = typeof markerColor === 'string' ? hexToRgbNormalized(markerColor) : markerColor;
            const resolvedGlowColor: RGB = typeof glowColor === 'string' ? hexToRgbNormalized(glowColor) : glowColor;

            const initGlobe = () => {
                  if (globeRef.current) {
                        globeRef.current.destroy();
                        globeRef.current = null;
                  }

                  if (frameRef.current) {
                        cancelAnimationFrame(frameRef.current);
                        frameRef.current = null;
                  }

                  const rect = canvas.getBoundingClientRect();
                  const size = Math.min(rect.width, rect.height);
                  const devicePixelRatio = window.devicePixelRatio || 1;
                  const internalWidth = size * devicePixelRatio;
                  const internalHeight = size * devicePixelRatio;

                  canvas.width = internalWidth;
                  canvas.height = internalHeight;

                  globeRef.current = createGlobe(canvas, {
                        devicePixelRatio,
                        width: internalWidth,
                        height: internalHeight,
                        phi: phiRef.current,
                        theta: thetaRef.current,
                        dark,
                        scale,
                        diffuse,
                        mapSamples,
                        mapBrightness,
                        baseColor: resolvedBaseColor,
                        markerColor: resolvedMarkerColor,
                        glowColor: resolvedGlowColor,
                        opacity: 1,
                        offset: [0, 0],
                        markers: [],
                  });

                  const renderLoop = () => {
                        if (!globeRef.current) return;

                        if (!isDraggingRef.current) {
                              phiRef.current += autoRotateSpeed;
                        }

                        globeRef.current.update({
                              phi: phiRef.current,
                              theta: thetaRef.current,
                        });

                        frameRef.current = requestAnimationFrame(renderLoop);
                  };

                  frameRef.current = requestAnimationFrame(renderLoop);
            };

            const onMouseDown = (event: MouseEvent) => {
                  if (!draggable) return;
                  isDraggingRef.current = true;
                  lastMouseXRef.current = event.clientX;
                  lastMouseYRef.current = event.clientY;
                  canvas.style.cursor = 'grabbing';
            };

            const onMouseMove = (event: MouseEvent) => {
                  if (!draggable || !isDraggingRef.current) return;

                  const deltaX = event.clientX - lastMouseXRef.current;
                  const deltaY = event.clientY - lastMouseYRef.current;
                  const rotationSpeed = 0.005;

                  phiRef.current += deltaX * rotationSpeed;
                  thetaRef.current = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, thetaRef.current - deltaY * rotationSpeed));

                  lastMouseXRef.current = event.clientX;
                  lastMouseYRef.current = event.clientY;
            };

            const onMouseUp = () => {
                  if (!draggable) return;
                  isDraggingRef.current = false;
                  canvas.style.cursor = 'grab';
            };

            const onMouseLeave = () => {
                  if (!draggable) return;
                  if (isDraggingRef.current) {
                        isDraggingRef.current = false;
                        canvas.style.cursor = 'grab';
                  }
            };

            const onResize = () => initGlobe();

            initGlobe();
            canvas.addEventListener('mousedown', onMouseDown);
            canvas.addEventListener('mousemove', onMouseMove);
            canvas.addEventListener('mouseup', onMouseUp);
            canvas.addEventListener('mouseleave', onMouseLeave);
            window.addEventListener('resize', onResize);

            return () => {
                  window.removeEventListener('resize', onResize);
                  canvas.removeEventListener('mousedown', onMouseDown);
                  canvas.removeEventListener('mousemove', onMouseMove);
                  canvas.removeEventListener('mouseup', onMouseUp);
                  canvas.removeEventListener('mouseleave', onMouseLeave);

                  if (globeRef.current) {
                        globeRef.current.destroy();
                        globeRef.current = null;
                  }

                  if (frameRef.current) {
                        cancelAnimationFrame(frameRef.current);
                        frameRef.current = null;
                  }
            };
      }, [
            autoRotateSpeed,
            baseColor,
            dark,
            diffuse,
            draggable,
            glowColor,
            mapBrightness,
            mapSamples,
            markerColor,
            scale,
      ]);

      return (
            <div className={`flex items-center justify-center mx-auto ${className ?? ''}`.trim()}>
                  <canvas
                        ref={canvasRef}
                        aria-label={ariaLabel}
                        role="img"
                        className={canvasClassName}
                        style={{
                              width: '100%',
                              height: '100%',
                              display: 'block',
                              cursor: draggable ? 'grab' : 'default',
                        }}
                  />
            </div>
      );
}
