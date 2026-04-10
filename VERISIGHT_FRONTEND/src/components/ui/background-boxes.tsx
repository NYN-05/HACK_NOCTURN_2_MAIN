import React from 'react';
import { cn } from '../../lib/utils';

interface BoxesProps extends React.HTMLAttributes<HTMLDivElement> {
      className?: string;
}

export const BoxesCore = ({ className, ...rest }: BoxesProps) => {
      const rows = new Array(90).fill(1);
      const cols = new Array(64).fill(1);

      return (
            <div
                  style={{
                        transform: 'translate(-18%,-22%) skewX(-34deg) skewY(12deg) scale(0.88) rotate(0deg) translateZ(0)',
                  }}
                  className={cn(
                        'absolute inset-0 z-0 flex h-full w-full p-2',
                        className
                  )}
                  {...rest}
            >
                  {rows.map((_, i) => (
                        <div key={`row${i}`} className="relative h-6 w-12 border-l border-[#4a86af]/80">
                              {cols.map((_, j) => (
                                    <div key={`col${j}`} className="relative h-6 w-12 border-t border-r border-[#4a86af]/80">
                                          {j % 3 === 0 && i % 3 === 0 ? (
                                                <svg
                                                      xmlns="http://www.w3.org/2000/svg"
                                                      fill="none"
                                                      viewBox="0 0 24 24"
                                                      strokeWidth="1.5"
                                                      stroke="currentColor"
                                                      className="pointer-events-none absolute -top-[11px] -left-[16px] h-5 w-8 stroke-[1px] text-[#6ea7cf]/85"
                                                >
                                                      <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v12m6-6H6" />
                                                </svg>
                                          ) : null}
                                    </div>
                              ))}
                        </div>
                  ))}
            </div>
      );
};

export const Boxes = React.memo(BoxesCore);
