/**
 * Composer stop glyph: a hollow ring with a filled squircle inside it.
 *
 * Two things this shape buys over the solid `Square` it replaces:
 *
 *  - Weight. Every other affordance in the composer row (Plus, Send) is a
 *    stroke-drawn 18px glyph, so a fully-filled square of the same size
 *    carried roughly double their visual mass and read as a UI box rather
 *    than an icon. The inner square is 40% of the ring's diameter, which
 *    puts the red mass back in line with its neighbours.
 *
 *  - Continuity while cancelling. The ring is not decoration — it IS the
 *    spinner track. The square fades out and an arc spins on the same
 *    circle, so the silhouette never changes. (Before, the square was
 *    swapped for a `Loader2`, jumping square -> circle mid-interaction.)
 *
 * The ring sits at 70% opacity and goes solid on hover, so the parent
 * button must carry Tailwind's `group` class.
 */

// viewBox is 20×20; the ring is inset by half its stroke (plus a hair) so the
// stroke stays inside the box at every rendered size.
const STROKE = 1.5;
const R = (20 - STROKE) / 2 - 0.5;
const CIRCUMFERENCE = 2 * Math.PI * R;
const ARC = CIRCUMFERENCE * 0.28;
const DASH = `${ARC} ${CIRCUMFERENCE - ARC}`;

interface StopGlyphProps {
  size?: number;
  /** Cancel is in flight: square fades, arc spins on the ring. */
  cancelling?: boolean;
}

export function StopGlyph({ size = 20, cancelling = false }: StopGlyphProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 20 20"
      fill="none"
      aria-hidden="true"
      className={cancelling ? 'animate-spin motion-reduce:animate-none' : undefined}
    >
      <circle
        cx="10"
        cy="10"
        r={R}
        stroke="currentColor"
        strokeWidth={STROKE}
        className={`transition-opacity duration-150 motion-reduce:transition-none ${
          cancelling ? 'opacity-25' : 'opacity-70 group-hover:opacity-100'
        }`}
      />
      {cancelling && (
        <circle
          cx="10"
          cy="10"
          r={R}
          stroke="currentColor"
          strokeWidth={STROKE}
          strokeLinecap="round"
          strokeDasharray={DASH}
        />
      )}
      <rect
        x="6"
        y="6"
        width="8"
        height="8"
        rx="2"
        fill="currentColor"
        className={`transition-opacity duration-150 motion-reduce:transition-none ${
          cancelling ? 'opacity-0' : 'opacity-100'
        }`}
      />
    </svg>
  );
}
