/**
 * Composer stop glyph: a hollow rounded-square ring with a filled mark inside.
 *
 * Three things this shape buys:
 *
 *  - Weight. Every other affordance in the composer row (Plus, Send) is a
 *    stroke-drawn 18px glyph, so a fully-filled square of the same size
 *    carried roughly double their visual mass and read as a UI box rather
 *    than an icon. The inner square is 40% of the ring's span, which puts the
 *    mass back in line with its neighbours.
 *
 *  - Geometry. The ring used to be a circle, the only round outline in a
 *    composer built entirely from the `--radius-*` ladder. `rx=5` on a 20px
 *    box is the same optical corner as the `rounded-lg` boxes around it.
 *
 *  - Continuity while cancelling. The ring is not decoration — it IS the
 *    spinner track. The inner mark fades out and an arc runs around the same
 *    outline, so the silhouette never changes. (Before, the square was
 *    swapped for a `Loader2`, jumping square -> circle mid-interaction.)
 *
 * The cancel animation is NOT `animate-spin`. A circle is rotation-invariant,
 * so rotating the old glyph was invisible; rotating a square is very visible
 * and wobbles the outline — exactly what the continuity rule above exists to
 * prevent. `animate-stop-arc` (index.css) drives stroke-dashoffset instead, so
 * the outline stays nailed in place and only the light moves.
 *
 * The ring sits at 70% opacity and goes solid on hover, so the parent button
 * must carry Tailwind's `group` class.
 */

// viewBox is 20×20; the ring is inset by half its stroke (plus a hair) so the
// stroke stays inside the box at every rendered size.
const STROKE = 1.5;
const INSET = STROKE / 2 + 0.5;
const SPAN = 20 - INSET * 2;
const RADIUS = 5;

// `pathLength` renormalizes the outline to 100 units, so the dash array is
// literally "28% lit, 72% dark" and needs no perimeter arithmetic — which for
// a rounded rect would otherwise mean summing four sides and four arcs.
const PATH_LENGTH = 100;
const ARC = 28;

interface StopGlyphProps {
  size?: number;
  /** Cancel is in flight: the inner mark fades, an arc runs the outline. */
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
    >
      <rect
        x={INSET}
        y={INSET}
        width={SPAN}
        height={SPAN}
        rx={RADIUS}
        stroke="currentColor"
        strokeWidth={STROKE}
        className={`transition-opacity duration-150 motion-reduce:transition-none ${
          cancelling ? 'opacity-25' : 'opacity-70 group-hover:opacity-100'
        }`}
      />
      {cancelling && (
        <rect
          x={INSET}
          y={INSET}
          width={SPAN}
          height={SPAN}
          rx={RADIUS}
          stroke="currentColor"
          strokeWidth={STROKE}
          strokeLinecap="round"
          pathLength={PATH_LENGTH}
          strokeDasharray={`${ARC} ${PATH_LENGTH - ARC}`}
          className="animate-stop-arc motion-reduce:animate-none"
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
