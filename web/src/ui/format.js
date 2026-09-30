const MINUS = "−";

// Fixed decimals with a real minus sign; -0 prints as 0.
export function fmt(x, digits = 3) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  if (!Number.isFinite(x)) return x > 0 ? "∞" : `${MINUS}∞`;
  const text = Math.abs(x).toFixed(digits);
  const zero = Number(text) === 0;
  return (x < 0 && !zero ? MINUS : "") + text;
}

export function fmtSigned(x, digits = 3) {
  const text = fmt(x, digits);
  return x > 0 && Number(Math.abs(x).toFixed(digits)) !== 0 ? `+${text}` : text;
}

// Shortest readable form: 1, 0.5, −13, 0.792
export function fmtShort(x, maxDigits = 3) {
  if (!Number.isFinite(x)) return fmt(x);
  const text = String(Number(Math.abs(x).toFixed(maxDigits)));
  return (x < 0 && Number(text) !== 0 ? MINUS : "") + text;
}

export const ARROWS = ["↑", "→", "↓", "←"];
