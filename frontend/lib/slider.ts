/** Base UI Slider onValueChange returns number | readonly number[], extract first value */
export function sliderVal(v: number | readonly number[]): number {
  return Array.isArray(v) ? v[0] : (v as number);
}
