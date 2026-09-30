// A small seeded generator (mulberry32) so every experiment can be replayed.
export class Rng {
  constructor(seed = 1) {
    this.state = seed >>> 0;
  }

  random() {
    let t = (this.state = (this.state + 0x6d2b79f5) >>> 0);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  integers(n) {
    return Math.floor(this.random() * n);
  }

  // Sample an index from a probability vector.
  choice(probs) {
    let u = this.random();
    for (let i = 0; i < probs.length; i++) {
      u -= probs[i];
      if (u < 0) return i;
    }
    return probs.length - 1;
  }
}
