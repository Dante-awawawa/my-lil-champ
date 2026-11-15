// Browser inference for the GPT defined in gpt.py (no dependencies).
// Uses a KV cache so each new token costs one pass over the weights.

function erf(x) {
  // Abramowitz & Stegun 7.1.26, max error ~1.5e-7
  const s = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}

function layerNorm(x, w, b, out) {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let v = 0;
  for (let i = 0; i < n; i++) { const d = x[i] - mean; v += d * d; }
  const inv = 1 / Math.sqrt(v / n + 1e-5);
  for (let i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * w[i] + b[i];
}

// out = W x, where W is (rows, cols) row-major (same layout as nn.Linear.weight)
function matvec(W, rows, cols, x, out) {
  for (let r = 0, base = 0; r < rows; r++, base += cols) {
    let s = 0;
    for (let c = 0; c < cols; c++) s += W[base + c] * x[c];
    out[r] = s;
  }
}

class GPTModel {
  constructor(manifest, buffer) {
    this.cfg = manifest.config;
    this.chars = manifest.chars;
    this.stoi = new Map(this.chars.map((c, i) => [c, i]));

    const all = new Float32Array(buffer);
    this.t = {};
    for (const [name, info] of Object.entries(manifest.tensors)) {
      const n = info.shape.reduce((a, b) => a * b, 1);
      this.t[name] = all.subarray(info.offset, info.offset + n);
    }

    const { n_layer, n_head, n_embd, block_size, vocab_size } = this.cfg;
    this.L = n_layer; this.H = n_head; this.C = n_embd;
    this.hd = n_embd / n_head; this.T = block_size; this.V = vocab_size;

    this.kc = Array.from({ length: this.L }, () => new Float32Array(this.T * this.C));
    this.vc = Array.from({ length: this.L }, () => new Float32Array(this.T * this.C));
    this.x = new Float32Array(this.C);
    this.xn = new Float32Array(this.C);
    this.qkv = new Float32Array(3 * this.C);
    this.att = new Float32Array(this.C);
    this.tmp = new Float32Array(this.C);
    this.hid = new Float32Array(4 * this.C);
    this.scores = new Float32Array(this.T);
    this.logits = new Float32Array(this.V);
    this.pos = 0;
  }

  encode(s) {
    const ids = [];
    for (const ch of s) {
      const i = this.stoi.get(ch);
      if (i !== undefined) ids.push(i);   // unknown characters are dropped, like gpt.py
    }
    return ids;
  }

  decode(ids) { return ids.map(i => this.chars[i]).join(""); }

  reset() { this.pos = 0; }

  // Feed one token at the current position; returns logits for the next token.
  step(token) {
    const { C, H, hd, L, t } = this;
    const pos = this.pos;
    const { x, xn, qkv, att, tmp, hid, scores } = this;
    const te = t["tok_emb.weight"], pe = t["pos_emb.weight"];
    for (let i = 0; i < C; i++) x[i] = te[token * C + i] + pe[pos * C + i];

    const scale = 1 / Math.sqrt(hd);
    for (let l = 0; l < L; l++) {
      const p = `blocks.${l}.`;
      const kc = this.kc[l], vc = this.vc[l];

      layerNorm(x, t[p + "ln1.weight"], t[p + "ln1.bias"], xn);
      matvec(t[p + "attn.qkv.weight"], 3 * C, C, xn, qkv);
      kc.set(qkv.subarray(C, 2 * C), pos * C);
      vc.set(qkv.subarray(2 * C, 3 * C), pos * C);

      for (let h = 0; h < H; h++) {
        const ho = h * hd;
        let max = -Infinity;
        for (let j = 0; j <= pos; j++) {
          let s = 0;
          const ko = j * C + ho;
          for (let d = 0; d < hd; d++) s += qkv[ho + d] * kc[ko + d];
          s *= scale;
          scores[j] = s;
          if (s > max) max = s;
        }
        let sum = 0;
        for (let j = 0; j <= pos; j++) { scores[j] = Math.exp(scores[j] - max); sum += scores[j]; }
        for (let d = 0; d < hd; d++) att[ho + d] = 0;
        for (let j = 0; j <= pos; j++) {
          const w = scores[j] / sum, vo = j * C + ho;
          for (let d = 0; d < hd; d++) att[ho + d] += w * vc[vo + d];
        }
      }
      matvec(t[p + "attn.proj.weight"], C, C, att, tmp);
      for (let i = 0; i < C; i++) x[i] += tmp[i];

      layerNorm(x, t[p + "ln2.weight"], t[p + "ln2.bias"], xn);
      matvec(t[p + "mlp.fc.weight"], 4 * C, C, xn, hid);
      for (let i = 0; i < 4 * C; i++) { const v = hid[i]; hid[i] = 0.5 * v * (1 + erf(v * Math.SQRT1_2)); }
      matvec(t[p + "mlp.proj.weight"], C, 4 * C, hid, tmp);
      for (let i = 0; i < C; i++) x[i] += tmp[i];
    }

    layerNorm(x, t["ln_f.weight"], t["ln_f.bias"], xn);
    matvec(te, this.V, C, xn, this.logits);   // tied output head
    this.pos++;
    return this.logits;
  }

  sample(logits, temperature, topK) {
    const V = logits.length;
    const temp = Math.max(temperature, 1e-6);
    const l = new Float32Array(V);
    for (let i = 0; i < V; i++) l[i] = logits[i] / temp;
    if (topK > 0 && topK < V) {
      const thr = Float32Array.from(l).sort()[V - topK];
      for (let i = 0; i < V; i++) if (l[i] < thr) l[i] = -Infinity;
    }
    let max = -Infinity;
    for (let i = 0; i < V; i++) if (l[i] > max) max = l[i];
    let sum = 0;
    for (let i = 0; i < V; i++) { l[i] = Math.exp(l[i] - max); sum += l[i]; }
    let r = Math.random() * sum;
    for (let i = 0; i < V; i++) { r -= l[i]; if (r <= 0) return i; }
    return V - 1;
  }

  // Generator that yields one new token id at a time.
  *generate(promptIds, nTokens, temperature = 0.8, topK = 40) {
    const ids = promptIds.length ? promptIds.slice() : [0];
    let logits;
    const prefill = (arr) => { this.reset(); for (const id of arr) logits = this.step(id); };
    prefill(ids.slice(-this.T));
    for (let i = 0; i < nTokens; i++) {
      const next = this.sample(logits, temperature, topK);
      ids.push(next);
      yield next;
      // when the context window is full, restart from the most recent half
      if (this.pos >= this.T) prefill(ids.slice(-Math.floor(this.T / 2)));
      else logits = this.step(next);
    }
  }
}

if (typeof module !== "undefined") module.exports = { GPTModel };
