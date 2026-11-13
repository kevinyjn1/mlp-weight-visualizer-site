// A small renderer for the Markdown the models wrote: headings, paragraphs, lists, tables, code, and enough of
// LaTeX's inline math to read their formulas. Everything is escaped first; only the tags added here reach the page.

export const escapeHtml = (s) => s.replace(/[&<>"]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[ch]);

const SYMBOLS = {
  times: '×', cdot: '·', pm: '±', le: '≤', leq: '≤', ge: '≥', geq: '≥', ldots: '…', dots: '…', sum: 'Σ', ell: 'ℓ',
  to: '→', rightarrow: '→', approx: '≈', neq: '≠', ne: '≠', infty: '∞', alpha: 'α', beta: 'β', sigma: 'σ', mu: 'μ',
  Delta: 'Δ', delta: 'δ', max: 'max', min: 'min', quad: ' ', qquad: ' ', mid: '|',
};

/** LaTeX-ish math to readable HTML: \operatorname{ReLU}(g) -> ReLU(g), a_{L,0} -> a<sub>L,0</sub>, and so on. */
export function math(tex) {
  let s = escapeHtml(tex);
  s = s.replace(/\\begin\{[bp]?matrix\}([\s\S]*?)\\end\{[bp]?matrix\}/g, (_, body) =>
    `[${body.trim().split(/\\\\/).map((row) => row.trim().split('&amp;').map((c) => c.trim()).join(', ')).join('; ')}]`);
  s = s.replace(/\\(?:operatorname|mathrm|text|mathbf|mathit)\{([^{}]*)\}/g, '$1');
  s = s.replace(/\\frac\{([^{}]*)\}\{([^{}]*)\}/g, '($1)/($2)');
  s = s.replace(/\\(?:bigl|bigr|Bigl|Bigr|left|right|displaystyle)\b/g, '');
  s = s.replace(/\\[,;:!]/g, ' ').replace(/\\ /g, ' ').replace(/\\%/g, '%').replace(/\\([{}])/g, '$1');
  s = s.replace(/\\([A-Za-z]+)/g, (_, name) => SYMBOLS[name] ?? name);
  s = s.replace(/_\{([^{}]*)\}/g, '<sub>$1</sub>').replace(/_([A-Za-z0-9])/g, '<sub>$1</sub>');
  s = s.replace(/\^\{([^{}]*)\}/g, '<sup>$1</sup>').replace(/\^([A-Za-z0-9+\-*])/g, '<sup>$1</sup>');
  return s.replace(/\s+/g, ' ').trim();
}

/** Inline Markdown: `code`, \( math \), **bold**, *italic*. */
export function inline(text) {
  const html = text
    .split(/(`[^`]*`|\\\([\s\S]*?\\\))/g)
    .map((part) => {
      if (part.startsWith('`') && part.length > 1) return `<code>${escapeHtml(part.slice(1, -1)).replaceAll('*', '&#42;')}</code>`;
      if (part.startsWith('\\(')) return `<span class="math">${math(part.slice(2, -2)).replaceAll('*', '&#42;')}</span>`;
      return escapeHtml(part)
        .replaceAll(String.fromCharCode(8212), ' - ')
        .replace(/(^|[\s(])\*([^*\s][^*]*?)\*(?=[\s).,;:]|$)/g, '$1<i>$2</i>');
    })
    .join('');
  // bold may wrap code or math, so its markers are paired over the whole line
  let open = false;
  const out = html.replace(/\*\*/g, () => ((open = !open) ? '<b>' : '</b>'));
  return open ? `${out}</b>` : out;
}

/** Split a table row on the bars that are not inside code or math. */
function cells(line) {
  const out = [];
  let cell = '';
  let code = false;
  let depth = 0;
  for (let k = 0; k < line.length; k++) {
    const two = line.slice(k, k + 2);
    if (line[k] === '`') code = !code;
    else if (!code && two === '\\(') depth++;
    else if (!code && two === '\\)') depth = Math.max(depth - 1, 0);
    if (line[k] === '|' && !code && depth === 0) {
      out.push(cell);
      cell = '';
    } else {
      cell += line[k];
    }
  }
  out.push(cell);
  return out.slice(1, out[out.length - 1].trim() ? undefined : -1).map((c) => c.trim());
}

export function markdown(text) {
  const lines = text.replace(/\r/g, '').split('\n');
  const html = [];
  let k = 0;
  const isRule = (line) => /^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$/.test(line ?? '');
  while (k < lines.length) {
    const line = lines[k];
    if (!line.trim()) {
      k++;
    } else if (/^```/.test(line)) {
      const body = [];
      for (k++; k < lines.length && !/^```/.test(lines[k]); k++) body.push(lines[k]);
      k++;
      html.push(`<pre class="code">${escapeHtml(body.join('\n'))}</pre>`);
    } else if (/^#{1,5}\s/.test(line)) {
      const level = Math.min(line.match(/^#+/)[0].length + 2, 6);
      html.push(`<h${level}>${inline(line.replace(/^#+\s*/, ''))}</h${level}>`);
      k++;
    } else if (line.trim().startsWith('\\[')) {
      const body = [];
      for (; k < lines.length; k++) {
        body.push(lines[k]);
        if (lines[k].trim().endsWith('\\]')) break;
      }
      k++;
      html.push(`<p class="math block">${math(body.join(' ').trim().replace(/^\\\[/, '').replace(/\\\]$/, ''))}</p>`);
    } else if (line.startsWith('|') && isRule(lines[k + 1])) {
      const head = cells(line);
      const rows = [];
      for (k += 2; k < lines.length && lines[k].startsWith('|'); k++) {
        if (!isRule(lines[k])) rows.push(cells(lines[k]));
      }
      html.push(`<div class="table"><table><thead><tr>${head.map((c) => `<th>${inline(c)}</th>`).join('')}</tr></thead><tbody>${rows
        .map((r) => `<tr>${r.map((c) => `<td>${inline(c)}</td>`).join('')}</tr>`)
        .join('')}</tbody></table></div>`);
    } else if (/^\s*(?:[-*]|\d+\.)\s+/.test(line)) {
      const ordered = /^\s*\d+\./.test(line);
      const items = [];
      for (; k < lines.length && lines[k].trim() && !/^#{1,5}\s|^\||^```/.test(lines[k]); k++) {
        if (/^\s*(?:[-*]|\d+\.)\s+/.test(lines[k])) items.push(lines[k].replace(/^\s*(?:[-*]|\d+\.)\s+/, ''));
        else items[items.length - 1] += ` ${lines[k].trim()}`;
      }
      html.push(`<${ordered ? 'ol' : 'ul'}>${items.map((it) => `<li>${inline(it)}</li>`).join('')}</${ordered ? 'ol' : 'ul'}>`);
    } else {
      const body = [];
      for (; k < lines.length && lines[k].trim() && !/^#{1,5}\s|^\||^```|^\s*\\\[|^\s*(?:[-*]|\d+\.)\s+/.test(lines[k]); k++) body.push(lines[k].trim());
      html.push(`<p>${inline(body.join(' '))}</p>`);
    }
  }
  return html.join('');
}

/**
 * A hard-wrapped plain-text prompt as HTML: prose is reflowed, aligned tables and code keep their layout.
 * `marks` is a list of {id, find}: the first place `find` occurs (whitespace ignored) is wrapped in <mark id>.
 */
export function plainText(text, marks = []) {
  let html = text
    .replace(/\r/g, '')
    .split(/\n\s*\n/)
    .map((block) => {
      const lines = block.split('\n');
      if (lines[0].startsWith('```')) return `<pre class="code">${escapeHtml(lines.slice(1, lines.at(-1).startsWith('```') ? -1 : undefined).join('\n'))}</pre>`;
      const table = lines.filter((l) => /\S {2,}\S/.test(l) || /^\|/.test(l)).length * 2 >= lines.length && lines.length > 1;
      if (table) return `<pre class="code">${escapeHtml(block)}</pre>`;
      const parts = [];
      for (const l of lines) {
        if (/^\s*(?:-|\d+\.)\s/.test(l) || /^\s{2,}\d+ /.test(l) || !parts.length) parts.push(l.trim());
        else parts[parts.length - 1] += ` ${l.trim()}`;
      }
      const first = escapeHtml(parts[0]).replace(/^([A-Z][A-Z' -]{4,}[A-Z])([.:])/, '<b class="cap">$1$2</b>').replace(/^#+\s*(.*)$/, '<b class="cap">$1</b>');
      return `<p>${[first, ...parts.slice(1).map(escapeHtml)].join('<br>')}</p>`;
    })
    .join('');
  for (const { id, find } of marks) {
    const pattern = new RegExp(escapeHtml(find).replace(/[.*+?^${}()|[\]\\]/g, '\\$&').replace(/\s+/g, '(?:\\s|<br>)+'));
    html = html.replace(pattern, (m) => `<mark id="${id}">${m}</mark>`);
  }
  return html;
}
