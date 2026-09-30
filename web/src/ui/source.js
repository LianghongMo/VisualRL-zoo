// Pull one function out of a Python file, so the "Code" step shows the code that really runs.
export function extractDef(source, name, { cutAt, replacement } = {}) {
  const lines = source.split("\n");
  const start = lines.findIndex((l) => l.trimStart().startsWith(`def ${name}(`));
  if (start < 0) throw new Error(`def ${name} not found`);
  const indent = lines[start].search(/\S/);
  let end = start + 1;
  while (end < lines.length && (lines[end].trim() === "" || lines[end].search(/\S/) > indent)) end++;
  let body = lines.slice(start, end).map((l) => l.slice(indent));
  if (cutAt) {
    const i = body.findIndex((l) => l.includes(cutAt));
    if (i >= 0) body = [...body.slice(0, i), ...(replacement ? [body[i].match(/^\s*/)[0] + replacement] : [])];
  }
  while (body.length && body[body.length - 1].trim() === "") body.pop();
  return body.join("\n");
}

export const lineOf = (code, needle) => code.split("\n").findIndex((l) => l.includes(needle)) + 1;
