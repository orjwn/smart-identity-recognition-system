function parseTerminalId(raw?: string): number {
  if (!raw) return 1;
  const m = String(raw).match(/(\d+)/);
  if (!m) return 1;
  const n = parseInt(m[1], 10);
  return Number.isFinite(n) && n > 0 ? Math.min(n, 9) : 1;
}

function parseGate(raw: string): { letter: string; num: number; label: string } {
  const s = raw.trim().toUpperCase();
  const m = s.match(/^([A-Z]+)?(\d+)$/);
  if (m) {
    const letter = (m[1] || "A").slice(-1);
    const num = parseInt(m[2], 10) || 1;
    return { letter, num, label: raw.trim() };
  }
  const m2 = s.match(/^([A-Z])/);
  const letter = m2 ? m2[1] : "A";
  const num = parseInt(s.replace(/\D/g, ""), 10) || 1;
  return { letter, num, label: raw.trim() || "?" };
}

const CONCOURSE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

export function concourseIndex(letter: string): number {
  const i = CONCOURSE_LETTERS.indexOf(letter.toUpperCase());
  return i < 0 ? 0 : i;
}

export type MapLayout = {
  start: { x: number; y: number };
  end: { x: number; y: number };
  pathD: string;
  terminalNum: number;
  concourse: string;
  gateNum: number;
  etaMinutes: number;
};

export function computeMapLayout(terminal?: string, gate?: string): MapLayout {
  const g = gate?.trim() || "A1";
  const { letter, num } = parseGate(g);
  const terminalNum = parseTerminalId(terminal);

  const ci = concourseIndex(letter);
  const maxConc = 12;
  const cNorm = Math.min(ci, maxConc) / maxConc;
  const xGate = 118 + cNorm * 232;

  const nClamped = Math.min(Math.max(num, 1), 99);
  const depth = (nClamped - 1) / 98;
  const yGate = 52 + (1 - depth) * 118;

  const tSlot = (terminalNum - 1) % 5;
  const startX = 32 + tSlot * 28;
  const startY = 270;

  const hallY = 222;
  const spineX = 85 + (terminalNum % 3) * 12;
  const joinX = Math.min(Math.max(xGate - 40, spineX + 30), xGate - 15);
  const midY = 125 + (ci % 4) * 8;

  const pathD = [
    `M ${startX} ${startY}`,
    `L ${startX} ${hallY}`,
    `L ${joinX} ${hallY}`,
    `L ${joinX} ${midY}`,
    `L ${xGate} ${midY}`,
    `L ${xGate} ${yGate}`,
  ].join(" ");

  const roughWalk =
    Math.abs(startX - joinX) +
    Math.abs(hallY - midY) +
    Math.abs(joinX - xGate) +
    Math.abs(midY - yGate) +
    Math.abs(yGate - startY) * 0.15;
  const etaMinutes = Math.max(3, Math.min(16, Math.round(roughWalk / 42)));

  return {
    start: { x: startX, y: startY },
    end: { x: xGate, y: yGate },
    pathD,
    terminalNum,
    concourse: letter,
    gateNum: num,
    etaMinutes,
  };
}
