import axios from "axios";

const api = axios.create({
  baseURL: "/api",
  timeout: 300_000, // 5 分钟，给长文档 + IDF 加权编码留足余量
});

/**
 * 文本查重 — 对应后端 POST /check
 *
 * @param {string} sourceText  源文档文本
 * @param {string} suspectText 可疑文档文本
 * @returns {Promise<{
 *   similarity: number,
 *   matched_char_ratio: number,
 *   matches: Array<{
 *     source_chunk: string,
 *     suspect_chunk: string,
 *     score: number,
 *     cosine_score: number,
 *     source_offset: number,
 *     source_length: number,
 *     suspect_offset: number,
 *     suspect_length: number,
 *   }>
 * }>}
 */
export async function checkPlagiarism(sourceText, suspectText) {
  const { data } = await api.post("/check", {
    source_text: sourceText,
    suspect_text: suspectText,
  });
  return data;
}

/**
 * 文件上传查重 — 对应后端 POST /check/files
 *
 * @param {File} sourceFile  源文档文件
 * @param {File} suspectFile 可疑文档文件
 * @returns {Promise<object>} 同 checkPlagiarism
 */
export async function checkPlagiarismFiles(sourceFile, suspectFile) {
  const form = new FormData();
  form.append("source_file", sourceFile);
  form.append("suspect_file", suspectFile);
  const { data } = await api.post("/check/files", form);
  return data;
}

/**
 * 文件上传查重并下载 PAN XML — 对应后端 POST /check/xml
 */
export async function checkPlagiarismXml(sourceFile, suspectFile) {
  const form = new FormData();
  form.append("source_file", sourceFile);
  form.append("suspect_file", suspectFile);
  const { data } = await api.post("/check/xml", form, {
    responseType: "text",
  });
  return data;
}

/**
 * 根据后端返回的 matches（含 offset/length），将原文切成
 * 带 highlight 标记的片段数组，用于高亮渲染。
 *
 * @param {string} text       原始全文
 * @param {Array}  matches    后端返回的 matches
 * @param {"source"|"suspect"} side  处理哪一侧
 * @returns {Array<{ text: string, highlight: boolean, score?: number }>}
 */
export function buildHighlightSegments(text, matches, side) {
  if (!text || !matches?.length) {
    return [{ text, highlight: false }];
  }

  // 收集匹配区间
  const offsetKey = side === "source" ? "source_offset" : "suspect_offset";
  const lengthKey = side === "source" ? "source_length" : "suspect_length";

  const ranges = matches
    .map((m) => ({
      start: m[offsetKey],
      end: m[offsetKey] + m[lengthKey],
      score: m.score,
    }))
    .filter((r) => r.start >= 0 && r.end > r.start && r.start < text.length)
    .sort((a, b) => a.start - b.start);

  // 合并重叠区间
  const merged = [];
  for (const r of ranges) {
    if (merged.length && r.start <= merged[merged.length - 1].end) {
      const last = merged[merged.length - 1];
      last.end = Math.max(last.end, r.end);
      last.score = Math.max(last.score, r.score);
    } else {
      merged.push({ ...r });
    }
  }

  // 切成片段
  const segments = [];
  let cursor = 0;
  for (const { start, end, score } of merged) {
    const s = Math.min(start, text.length);
    const e = Math.min(end, text.length);
    if (cursor < s) {
      segments.push({ text: text.slice(cursor, s), highlight: false });
    }
    segments.push({ text: text.slice(s, e), highlight: true, score });
    cursor = e;
  }
  if (cursor < text.length) {
    segments.push({ text: text.slice(cursor), highlight: false });
  }
  return segments;
}
