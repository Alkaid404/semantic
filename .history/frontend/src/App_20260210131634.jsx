import { useMemo, useRef, useState } from "react";
import { checkPlagiarism, buildHighlightSegments } from "./services/api";

export default function App() {
  const [sourceText, setSourceText] = useState("");
  const [suspectText, setSuspectText] = useState("");
  const [sourceFile, setSourceFile] = useState(null);
  const [suspectFile, setSuspectFile] = useState(null);
  const [isChecking, setIsChecking] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const resultRef = useRef(null);

  const canCheck = sourceText.trim() && suspectText.trim() && !isChecking;

  // --- 后端返回的百分制得分 ---
  const scorePercent = useMemo(() => {
    if (!result) return 0;
    return Math.round(result.similarity * 100);
  }, [result]);

  const charRatioPercent = useMemo(() => {
    if (!result) return 0;
    return Math.round(result.matched_char_ratio * 100);
  }, [result]);

  // --- 基于 offset 构建高亮片段 ---
  const sourceSegments = useMemo(
    () =>
      result
        ? buildHighlightSegments(sourceText, result.matches, "source")
        : [],
    [result, sourceText]
  );

  const suspectSegments = useMemo(
    () =>
      result
        ? buildHighlightSegments(suspectText, result.matches, "suspect")
        : [],
    [result, suspectText]
  );

  // --- 文件选择 ---
  const handleFilePick = (event, setter, fileSetter) => {
    const file = event.target.files?.[0];
    if (!file) return;
    fileSetter(file);
    const reader = new FileReader();
    reader.onload = () => setter(String(reader.result || ""));
    reader.readAsText(file);
  };

  // --- 发起检测 ---
  const onCheck = async () => {
    if (!canCheck) return;
    setIsChecking(true);
    setResult(null);
    setError(null);

    try {
      const data = await checkPlagiarism(sourceText, suspectText);
      setResult(data);
      setTimeout(() => {
        resultRef.current?.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      }, 100);
    } catch (err) {
      const msg =
        err.response?.data?.detail ||
        err.message ||
        "检测失败，请检查后端服务是否启动";
      setError(msg);
    } finally {
      setIsChecking(false);
    }
  };

  return (
    <div className="page">
      {/* ---------- Hero 头部 ---------- */}
      <header className="hero">
        <div>
          <p className="eyebrow">Document Intelligence</p>
          <h1>文档深度查重</h1>
          <p className="subtitle">
            对比文本语义与结构相似度，快速定位重复片段。
          </p>
        </div>
        <div className="hero-card">
          <div className="hero-metric">
            <span className="metric-label">匹配段落</span>
            <strong>{result ? result.matches.length : "--"}</strong>
          </div>
          <div className="hero-metric">
            <span className="metric-label">字符覆盖</span>
            <strong>{result ? `${charRatioPercent}%` : "--"}</strong>
          </div>
        </div>
      </header>

      {/* ---------- 输入区 ---------- */}
      <section className="input-grid">
        <article className="card">
          <div className="card-header">
            <div>
              <h2>源文件 (Original TXT)</h2>
              <p>粘贴文本或导入 txt 文件</p>
            </div>
            <label className="file-pill">
              {sourceFile ? sourceFile.name : "选择文件"}
              <input
                type="file"
                accept=".txt"
                onChange={(event) =>
                  handleFilePick(event, setSourceText, setSourceFile)
                }
              />
            </label>
          </div>
          <textarea
            rows={12}
            value={sourceText}
            onChange={(event) => setSourceText(event.target.value)}
            placeholder="在这里粘贴源文档内容..."
          />
        </article>

        <article className="card">
          <div className="card-header">
            <div>
              <h2>怀疑文件 (Suspect TXT)</h2>
              <p>支持输入文本或导入文件对比</p>
            </div>
            <label className="file-pill">
              {suspectFile ? suspectFile.name : "选择文件"}
              <input
                type="file"
                accept=".txt"
                onChange={(event) =>
                  handleFilePick(event, setSuspectText, setSuspectFile)
                }
              />
            </label>
          </div>
          <textarea
            rows={12}
            value={suspectText}
            onChange={(event) => setSuspectText(event.target.value)}
            placeholder="在这里粘贴疑似文档内容..."
          />
        </article>
      </section>

      {/* ---------- 按钮区 ---------- */}
      <section className="cta">
        <button
          type="button"
          className="primary-button"
          onClick={onCheck}
          disabled={!canCheck}
        >
          {isChecking ? (
            <span className="button-loader">
              <span className="spinner" />
              正在检查...
            </span>
          ) : (
            "开始检查"
          )}
        </button>
      </section>

      {/* ---------- 错误提示 ---------- */}
      {error && (
        <section className="error-banner">
          <p>⚠ {error}</p>
        </section>
      )}

      {/* ---------- 结果区 ---------- */}
      <section className="results" ref={resultRef}>
        <div className="result-header">
          <div>
            <h3>查重结果</h3>
            <p>
              {result
                ? `共发现 ${result.matches.length} 个相似段落`
                : "展示总体相似度与高亮片段"}
            </p>
          </div>
          <div className="progress-ring" aria-hidden="true">
            <svg viewBox="0 0 120 120">
              <circle className="ring-bg" cx="60" cy="60" r="52" />
              <circle
                className="ring-value"
                cx="60"
                cy="60"
                r="52"
                style={{
                  strokeDashoffset: 327 - (327 * scorePercent) / 100,
                }}
              />
            </svg>
            <div className="ring-label">
              <strong>{result ? `${scorePercent}%` : "--"}</strong>
              <span>总查重率</span>
            </div>
          </div>
        </div>

        {/* --- 全文高亮对比 --- */}
        <div className="compare-grid">
          <article className="compare-card">
            <h4>源文档片段</h4>
            <p>
              {sourceSegments.map((seg, i) => (
                <span
                  key={`s-${i}`}
                  className={seg.highlight ? "highlight" : ""}
                  title={
                    seg.highlight ? `相似度: ${(seg.score * 100).toFixed(0)}%` : undefined
                  }
                >
                  {seg.text}
                </span>
              ))}
            </p>
            {!result && <p className="placeholder">等待检查结果...</p>}
          </article>
          <article className="compare-card">
            <h4>疑似文档片段</h4>
            <p>
              {suspectSegments.map((seg, i) => (
                <span
                  key={`u-${i}`}
                  className={seg.highlight ? "highlight" : ""}
                  title={
                    seg.highlight ? `相似度: ${(seg.score * 100).toFixed(0)}%` : undefined
                  }
                >
                  {seg.text}
                </span>
              ))}
            </p>
            {!result && <p className="placeholder">等待检查结果...</p>}
          </article>
        </div>

        {/* --- 匹配段落明细表 --- */}
        {result && result.matches.length > 0 && (
          <div className="match-table-wrap">
            <h4>匹配段落明细</h4>
            <table className="match-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>源文档段落</th>
                  <th>可疑文档段落</th>
                  <th>余弦</th>
                  <th>Rerank</th>
                </tr>
              </thead>
              <tbody>
                {result.matches.map((m, i) => (
                  <tr key={i}>
                    <td>{i + 1}</td>
                    <td className="chunk-cell">{m.source_chunk}</td>
                    <td className="chunk-cell">{m.suspect_chunk}</td>
                    <td>{(m.cosine_score * 100).toFixed(1)}%</td>
                    <td>{(m.score * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
