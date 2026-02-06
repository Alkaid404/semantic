import { useMemo, useRef, useState } from "react";

const MOCK_DELAY_MS = 1100;

const mockSegments = {
  source: [
    {
      text: "In this study we propose a scalable approach to ",
      highlight: false,
    },
    { text: "semantic similarity detection", highlight: true },
    {
      text: " using vector embeddings and approximate search.",
      highlight: false,
    },
  ],
  suspect: [
    { text: "This paper presents a scalable method for ", highlight: false },
    { text: "semantic similarity detection", highlight: true },
    { text: " with embeddings and fast retrieval.", highlight: false },
  ],
};

export default function App() {
  const [sourceText, setSourceText] = useState("");
  const [suspectText, setSuspectText] = useState("");
  const [sourceFile, setSourceFile] = useState(null);
  const [suspectFile, setSuspectFile] = useState(null);
  const [isChecking, setIsChecking] = useState(false);
  const [result, setResult] = useState(null);
  const resultRef = useRef(null);

  const canCheck = sourceText.trim() && suspectText.trim() && !isChecking;

  const score = useMemo(() => {
    if (!result) {
      return 0;
    }
    return result.score;
  }, [result]);

  const handleFilePick = (event, setter, fileSetter) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    fileSetter(file);
    const reader = new FileReader();
    reader.onload = () => {
      setter(String(reader.result || ""));
    };
    reader.readAsText(file);
  };

  const onCheck = () => {
    if (!canCheck) {
      return;
    }
    setIsChecking(true);
    setResult(null);
    setTimeout(() => {
      setResult({
        score: 78,
        sourceSegments: mockSegments.source,
        suspectSegments: mockSegments.suspect,
      });
      setIsChecking(false);
      resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }, MOCK_DELAY_MS);
  };

  return (
    <div className="page">
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
            <span className="metric-label">平均响应</span>
            <strong>1.2s</strong>
          </div>
          <div className="hero-metric">
            <span className="metric-label">高亮精度</span>
            <strong>92%</strong>
          </div>
        </div>
      </header>

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

      <section className="results" ref={resultRef}>
        <div className="result-header">
          <div>
            <h3>查重结果</h3>
            <p>展示总体相似度与高亮片段</p>
          </div>
          <div className="progress-ring" aria-hidden="true">
            <svg viewBox="0 0 120 120">
              <circle className="ring-bg" cx="60" cy="60" r="52" />
              <circle
                className="ring-value"
                cx="60"
                cy="60"
                r="52"
                style={{ strokeDashoffset: 327 - (327 * score) / 100 }}
              />
            </svg>
            <div className="ring-label">
              <strong>{result ? `${score}%` : "--"}</strong>
              <span>总查重率</span>
            </div>
          </div>
        </div>

        <div className="compare-grid">
          <article className="compare-card">
            <h4>源文档片段</h4>
            <p>
              {(result?.sourceSegments || []).map((segment, index) => (
                <span
                  key={`source-${index}`}
                  className={segment.highlight ? "highlight" : ""}
                >
                  {segment.text}
                </span>
              ))}
            </p>
            {!result && <p className="placeholder">等待检查结果...</p>}
          </article>
          <article className="compare-card">
            <h4>疑似文档片段</h4>
            <p>
              {(result?.suspectSegments || []).map((segment, index) => (
                <span
                  key={`suspect-${index}`}
                  className={segment.highlight ? "highlight" : ""}
                >
                  {segment.text}
                </span>
              ))}
            </p>
            {!result && <p className="placeholder">等待检查结果...</p>}
          </article>
        </div>
      </section>
    </div>
  );
}
