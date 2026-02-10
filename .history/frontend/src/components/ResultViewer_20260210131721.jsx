/**
 * 查重结果查看器
 *
 * props.result — 后端 PlagiarismResponse:
 *   { similarity, matched_char_ratio, matches[] }
 */
export default function ResultViewer({ result }) {
  if (!result) {
    return null;
  }

  const scorePercent = Math.round(result.similarity * 100);
  const charPercent = Math.round(result.matched_char_ratio * 100);

  return (
    <div className="result-viewer">
      <h3>查重报告</h3>
      <div className="result-metrics">
        <div>
          <span className="metric-label">相似度</span>
          <strong>{scorePercent}%</strong>
        </div>
        <div>
          <span className="metric-label">字符覆盖率</span>
          <strong>{charPercent}%</strong>
        </div>
        <div>
          <span className="metric-label">匹配段落数</span>
          <strong>{result.matches.length}</strong>
        </div>
      </div>

      {result.matches.length > 0 && (
        <div className="match-list">
          {result.matches.map((m, i) => (
            <div key={i} className="match-item">
              <div className="match-header">
                <span>#{i + 1}</span>
                <span>余弦: {(m.cosine_score * 100).toFixed(1)}%</span>
                <span>Rerank: {(m.score * 100).toFixed(1)}%</span>
              </div>
              <div className="match-body">
                <div>
                  <strong>源文档</strong>
                  <p>{m.source_chunk}</p>
                </div>
                <div>
                  <strong>可疑文档</strong>
                  <p>{m.suspect_chunk}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
