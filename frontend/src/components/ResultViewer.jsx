export default function ResultViewer({ result }) {
  // 职责：
  // - 高亮重复片段
  // - 显示相似度
  if (!result) {
    return null;
  }

  return (
    <div style={{ marginTop: "16px" }}>
      <h2>Result</h2>
      <div>Similarity: {result.similarity}</div>
      <pre style={{ whiteSpace: "pre-wrap" }}>
        {JSON.stringify(result.matches, null, 2)}
      </pre>
    </div>
  );
}
