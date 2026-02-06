export default function TextUploader({ value, onChange }) {
  // 职责：
  // - 输入原始文本
  // - 如有需要上传 txt
  return (
    <div>
      <h2>Source Text</h2>
      <textarea
        rows={10}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder="Paste source text here"
        style={{ width: "100%" }}
      />
    </div>
  );
}
