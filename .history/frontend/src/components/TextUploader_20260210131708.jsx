import { useRef } from "react";

/**
 * 通用文本上传组件：支持粘贴文本 + 选择 txt 文件。
 *
 * props:
 *   value        — 当前文本内容
 *   onChange     — 文本变化回调
 *   onFileChange — 选中的 File 对象回调（可选）
 *   label        — 标题
 *   placeholder  — 占位提示
 */
export default function TextUploader({
  value,
  onChange,
  onFileChange,
  label = "文本",
  placeholder = "在这里粘贴文本内容...",
}) {
  const fileRef = useRef(null);

  const handleFile = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    onFileChange?.(file);
    const reader = new FileReader();
    reader.onload = () => onChange(String(reader.result || ""));
    reader.readAsText(file);
  };

  return (
    <article className="card">
      <div className="card-header">
        <div>
          <h2>{label}</h2>
          <p>粘贴文本或导入 txt 文件</p>
        </div>
        <label className="file-pill">
          选择文件
          <input
            ref={fileRef}
            type="file"
            accept=".txt"
            onChange={handleFile}
          />
        </label>
      </div>
      <textarea
        rows={12}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
      />
    </article>
  );
}
