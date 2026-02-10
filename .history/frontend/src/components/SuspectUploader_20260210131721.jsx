/**
 * 可疑文档上传组件：支持选择一个或多个 txt 文件，
 * 将文件内容读取后通过回调返回文本数组。
 *
 * props:
 *   onFilesChange(texts: string[]) — 读取完成后的文本回调
 */
export default function SuspectUploader({ onFilesChange }) {
  const handleChange = async (event) => {
    const files = Array.from(event.target.files || []);
    if (!files.length) return;

    const results = await Promise.all(
      files.map(
        (file) =>
          new Promise((resolve) => {
            const reader = new FileReader();
            reader.onload = () => resolve(String(reader.result || ""));
            reader.onerror = () => resolve("");
            reader.readAsText(file);
          }),
      ),
    );

    onFilesChange(results.filter((t) => t.trim()));
  };

  return (
    <div>
      <h2>批量疑似文档</h2>
      <p style={{ fontSize: 13, color: "#8a98ab", marginBottom: 8 }}>
        支持同时选择多个 txt 文件进行批量查重
      </p>
      <input type="file" multiple accept=".txt" onChange={handleChange} />
    </div>
  );
}
