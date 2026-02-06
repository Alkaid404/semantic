export default function SuspectUploader({ onFilesChange }) {
  // 职责：
  // - 支持多个疑似文档
  const handleChange = (event) => {
    const files = Array.from(event.target.files || []);
    const texts = files.map((file) => file.name);
    onFilesChange(texts);
  };

  return (
    <div>
      <h2>Suspect Documents</h2>
      <input type="file" multiple onChange={handleChange} />
    </div>
  );
}
