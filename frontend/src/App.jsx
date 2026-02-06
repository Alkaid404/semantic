import { useState } from "react";

import { checkPlagiarism } from "./services/api";
import TextUploader from "./components/TextUploader";
import SuspectUploader from "./components/SuspectUploader";
import ResultViewer from "./components/ResultViewer";

export default function App() {
  const [sourceText, setSourceText] = useState("");
  const [suspects, setSuspects] = useState([]);
  const [result, setResult] = useState(null);

  const onCheck = async () => {
    const payload = {
      source_text: sourceText,
      suspects,
    };
    const response = await checkPlagiarism(payload);
    setResult(response.data);
  };

  return (
    <div style={{ padding: "24px" }}>
      <h1>Semantic Plagiarism Detector</h1>
      <div
        style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}
      >
        <TextUploader value={sourceText} onChange={setSourceText} />
        <SuspectUploader onFilesChange={setSuspects} />
      </div>
      <button type="button" onClick={onCheck} style={{ marginTop: "16px" }}>
        Check
      </button>
      <ResultViewer result={result} />
    </div>
  );
}
