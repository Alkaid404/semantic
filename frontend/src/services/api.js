import axios from "axios";

// 职责：
// - 调用后端查重接口
export async function checkPlagiarism(data) {
  return axios.post("/check", data);
}
