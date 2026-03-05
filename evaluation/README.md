# Evaluation Script for [PAN 25 Generated Plagiarism Detection](https://pan.webis.de/clef25/pan25-web/generated-plagiarism-detection.html)

This directory contains the evaluation script for the [2025 edition of the generated plagiarism detection task](https://pan.webis.de/clef25/pan25-web/generated-plagiarism-detection.html).

You can either run the script directly, or you can run the dockerized version via the `tira` python package (install via `pip3 install tira`).

## Development

Build the docker image via:
```
docker build -t mam10eks/pan25-generated-plagiarism-detection-evaluator:0.0.1 .
```

Upload the docker image via:

```
docker push mam10eks/pan25-generated-plagiarism-detection-evaluator:0.0.1
```

# 在 spot-check (50对) 上快速测试
bash evaluation/run_eval.sh full spot-check
bash run_eval.sh full spot-check

# 在验证集 (7975对) 上评估
bash evaluation/run_eval.sh full validation

# 在训练集 (62159对) 上参数扫描
bash evaluation/run_eval.sh sweep train

