# ddddocr（魔改 / OCR-only）

本仓库为 [86maid/ddddocr](https://github.com/86maid/ddddocr) 的精简 fork（Demo11101）。

## 变更

- **仅适配 x86_64 Linux**；监听地址硬编码为 `0.0.0.0:8000`
- **移除** MCP、`old` 旧模型、`slide` 滑块、`det` 目标检测
- **默认开启 OCR**（可用 `--no-ocr` 关闭，关闭后进程直接退出）
- GitHub Actions CI：任意 push / PR 自动编译 `x86_64-unknown-linux-gnu`

## 本地编译（Linux x86_64）

```bash
# 内联模型（默认 features）
cargo build --release --target x86_64-unknown-linux-gnu

# 外置 model/ 目录
cargo build --release --target x86_64-unknown-linux-gnu \
  --no-default-features --features download-binaries
```

## 运行

```bash
./ddddocr
# 固定监听 http://0.0.0.0:8000 ；Swagger: /swagger-ui
```

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/status` | 服务状态 |
| POST | `/ocr` | 内容识别（image base64） |

详见 `test_api.py`。

## GitHub Actions

任意 `push` / `pull_request` / 手动 `workflow_dispatch` 触发 CI，仅构建并上传 `x86_64-unknown-linux-gnu` artifact。
