# ddddocr（魔改 / OCR-only）

本仓库为 [86maid/ddddocr](https://github.com/86maid/ddddocr) 的精简 fork（Demo11101）。

## 变更

- **仅适配 x86_64 Linux** 构建与发布
- **移除** MCP、`old` 旧模型、`slide` 滑块、`det` 目标检测
- **默认开启 OCR**（可用 `--no-ocr` 关闭，关闭后进程直接退出）
- GitHub Actions：`.github/workflows/build.yml` 仅编译 `x86_64-unknown-linux-gnu`

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
./ddddocr --address 0.0.0.0:8000
# OCR 默认开启；Swagger: /swagger-ui
```

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/status` | 服务状态 |
| POST | `/ocr` | 内容识别（image base64） |

详见 `test_api.py`。

## GitHub Actions

- `push` / `pull_request` → 自动构建并上传 artifact  
- `workflow_dispatch` 且 `publish=true` → 创建 Release 并挂载 zip
