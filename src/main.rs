use base64::prelude::*;
use clap::Parser;
use ddddocr::*;
use enable_ansi_support::enable_ansi_support;
use lru::LruCache;
use salvo::catcher::Catcher;
use salvo::http::ResBody;
use salvo::oapi::extract::JsonBody;
use salvo::prelude::*;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use std::num::NonZero;
use std::sync::LazyLock;
use std::sync::OnceLock;
use tokio::sync::Mutex;
use tokio::task::spawn_blocking;
use tracing::debug;
use tracing::info;
use tracing_subscriber::EnvFilter;

static ARGS: OnceLock<Args> = OnceLock::new();
static OCR: OnceLock<Ddddocr> = OnceLock::new();
static CACHE: LazyLock<Mutex<LruCache<String, Vec<String>>>> =
    LazyLock::new(|| Mutex::new(LruCache::new(NonZero::new(64).unwrap())));

/// 固定监听地址。
const LISTEN_ADDR: &str = "0.0.0.0:8000";

#[derive(Parser, Debug, Clone)]
struct Args {
    /// 关闭 OCR（默认开启）。
    #[arg(long, default_value_t = false)]
    no_ocr: bool,

    /// 全局默认字符集，用于概率识别，
    /// 如果 API 未提供字符集，则使用此参数，
    /// 当值为 0~7 时，表示选择内置字符集，
    /// 其他值表示自定义字符集，例如 "0123456789+-x/="，
    /// 如果未设置，则使用完整字符集，不做限制。
    #[arg(long)]
    ocr_charset_range: Option<String>,

    /// 内容识别模型以及字符集路径，
    /// 开启 features inline-model（默认）时可不传；自定义模型时模型与 json 同名。
    #[arg(long, default_value_t = { "model/common.onnx".to_string() })]
    ocr_path: String,

    /// 输入域名以自动获取 SSL 证书（HTTPS）。
    #[arg(long)]
    acme: Option<String>,
}

impl Args {
    fn ocr_enabled(&self) -> bool {
        !self.no_ocr
    }
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
struct OCRRequest {
    /// 要进行识别的图片，base64 编码。
    image: String,

    /// 如果 png_fix 为 true，则支持透明黑色背景的 png 图片。
    png_fix: Option<bool>,

    /// 是否返回概率信息。
    probability: Option<bool>,

    /// 限定字符范围，只对本次 ocr 生效。
    charset_range: Option<String>,

    /// 颜色过滤，例如 `red` 或 `["red", "blue"]` 或 `[[[0, 50, 50], [10, 255, 255]]]`。
    color_filter: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
struct OCRResponse {
    text: String,
    probability: Option<Vec<Vec<f32>>>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
struct StatusResponse {
    service_status: String,
    enabled_features: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
struct APIResponse<T> {
    code: u16,
    msg: String,
    data: Option<T>,
}

#[endpoint(responses((status_code = 200, body = APIResponse<OCRResponse>)))]
async fn route_ocr(req: JsonBody<OCRRequest>, res: &mut Response) -> anyhow::Result<()> {
    let image = BASE64_STANDARD.decode(&req.image)?;
    let png_fix = req.png_fix.unwrap_or_default();
    let probability = req.probability.unwrap_or_default();
    let color_filter = if let Some(v) = req.color_filter.clone() {
        Some(serde_json::from_value::<ColorFilter>(v)?)
    } else {
        None
    };

    let charset_range = if let Some(ref v) = req.charset_range {
        let ocr_charset_range = match v.as_str() {
            "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" => {
                CharsetRange::from(v.parse::<i32>().unwrap())
            }
            v => CharsetRange::from(v),
        };

        Some(CharsetRange::Charset(
            CACHE
                .lock()
                .await
                .get_or_insert(v.to_string(), || {
                    OCR.get().unwrap().calc_ranges(ocr_charset_range)
                })
                .clone(),
        ))
    } else {
        None
    };

    let (text, probability) = if charset_range.is_some() || probability {
        let mut result = spawn_blocking({
            let color_filter = color_filter.clone();
            let charset_range = charset_range.clone();

            move || {
                OCR.get().unwrap().classification_probability_with_options(
                    image,
                    png_fix,
                    color_filter,
                    charset_range,
                )
            }
        })
        .await??;

        (
            result.get_text().to_string(),
            probability.then_some(result.probability),
        )
    } else {
        let text = spawn_blocking({
            let color_filter = color_filter.clone();

            move || {
                OCR.get()
                    .unwrap()
                    .classification_with_options(image, png_fix, color_filter)
            }
        })
        .await??;

        (text, None)
    };

    let response = APIResponse {
        code: 200,
        msg: "success".to_string(),
        data: Some(OCRResponse { text, probability }),
    };

    debug!(
        "ocr response: {}",
        serde_json::to_string_pretty(&response).unwrap()
    );

    res.render(Json(response));
    Ok(())
}

#[endpoint(responses((status_code = 200, body = APIResponse<StatusResponse>)))]
async fn route_status(res: &mut Response) {
    let args = ARGS.get().unwrap();
    let mut enabled_features = Vec::new();
    if args.ocr_enabled() {
        enabled_features.push("ocr".to_string());
    }

    let response = APIResponse {
        code: 200,
        msg: "success".to_string(),
        data: Some(StatusResponse {
            service_status: "running".to_string(),
            enabled_features,
        }),
    };

    debug!(
        "status response: {}",
        serde_json::to_string_pretty(&response).unwrap()
    );

    res.render(Json(response));
}

#[handler]
fn default_error_handler(res: &mut Response) {
    if let ResBody::Error(v) = &res.body {
        res.render(Json(APIResponse {
            code: v.code.as_u16(),
            msg: v.to_string(),
            data: <Option<String>>::None,
        }));
    } else {
        res.render(Text::Plain(""));
    }
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    tracing_subscriber::fmt()
        .with_ansi(enable_ansi_support().is_ok())
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    if !args.ocr_enabled() {
        tracing::error!("OCR is disabled (--no-ocr); nothing to serve");
        std::process::exit(1);
    }

    ARGS.set(args.clone()).unwrap();
    init_ocr(&args);
    info!("ocr enabled by default");

    let mut router = Router::new()
        .hoop(salvo::prelude::Logger::new())
        .push(Router::with_path("/status").get(route_status))
        .push(Router::with_path("/ocr").post(route_ocr));

    let doc = OpenApi::new("DdddOcr API", env!("CARGO_PKG_VERSION")).merge_router(&router);
    router = router
        .unshift(doc.into_router("/api-doc/openapi.json"))
        .unshift(SwaggerUi::new("/api-doc/openapi.json").into_router("/swagger-ui"));

    router = router.catcher(Catcher::default().hoop(default_error_handler));

    let acceptor = if let Some(domain) = &args.acme {
        TcpListener::new(LISTEN_ADDR)
            .acme()
            .cache_path("temp/letsencrypt")
            .add_domain(domain)
            .quinn(LISTEN_ADDR)
            .bind()
            .await
    } else {
        TcpListener::new(LISTEN_ADDR).bind().await
    };

    info!("listening on http://{}", LISTEN_ADDR);
    Server::new(acceptor).serve(router).await;
}

fn ocr_charset_range(args: &Args) -> Option<CharsetRange> {
    args.ocr_charset_range
        .as_ref()
        .map(|v| match v.as_str() {
            "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" => {
                CharsetRange::from(v.parse::<i32>().unwrap())
            }
            v => CharsetRange::from(v),
        })
}

#[cfg(feature = "inline-model")]
fn init_ocr(args: &Args) {
    let mut ddddocr = ddddocr_classification().expect("failed to load OCR model");
    if let Some(v) = ocr_charset_range(args) {
        ddddocr.set_ranges(v);
    }
    OCR.set(ddddocr).unwrap();
}

#[cfg(not(feature = "inline-model"))]
fn init_ocr(args: &Args) {
    use std::fs::read;
    use std::path::Path;

    let path = Path::new(&args.ocr_path);
    let model = read(path).expect("failed to open the ocr model file");
    let charset_path = path.with_extension("json");
    let charset = read(&charset_path).expect("failed to open the ocr charset file");
    let charset: Charset =
        serde_json::from_slice(&charset).expect("failed to parse the ocr charset file");
    let mut ddddocr = Ddddocr::new(model, charset).expect("failed to create OCR instance");
    if let Some(v) = ocr_charset_range(args) {
        ddddocr.set_ranges(v);
    }
    OCR.set(ddddocr).unwrap();
}
