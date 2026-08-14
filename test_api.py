import base64
import requests


def test_routes(base_url="http://127.0.0.1:8000", image_path="./image/4.png"):
    image_b64 = base64.b64encode(open(image_path, "rb").read()).decode()

    routes = [
        {"method": "get", "path": "/status", "json": None},
        {"method": "post", "path": "/ocr", "json": {"image": image_b64}},
        {
            "method": "post",
            "path": "/ocr",
            "json": {"image": image_b64, "color_filter": "green"},
        },
        {
            "method": "post",
            "path": "/ocr",
            "json": {
                "image": image_b64,
                "charset_range": "0123456789",
                "probability": True,
            },
        },
    ]

    for route in routes:
        url = base_url.rstrip("/") + route["path"]
        method = route["method"].lower()
        print("=" * 60)
        print(method.upper(), url)
        if method == "get":
            r = requests.get(url, timeout=30)
        else:
            r = requests.post(url, json=route["json"], timeout=30)
        print("status:", r.status_code)
        try:
            print(r.json())
        except Exception:
            print(r.text)


if __name__ == "__main__":
    test_routes()
