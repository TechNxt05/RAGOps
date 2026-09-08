"""
Central HTTPX Client Patch for Groq & LangChain Compatibility.
Ensures that legacy 'proxies' dictionary kwargs are safely converted to 'proxy'
and TestClient 'app' kwargs are handled cleanly across all HTTPX instances.
"""

import httpx

def _clean_httpx_kwargs(kwargs: dict):
    if "proxies" in kwargs:
        p = kwargs.pop("proxies")
        if p and "proxy" not in kwargs:
            if isinstance(p, dict):
                kwargs["proxy"] = p.get("http://") or p.get("https://") or p.get("all://")
            else:
                kwargs["proxy"] = p
    if "app" in kwargs:
        app_arg = kwargs.pop("app")
        if "transport" not in kwargs:
            try:
                from httpx import ASGITransport
                kwargs["transport"] = ASGITransport(app=app_arg)
            except Exception:
                pass

_orig_client_init = httpx.Client.__init__
def _patched_client_init(self, *args, **kwargs):
    _clean_httpx_kwargs(kwargs)
    return _orig_client_init(self, *args, **kwargs)
httpx.Client.__init__ = _patched_client_init

_orig_async_client_init = httpx.AsyncClient.__init__
def _patched_async_client_init(self, *args, **kwargs):
    _clean_httpx_kwargs(kwargs)
    return _orig_async_client_init(self, *args, **kwargs)
httpx.AsyncClient.__init__ = _patched_async_client_init
