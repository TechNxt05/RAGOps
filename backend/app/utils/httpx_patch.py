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


# --- Shared TLS context -----------------------------------------------------------
# Every httpx client builds a fresh SSLContext, re-parsing the whole CA bundle
# (~57 ms CPU each; ~0.5 s on a 0.1-CPU host). LLM SDKs create a sync and an async
# client per model, so the first request of each kind paid this many times over.
# SSLContext objects are safe to share across clients and threads, so the default
# configuration (standard verification, no client certificate) is built once.
import threading as _threading

from httpx import _config as _httpx_config
from httpx._transports import default as _httpx_transport

_orig_create_ssl_context = _httpx_config.create_ssl_context
_ssl_cache: dict = {}
_ssl_lock = _threading.Lock()


def _shared_ssl_context(verify=True, cert=None, trust_env=True):
    if cert is not None or not isinstance(verify, (bool, str)) or verify is False:
        return _orig_create_ssl_context(verify=verify, cert=cert, trust_env=trust_env)
    key = (verify, trust_env)
    ctx = _ssl_cache.get(key)
    if ctx is None:
        with _ssl_lock:
            ctx = _ssl_cache.get(key)
            if ctx is None:
                ctx = _ssl_cache[key] = _orig_create_ssl_context(verify=verify, cert=cert, trust_env=trust_env)
    return ctx


_httpx_config.create_ssl_context = _shared_ssl_context
_httpx_transport.create_ssl_context = _shared_ssl_context
