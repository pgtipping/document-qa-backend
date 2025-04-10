from typing import Any, Dict, List, Optional, Union

def run(
    app: Union[str, Any],
    host: str = ...,
    port: int = ...,
    uds: Optional[str] = ...,
    fd: Optional[int] = ...,
    loop: Optional[str] = ...,
    http: str = ...,
    ws: str = ...,
    ws_max_size: int = ...,
    ws_ping_interval: float = ...,
    ws_ping_timeout: float = ...,
    ws_per_message_deflate: bool = ...,
    lifespan: str = ...,
    interface: str = ...,
    reload: bool = ...,
    reload_dirs: Optional[Union[List[str], str]] = ...,
    reload_includes: Optional[List[str]] = ...,
    reload_excludes: Optional[List[str]] = ...,
    reload_delay: float = ...,
    workers: Optional[int] = ...,
    env_file: Optional[str] = ...,
    log_config: Optional[Union[Dict[str, Any], str]] = ...,
    log_level: Optional[str] = ...,
    access_log: bool = ...,
    proxy_headers: bool = ...,
    server_header: bool = ...,
    date_header: bool = ...,
    forwarded_allow_ips: Optional[str] = ...,
    root_path: str = ...,
    limit_concurrency: Optional[int] = ...,
    limit_max_requests: Optional[int] = ...,
    timeout_keep_alive: int = ...,
    timeout_notify: int = ...,
    callback_notify: Optional[Any] = ...,
    ssl_keyfile: Optional[str] = ...,
    ssl_certfile: Optional[str] = ...,
    ssl_keyfile_password: Optional[str] = ...,
    ssl_version: int = ...,
    ssl_cert_reqs: int = ...,
    ssl_ca_certs: Optional[str] = ...,
    ssl_ciphers: str = ...,
    headers: Optional[List[tuple[str, str]]] = ...,
    factory: bool = ...,
) -> None: ... 