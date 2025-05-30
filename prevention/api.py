from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import logging
from typing import Optional
import subprocess
from pydantic import BaseModel
from prometheus_client import Counter, make_asgi_app


REQUEST_COUNT = Counter("request_count", "Total API requests")


# Настройка логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
API_KEY = "p4ohvMmbCQlIfi02"  # Ваш ключ

# Модель для запроса
class IPRequest(BaseModel):
    ip: str

# Защита API
api_key_header = APIKeyHeader(name="X-API-Key")

def validate_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

# Инициализация лимитера
@app.on_event("startup")
async def startup():
    FastAPILimiter.init("redis://localhost")

def block_ip_windows(ip: str) -> bool:
    """Функция блокировки IP через Windows Firewall"""
    try:
        # Проверяем, не заблокирован ли IP уже
        check_cmd = f'netsh advfirewall firewall show rule name="Block {ip}"'
        result = subprocess.run(check_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if "Block " + ip in result.stdout:
            logger.warning(f"IP {ip} уже заблокирован")
            return False

        # Блокируем IP
        block_cmd = f'''
        netsh advfirewall firewall add rule name="Block {ip}" dir=in action=block remoteip={ip} protocol=any
        '''
        subprocess.run(block_cmd, shell=True, check=True)
        logger.info(f"IP {ip} успешно заблокирован")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Ошибка блокировки IP {ip}: {e.stderr}")
        return False
    
    

@app.post("/block_ip", dependencies=[Depends(RateLimiter(times=10, seconds=1))])
async def block_ip(request: IPRequest, api_key: str = Depends(validate_api_key)):
    """
    Блокирует указанный IP адрес
    Требует:
    - X-API-Key в заголовках
    - JSON с полем ip
    """
    try:
        if not block_ip_windows(request.ip):
            raise HTTPException(status_code=400, detail="IP уже заблокирован или ошибка блокировки")
        return {"status": "success", "ip": request.ip}
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


































# from fastapi import FastAPI, HTTPException, Depends
# from fastapi.security import APIKeyHeader
# from fastapi_limiter import FastAPILimiter
# from fastapi_limiter.depends import RateLimiter
# from logging_config import logger  # Абсолютный импорт
# from prevention.windows_firewall import block_ip_windows  # Абсолютный импорт

# app = FastAPI()
# API_KEY = "p4ohvMmbCQlIfi02"  # Замените на реальный ключ

# # Защита API
# api_key_header = APIKeyHeader(name="X-API-Key")

# def validate_api_key(api_key: str = Depends(api_key_header)):
#     if api_key != API_KEY:
#         raise HTTPException(status_code=403, detail="Invalid API Key")

# @app.post("/block_ip", dependencies=[Depends(RateLimiter(times=10, seconds=1))])
# async def block_ip(ip: str, api_key: str = Depends(validate_api_key)):
#     if not block_ip_windows(ip):  # Или block_ip_emulated(ip)
#         raise HTTPException(status_code=400, detail="IP уже заблокирован")
#     return {"status": "success", "ip": ip}