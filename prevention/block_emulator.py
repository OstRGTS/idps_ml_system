# import json
# from pathlib import Path
# from logging_config import logger

# BLOCKED_IPS_FILE = "blocked_ips.json"

# def block_ip_emulated(ip: str):
#     try:
#         # Создаем файл, если его нет
#         if not Path(BLOCKED_IPS_FILE).exists():
#             with open(BLOCKED_IPS_FILE, "w") as f:
#                 json.dump([], f)

#         # Читаем и обновляем список
#         with open(BLOCKED_IPS_FILE, "r+") as f:
#             blocked_ips = json.load(f)
#             if ip in blocked_ips:
#                 logger.warning(f"IP {ip} уже в списке блокировки")
#                 return False
#             blocked_ips.append(ip)
#             f.seek(0)
#             json.dump(blocked_ips, f)
        
#         logger.info(f"IP {ip} добавлен в файл блокировки")
#         return True

#     except Exception as e:
#         logger.error(f"Ошибка: {str(e)}")
#         return False