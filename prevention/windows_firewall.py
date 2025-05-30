# prevention/windows_firewall.py
import subprocess
from logging_config import logger

def block_ip_windows(ip: str):
    try:
        # Проверяем, не заблокирован ли IP уже
        check_cmd = f'netsh advfirewall firewall show rule name="Block {ip}"'
        result = subprocess.run(check_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if "Block " + ip in result.stdout:
            logger.warning(f"IP {ip} уже заблокирован в Windows Firewall")
            return False

        # Блокируем IP
        block_cmd = f'''
        netsh advfirewall firewall add rule name="Block {ip}" dir=in action=block remoteip={ip} protocol=any
        '''
        subprocess.run(block_cmd, shell=True, check=True)
        logger.info(f"IP {ip} заблокирован в Windows Firewall")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Ошибка блокировки IP {ip}: {e.stderr}")
        return False