import logging
import sys
import os
from logging.handlers import RotatingFileHandler
from contextvars import ContextVar

# ContextVar para manter o request_id durante a thread de execução
request_id_var: ContextVar[str] = ContextVar("request_id", default="SYSTEM")

class ContextFilter(logging.Filter):
    """
    Filtro de log que injeta o request_id do ContextVar no registro de log.
    """
    def filter(self, record):
        record.request_id = request_id_var.get()
        return True

def get_logger(name: str) -> logging.Logger:
    """
    Retorna um logger configurado com handlers de arquivo e console.
    Garante que os handlers sejam adicionados apenas uma vez.

    Args:
        name (str): Nome do logger (geralmente __name__).

    Returns:
        logging.Logger: Instância do logger configurada.
    """
    logger = logging.getLogger(name)
    
    # Evita adicionar handlers múltiplas vezes se get_logger for chamado repetidamente
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.propagate = False  

        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(request_id)s] - %(name)s - %(message)s'
        )

        # Garante que o diretório de logs exista
        log_dir = "logs"
        log_file_name = "app.log"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Handler de Arquivo: tamanho máx 1MB, 3 backups
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, log_file_name), mode='w', maxBytes=1_000_000, backupCount=3, encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(ContextFilter())
        logger.addHandler(file_handler)

        # Handler de Console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(ContextFilter())
        logger.addHandler(console_handler)

    return logger

def set_request_id(req_id: str):
    """
    Define o request_id para o contexto atual.

    Args:
        req_id (str): ID da requisição.

    Returns:
        contextvars.Token: Token para resetar o contexto posteriormente.
    """
    return request_id_var.set(req_id)

def reset_request_id(token):
    """
    Reseta o request_id para o valor anterior.

    Args:
        token (contextvars.Token): Token retornado por set_request_id.
    """
    request_id_var.reset(token)