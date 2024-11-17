import sys
import os
import threading
import uvicorn
from pathlib import Path
import importlib.util
import logging
import subprocess
import traceback
from dotenv import load_dotenv
import shutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_environment():
    """환경 변수 설정"""
    try:
        # 실행 파일이 있는 디렉토리
        exe_dir = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(os.path.dirname(os.path.abspath(__file__)))
        
        # 외부 .env 파일 경로 (실행 파일과 같은 디렉토리)
        external_env_path = exe_dir / '.env'
        
        # PyInstaller 내장 .env 파일 경로
        internal_env_path = Path(sys._MEIPASS) / '.env' if getattr(sys, 'frozen', False) else Path(os.path.dirname(os.path.abspath(__file__))) / '.env'
        
        # 외부 .env 파일이 없고 내장 .env 파일이 있는 경우, 복사
        if not external_env_path.exists() and internal_env_path.exists():
            shutil.copy2(internal_env_path, external_env_path)
            logger.info(f"Created external .env file at {external_env_path}")
        
        # 외부 .env 파일 로드
        if external_env_path.exists():
            load_dotenv(external_env_path)
            os.environ['DOTENV_PATH'] = str(external_env_path)
            logger.info(f"Loaded environment variables from external .env: {external_env_path}")
        else:
            logger.warning("No .env file found!")
            
    except Exception as e:
        logger.error(f"Error setting up environment: {e}")
        raise

def verify_dependencies():
    """필요한 패키지 확인 및 설치"""
    required_packages = [
        'customtkinter',
        'requests',
        'python-dotenv',
        'fastapi',
        'uvicorn'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"{package} is already installed")
        except ImportError:
            logger.warning(f"{package} not found, attempting to install...")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                logger.info(f"Successfully installed {package}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install {package}: {e}")
                raise

def setup_import_paths():
    """Import 경로 설정"""
    try:
        if getattr(sys, 'frozen', False):
            # PyInstaller로 빌드된 경우
            current_dir = os.path.dirname(sys.executable)
        else:
            # 일반 Python 실행의 경우
            current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 프로젝트 루트 디렉토리 추가
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # app 디렉토리 추가
        app_dir = os.path.join(current_dir, 'app')
        if app_dir not in sys.path:
            sys.path.insert(0, app_dir)
        
        # ui 디렉토리 추가
        ui_dir = os.path.join(app_dir, 'ui')
        if ui_dir not in sys.path:
            sys.path.insert(0, ui_dir)
        
        # components 디렉토리 추가
        components_dir = os.path.join(ui_dir, 'components')
        if components_dir not in sys.path:
            sys.path.insert(0, components_dir)
        
        logger.info(f"Python path: {sys.path}")
    except Exception as e:
        logger.error(f"Error setting up import paths: {e}")
        raise


def import_module_from_path(module_name, file_path):
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logger.error(f"Error importing {module_name} from {file_path}: {e}")
        logger.error(f"Current sys.path: {sys.path}")
        raise

def run_api_server():
    try:
        logger.info("Starting API server...")
        import uvicorn
        from app.main import app
        uvicorn.run(app, host="127.0.0.1", port=8000)
    except Exception as e:
        logger.error(f"Error starting API server: {e}")
        raise

def run_ui():
    try:
        logger.info("Starting UI...")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ui_path = os.path.join(current_dir, "app", "ui", "main_window.py")
        
        # Import 경로 설정
        setup_import_paths()
        
        # 패키지 종속성 확인
        verify_dependencies()
        
        # main_window 모듈 import하고 ResearchAutomationUI 실행
        main_window = import_module_from_path("main_window", ui_path)
        app = main_window.ResearchAutomationUI()
        app.mainloop()
    except Exception as e:
        logger.error(f"Error starting UI: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def main():
    try:
        # 환경변수 설정
        setup_environment()
        
        # Import 경로 설정
        setup_import_paths()
        
        # 패키지 종속성 확인
        verify_dependencies()
        
        # API 서버를 별도 스레드로 실행
        api_thread = threading.Thread(target=run_api_server, daemon=True)
        api_thread.start()
        logger.info("API server thread started")
        
        # UI 실행 (메인 스레드)
        logger.info("Starting UI in main thread")
        run_ui()
    except Exception as e:
        logger.error(f"Application error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()