from setuptools import setup
import os
import sys
import subprocess

# 재귀 제한 증가
sys.setrecursionlimit(5000)

def collect_package_data():
    package_data = {
        'app': [
            'ui/assets/*',
            'core/config/*',
        ]
    }
    return package_data

def collect_data_files():
    data_files = [
        ('', ['alembic.ini']),
        ('alembic', [os.path.join('alembic', f) for f in os.listdir('alembic') if f.endswith('.py') or f == 'README']),
        ('alembic/versions', [os.path.join('alembic/versions', f) for f in os.listdir('alembic/versions') if f.endswith('.py')])
    ]
    return data_files

APP = ['launcher.py']
DATA_FILES = collect_data_files()
OPTIONS = {
    'argv_emulation': True,
    'packages': [
        'customtkinter',
        'fastapi',
        'uvicorn',
        'sqlalchemy',
        'alembic',
        'anthropic',
        'openai',
        'pydantic',
        'requests',
        'numpy',
        'pandas',
    ],
    'excludes': [
        'PyInstaller',
        'PySide6',
        'PyQt6',
        'tkinter',
        'test',
        'distutils',
        'setuptools',
        'pip',
        'numpy.random._examples'
    ],
    'semi_standalone': True,  # 시스템 Python 프레임워크 사용
    'site_packages': True,    # 사이트 패키지 포함
    'includes': [
        'urllib3',
        'packaging',
        'email_validator',
        'python_multipart',
        'passlib',
        'bcrypt',
        'customtkinter.windows.widgets.core_rendering',
        'customtkinter.windows.widgets.core_widget_classes',
        'customtkinter.windows.widgets.font',
    ],
    'iconfile': 'app/ui/assets/icon.icns',
    'plist': {
        'CFBundleName': 'Suhangssalmuk',
        'CFBundleDisplayName': '수행쌀먹',
        'CFBundleIdentifier': 'com.suhangssalmuk.app',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '10.13',
    }
}

setup(
    name='Suhangssalmuk',
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
    packages=['app'],
    package_data=collect_package_data(),
    install_requires=[
        'fastapi',
        'uvicorn',
        'sqlalchemy',
        'customtkinter',
        'anthropic',
        'openai',
        'python-dotenv',
        'alembic',
    ]
)