# app/ui/components/settings_view.py

import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
from typing import List, Any, Dict
import logging
import os
from dotenv import load_dotenv, set_key
import json

class SettingsView(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.init_ui()
        self.load_current_settings()

    def init_ui(self):
        # API 설정
        api_frame = ctk.CTkFrame(self)
        api_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(api_frame, text="API Settings", 
                    font=("Helvetica", 16, "bold")).pack(pady=10)
        
        # Cohere API
        cohere_frame = ctk.CTkFrame(api_frame)
        cohere_frame.pack(fill=tk.X, pady=5)
        
        ctk.CTkLabel(cohere_frame, text="Cohere API Key:").pack(side=tk.LEFT, padx=5)
        self.cohere_key = ctk.CTkEntry(cohere_frame, show="*", width=400)
        self.cohere_key.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Perplexity API
        perplexity_frame = ctk.CTkFrame(api_frame)
        perplexity_frame.pack(fill=tk.X, pady=5)
        
        ctk.CTkLabel(perplexity_frame, text="Perplexity API Key:").pack(side=tk.LEFT, padx=5)
        self.perplexity_key = ctk.CTkEntry(perplexity_frame, show="*", width=400)
        self.perplexity_key.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # 데이터베이스 설정
        db_frame = ctk.CTkFrame(self)
        db_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(db_frame, text="Database Settings", 
                    font=("Helvetica", 16, "bold")).pack(pady=10)
        
        # Host
        host_frame = ctk.CTkFrame(db_frame)
        host_frame.pack(fill=tk.X, pady=5)
        ctk.CTkLabel(host_frame, text="Host:").pack(side=tk.LEFT, padx=5)
        self.db_host = ctk.CTkEntry(host_frame, width=200)
        self.db_host.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # User
        user_frame = ctk.CTkFrame(db_frame)
        user_frame.pack(fill=tk.X, pady=5)
        ctk.CTkLabel(user_frame, text="User:").pack(side=tk.LEFT, padx=5)
        self.db_user = ctk.CTkEntry(user_frame)
        self.db_user.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Password
        pass_frame = ctk.CTkFrame(db_frame)
        pass_frame.pack(fill=tk.X, pady=5)
        ctk.CTkLabel(pass_frame, text="Password:").pack(side=tk.LEFT, padx=5)
        self.db_password = ctk.CTkEntry(pass_frame, show="*")
        self.db_password.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Database Name
        name_frame = ctk.CTkFrame(db_frame)
        name_frame.pack(fill=tk.X, pady=5)
        ctk.CTkLabel(name_frame, text="Database:").pack(side=tk.LEFT, padx=5)
        self.db_name = ctk.CTkEntry(name_frame)
        self.db_name.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 저장 버튼
        self.save_btn = ctk.CTkButton(self, text="Save Settings", 
                                    command=self.save_settings)
        self.save_btn.pack(pady=20)

        # 테스트 연결 버튼
        self.test_btn = ctk.CTkButton(self, text="Test Connection", 
                                    command=self.test_connection)
        self.test_btn.pack(pady=10)

    def load_current_settings(self):
        """현재 환경 변수에서 설정 로드"""
        try:
            # .env 파일 로드
            load_dotenv()
            
            # API 키
            self.cohere_key.insert(0, os.getenv("COHERE_API_KEY", ""))
            self.perplexity_key.insert(0, os.getenv("PERPLEXITY_API_KEY", ""))
            
            # 데이터베이스 설정
            self.db_host.insert(0, os.getenv("POSTGRES_HOST", "localhost"))
            self.db_user.insert(0, os.getenv("POSTGRES_USER", ""))
            self.db_password.insert(0, os.getenv("POSTGRES_PASSWORD", ""))
            self.db_name.insert(0, os.getenv("POSTGRES_DB", ""))
            
        except Exception as e:
            logging.error(f"Error loading settings: {str(e)}")
            messagebox.showerror("Error", f"Failed to load settings: {str(e)}")

    def save_settings(self):
        """설정을 .env 파일에 저장"""
        try:
            env_path = ".env"
            
            # 현재 설정 가져오기
            settings = {
                "COHERE_API_KEY": self.cohere_key.get(),
                "PERPLEXITY_API_KEY": self.perplexity_key.get(),
                "POSTGRES_HOST": self.db_host.get(),
                "POSTGRES_USER": self.db_user.get(),
                "POSTGRES_PASSWORD": self.db_password.get(),
                "POSTGRES_DB": self.db_name.get(),
                "DATABASE_URL": f"postgresql://{self.db_user.get()}:{self.db_password.get()}@{self.db_host.get()}/{self.db_name.get()}"
            }
            
            # .env 파일 업데이트
            for key, value in settings.items():
                set_key(env_path, key, value)
            
            messagebox.showinfo("Success", "Settings saved successfully")
            
        except Exception as e:
            logging.error(f"Error saving settings: {str(e)}")
            messagebox.showerror("Error", f"Failed to save settings: {str(e)}")

    def test_connection(self):
        """데이터베이스 연결 테스트"""
        try:
            import psycopg2
            
            conn = psycopg2.connect(
                host=self.db_host.get(),
                database=self.db_name.get(),
                user=self.db_user.get(),
                password=self.db_password.get()
            )
            
            conn.close()
            messagebox.showinfo("Success", "Database connection successful!")
            
        except Exception as e:
            logging.error(f"Database connection error: {str(e)}")
            messagebox.showerror("Error", f"Failed to connect to database: {str(e)}")