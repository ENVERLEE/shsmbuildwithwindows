# app/ui/main_window.py
from typing import List, Any, Dict
import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk
import requests
import json
from typing import Optional
import os
from dotenv import load_dotenv
import traceback
import logging
from components.project_detail import ProjectDetailWindow
from components.project_list import ProjectList
from components.reference_view import ReferenceView
from components.settings_view import SettingsView
from components.dialogs import RegisterDialog, NewProjectDialog

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ui_debug.log'),
        logging.StreamHandler()
    ]
)

# 환경변수 로드
load_dotenv()

class ResearchAutomationUI(ctk.CTk):
    def __init__(self):
        try:
            super().__init__()
            logging.info("Starting UI initialization...")

            # 기본 설정
            self.title("Research Automation System")
            self.geometry("1200x800")
            
            # 초기 속성 설정
            self.api_base_url = "http://localhost:8000/api/v1"
            self.token = None
            self.current_project_id = None
            
            # UI 컴포넌트
            self.main_frame = None
            self.login_frame = None
            self.project_list = None
            self.reference_view = None
            self.settings_view = None
            self.email_entry = None
            self.password_entry = None
            
            # UI 스타일 설정
            ctk.set_appearance_mode("system")
            ctk.set_default_color_theme("blue")
            
            # 메인 컨테이너
            self.main_container = ctk.CTkFrame(self)
            self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # 로그인 프레임 초기화
            self.init_login_frame()
            logging.info("Login frame initialized")

            self.protocol("WM_DELETE_WINDOW", self.on_closing)
            logging.info("UI initialization completed")

        except Exception as e:
            logging.error(f"Error during initialization: {str(e)}")
            logging.error(traceback.format_exc())
            messagebox.showerror("Initialization Error", f"Error: {str(e)}\n\nCheck ui_debug.log for details")
            raise

    def init_login_frame(self):
        """로그인 프레임 초기화"""
        try:
            # 기존 프레임 제거
            if self.main_frame:
                self.main_frame.pack_forget()
            if self.login_frame:
                self.login_frame.pack_forget()

            # 로그인 프레임 생성
            self.login_frame = ctk.CTkFrame(self.main_container)
            self.login_frame.pack(fill=tk.BOTH, expand=True)

            # 로고 또는 타이틀
            ctk.CTkLabel(
                self.login_frame,
                text="Research Automation System",
                font=("Helvetica", 24, "bold")
            ).pack(pady=40)

            # 이메일 입력
            email_frame = ctk.CTkFrame(self.login_frame)
            email_frame.pack(fill=tk.X, padx=20, pady=5)
            ctk.CTkLabel(email_frame, text="Email:").pack(side=tk.LEFT, padx=5)
            self.email_entry = ctk.CTkEntry(email_frame, width=300)
            self.email_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

            # 비밀번호 입력
            pass_frame = ctk.CTkFrame(self.login_frame)
            pass_frame.pack(fill=tk.X, padx=20, pady=5)
            ctk.CTkLabel(pass_frame, text="Password:").pack(side=tk.LEFT, padx=5)
            self.password_entry = ctk.CTkEntry(pass_frame, show="*", width=300)
            self.password_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

            # 버튼 프레임
            button_frame = ctk.CTkFrame(self.login_frame)
            button_frame.pack(fill=tk.X, padx=20, pady=20)

            # 로그인 버튼
            ctk.CTkButton(
                button_frame,
                text="Login",
                command=self.handle_login
            ).pack(side=tk.LEFT, expand=True, padx=5)

            # 회원가입 버튼
            ctk.CTkButton(
                button_frame,
                text="Register",
                command=self.show_register_dialog
            ).pack(side=tk.LEFT, expand=True, padx=5)

        except Exception as e:
            logging.error(f"Error in init_login_frame: {str(e)}")
            logging.error(traceback.format_exc())
            raise

    def handle_login(self):
        """로그인 처리"""
        try:
            email = self.email_entry.get().strip()
            password = self.password_entry.get()

            if not email or not password:
                messagebox.showwarning("Warning", "Please enter both email and password")
                return

            response = requests.post(
                f"{self.api_base_url}/auth/token",
                data={
                    "username": email,
                    "password": password
                }
            )

            if response.status_code == 200:
                data = response.json()
                self.token = data["access_token"]
                # 헤더에 토큰 추가
                headers = {"Authorization": f"Bearer {self.token}"}
                # 메인 인터페이스 표시
                if self.login_frame:
                    self.login_frame.pack_forget()
                self.init_main_interface()
                messagebox.showinfo("Success", "Login successful!")
            else:
                messagebox.showerror("Error", "Invalid email or password")
        except Exception as e:
            logging.error(f"Error in login: {str(e)}")
            messagebox.showerror("Error", f"Login failed: {str(e)}")
# ... (나머지 코드)


    def show_register_dialog(self):
        """회원가입 다이얼로그 표시"""
        try:
            def register_callback():
                # 회원가입 성공 후 처리 (예: 로그인 화면 리프레시)
                self.email_entry.delete(0, tk.END)
                self.password_entry.delete(0, tk.END)
                messagebox.showinfo("Success", "Registration successful! Please login with your new account.")
            
            dialog = RegisterDialog(self, callback=register_callback)
            dialog.grab_set()
            
        except Exception as e:
            logging.error(f"Error showing register dialog: {str(e)}")
            messagebox.showerror("Error", f"Failed to show registration dialog: {str(e)}")

    def handle_registration(self, email: str, password: str, full_name: str):
        """회원가입 처리"""
        try:
            response = requests.post(
                f"{self.api_base_url}/auth/register",
                json={
                    "email": email,
                    "password": password,
                    "full_name": full_name
                }
            )

            if response.status_code == 200:
                data = response.json()
                self.token = data["access_token"]
                messagebox.showinfo("Success", "Registration successful! You are now logged in.")
                self.init_main_interface()
                return True
            else:
                error_detail = response.json().get("detail", "Registration failed")
                messagebox.showerror("Error", error_detail)
                return False
        except Exception as e:
            logging.error(f"Error in registration: {str(e)}")
            messagebox.showerror("Error", f"Registration failed: {str(e)}")
            return False

    def init_main_interface(self):
        try:
            # 메인 프레임 생성
            self.main_frame = ctk.CTkFrame(self.main_container)
            self.main_frame.pack(fill=tk.BOTH, expand=True)

            # 탭 컨트롤
            self.tab_view = ctk.CTkTabview(self.main_frame)
            self.tab_view.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # 프로젝트 탭
            projects_tab = self.tab_view.add("Projects")
            self.project_list = ProjectList(projects_tab, self.token)
            self.project_list.pack(fill=tk.BOTH, expand=True)

            # 참고문헌 탭
            references_tab = self.tab_view.add("References")
            self.reference_view = ReferenceView(references_tab, self.token)
            self.reference_view.pack(fill=tk.BOTH, expand=True)

            # 설정 탭
            settings_tab = self.tab_view.add("Settings")
            self.settings_view = SettingsView(settings_tab)
            self.settings_view.pack(fill=tk.BOTH, expand=True)

            # 로그아웃 버튼 추가
            logout_btn = ctk.CTkButton(
                self.main_frame,
                text="Logout",
                command=self.logout
            )
            logout_btn.pack(padx=10, pady=5)

            logging.info("Main interface initialized")
        except Exception as e:
            logging.error(f"Error in init_main_interface: {str(e)}")
            logging.error(traceback.format_exc())
            raise

    def logout(self):
        """로그아웃 처리"""
        self.token = None
        self.current_project_id = None
        if self.main_frame:
            self.main_frame.pack_forget()
        self.init_login_frame()

    def on_closing(self):
        """애플리케이션 종료 처리"""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.quit()

if __name__ == "__main__":
    try:
        logging.info("Starting application...")
        app = ResearchAutomationUI()
        app.mainloop()
    except Exception as e:
        logging.error(f"Critical error in main: {str(e)}")
        logging.error(traceback.format_exc())
        messagebox.showerror("Critical Error", f"A critical error occurred: {str(e)}\n\nCheck ui_debug.log for details")