import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
import requests
import logging
from datetime import datetime

class ConfirmDialog(ctk.CTkToplevel):
    def __init__(self, parent, title, message, callback=None):
        super().__init__(parent)
        self.callback = callback
        self.result = False
        
        # 기본 설정
        self.title(title)
        self.geometry("300x150")
        self.resizable(False, False)
        
        # UI 초기화
        self.init_ui(message)
        
        # 모달로 설정
        self.transient(parent)
        self.grab_set()

    def init_ui(self, message):
        # 메시지
        ctk.CTkLabel(self, text=message, 
                    wraplength=250,
                    font=("Helvetica", 12)).pack(pady=20)
        
        # 버튼
        button_frame = ctk.CTkFrame(self)
        button_frame.pack(fill=tk.X, padx=20, pady=20)
        
        ctk.CTkButton(button_frame, text="Yes", 
                     command=self.confirm).pack(side=tk.LEFT, expand=True, padx=5)
        ctk.CTkButton(button_frame, text="No", 
                     command=self.cancel).pack(side=tk.LEFT, expand=True, padx=5)

    def confirm(self):
        self.result = True
        if self.callback:
            self.callback(True)
        self.destroy()

    def cancel(self):
        self.result = False
        if self.callback:
            self.callback(False)
        self.destroy()

class RegisterDialog(ctk.CTkToplevel):
    def __init__(self, parent, callback=None):
        super().__init__(parent)
        self.callback = callback
        self.api_base_url = "http://localhost:8000/api/v1"
        
        # UI 컴포넌트
        self.email_entry = None
        self.password_entry = None
        self.confirm_pass_entry = None
        self.name_entry = None
        
        # 기본 설정
        self.title("새 계정 등록")
        self.geometry("400x500")
        self.resizable(False, False)
        
        # UI 초기화
        self.init_ui()
        
        # 모달로 설정
        self.transient(parent)
        self.grab_set()

    def init_ui(self):
        # 제목
        ctk.CTkLabel(self, text="계정 생성", 
                    font=("Helvetica", 20, "bold")).pack(pady=20)
        
        # 입력 필드들
        self.email_entry = self._create_input_field("이메일:")
        self.password_entry = self._create_input_field("비밀번호:", show="*")
        self.confirm_pass_entry = self._create_input_field("비밀번호 확인:", show="*")
        self.name_entry = self._create_input_field("이름:")
        
        # 버튼
        button_frame = ctk.CTkFrame(self)
        button_frame.pack(fill=tk.X, padx=20, pady=20)
        
        ctk.CTkButton(button_frame, text="등록", 
                     command=self.register).pack(side=tk.LEFT, expand=True, padx=5)
        ctk.CTkButton(button_frame, text="취소", 
                     command=self.destroy).pack(side=tk.LEFT, expand=True, padx=5)

    def _create_input_field(self, label, show=None):
        frame = ctk.CTkFrame(self)
        frame.pack(fill=tk.X, padx=20, pady=5)
        
        ctk.CTkLabel(frame, text=label).pack(anchor=tk.W, padx=5, pady=2)
        entry = ctk.CTkEntry(frame, show=show, width=300)
        entry.pack(fill=tk.X, padx=5, pady=2)
        
        return entry

    def register(self):
        try:
            # 입력 검증
            email = self.email_entry.get().strip()
            password = self.password_entry.get()
            confirm_password = self.confirm_pass_entry.get()
            full_name = self.name_entry.get().strip()
            
            # 기본 검증
            if not all([email, password, confirm_password, full_name]):
                messagebox.showwarning("경고", "모든 필드를 입력해주세요")
                return
            
            if password != confirm_password:
                messagebox.showwarning("경고", "비밀번호가 일치하지 않습니다")
                return
            
            if len(password) < 8:
                messagebox.showwarning("경고", "비밀번호는 8자 이상이어야 합니다")
                return
            
            # API 요청
            response = requests.post(
                f"{self.api_base_url}/auth/register",
                json={
                    "email": email,
                    "password": password,
                    "full_name": full_name
                }
            )
            
            if response.status_code == 200:
                messagebox.showinfo("성공", "등록이 완료되었습니다! 로그인해주세요.")
                if self.callback:
                    self.callback()
                self.destroy()
            else:
                messagebox.showerror("오류", 
                                f"등록 실패: {response.json().get('detail', '알 수 없는 오류')}")
                
        except Exception as e:
            logging.error(f"등록 중 오류 발생: {str(e)}")
            messagebox.showerror("오류", f"등록 실패: {str(e)}")

class NewProjectDialog(ctk.CTkToplevel):
    def __init__(self, parent, token, callback=None):
        super().__init__(parent)
        self.token = token
        self.callback = callback
        self.api_base_url = "http://localhost:8000/api/v1"
        
        # 기본 설정
        self.title("새 프로젝트")
        self.geometry("600x600")
        self.resizable(True, True)
        
        # UI 초기화
        self.init_ui()
        
        # 모달로 설정
        self.transient(parent)
        self.grab_set()

    def init_ui(self):
        # 제목
        ctk.CTkLabel(self, text="새 연구 프로젝트 생성", 
                    font=("Helvetica", 20, "bold")).pack(pady=20)
        
        # 프로젝트 제목
        title_frame = ctk.CTkFrame(self)
        title_frame.pack(fill=tk.X, padx=20, pady=5)
        ctk.CTkLabel(title_frame, text="제목:").pack(side=tk.LEFT, padx=5)
        self.title_entry = ctk.CTkEntry(title_frame, width=400)
        self.title_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 프로젝트 설명
        desc_frame = ctk.CTkFrame(self)
        desc_frame.pack(fill=tk.BOTH, padx=20, pady=5, expand=True)
        ctk.CTkLabel(desc_frame, text="설명:").pack(anchor=tk.W, padx=5)
        self.desc_text = ctk.CTkTextbox(desc_frame, height=150)
        self.desc_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 평가 계획
        eval_frame = ctk.CTkFrame(self)
        eval_frame.pack(fill=tk.BOTH, padx=20, pady=5, expand=True)
        ctk.CTkLabel(eval_frame, text="평가 계획:").pack(anchor=tk.W, padx=5)
        self.eval_text = ctk.CTkTextbox(eval_frame, height=100)
        self.eval_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 제출 형식
        format_frame = ctk.CTkFrame(self)
        format_frame.pack(fill=tk.X, padx=20, pady=5)
        ctk.CTkLabel(format_frame, text="제출 형식:").pack(side=tk.LEFT, padx=5)
        self.format_entry = ctk.CTkEntry(format_frame, width=300)
        self.format_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 버튼
        button_frame = ctk.CTkFrame(self)
        button_frame.pack(fill=tk.X, padx=20, pady=20)
        
        ctk.CTkButton(button_frame, text="생성", 
                     command=self.create_project).pack(side=tk.LEFT, expand=True, padx=5)
        ctk.CTkButton(button_frame, text="취소", 
                     command=self.destroy).pack(side=tk.LEFT, expand=True, padx=5)

    def create_project(self):
        try:
            # 입력 검증
            title = self.title_entry.get().strip()
            description = self.desc_text.get("1.0", tk.END).strip()
            eval_plan = self.eval_text.get("1.0", tk.END).strip()
            sub_format = self.format_entry.get().strip()
            
            if not all([title, description, eval_plan, sub_format]):
                messagebox.showwarning("경고", "모든 필드를 입력해주세요")
                return
            
            # API 요청
            response = requests.post(
                f"{self.api_base_url}/projects/",
                headers={"Authorization": f"Bearer {self.token}"},
                json={
                    "title": title,
                    "description": description,
                    "evaluation_plan": eval_plan,
                    "submission_format": sub_format,
                    "metadata1": {
                        "created_at": datetime.now().isoformat(),
                        "status": "draft"
                    }
                }
            )
            
            if response.status_code == 200:
                messagebox.showinfo("성공", "프로젝트가 생성되었습니다!")
                if self.callback:
                    self.callback()
                self.destroy()
            else:
                messagebox.showerror("오류", 
                                f"프로젝트 생성 실패: {response.json().get('detail', '알 수 없는 오류')}")
                
        except Exception as e:
            logging.error(f"프로젝트 생성 중 오류 발생: {str(e)}")
            messagebox.showerror("오류", f"프로젝트 생성 실패: {str(e)}")
