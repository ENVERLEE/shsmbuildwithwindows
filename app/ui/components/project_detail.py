# app/ui/components/project_detail.py
# 기본 Python 모듈
import os
import sys
import json
import logging
from datetime import datetime
from typing import List, Any, Dict, Optional, Union

# GUI 관련 모듈
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import customtkinter as ctk

# 데이터 처리 및 시각화
import requests
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np  # 데이터 처리를 위해 필요할 수 있음
import pandas as pd  # 데이터 분석을 위해 필요할 수 있음

sys.stdout.flush()

class ProjectDetailWindow(ctk.CTkToplevel):
    def __init__(self, parent, project_id, token):
        super().__init__(parent)
        self.token = token
        self.project_id = project_id
        self.api_base_url = "http://localhost:8000/api/v1"
        
        self.title("연구 프로젝트 상세")
        self.geometry("1200x800")
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        self.init_attributes()
        self.init_ui()
        self.load_project_info()
        
        # 자동 새로고침 설정
        self.after(5000, self.refresh_data)
    
    def init_attributes(self):
        #속성 초기화#
        self.progress_bars = {}
        self.status_labels = {}
        self.step_results = {}
        self.current_step = None
        self.step_detail = None
        self.fig = None
        self.canvas = None
    
    def init_ui(self):
        #UI 초기화#
        self.tab_view = ctk.CTkTabview(self)
        self.tab_view.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 탭 추가
        self.overview_tab = self.tab_view.add("개요")
        self.steps_tab = self.tab_view.add("연구 단계")
        self.results_tab = self.tab_view.add("결과 분석")
        
        self.init_overview_tab()
        self.init_steps_tab()
        self.init_results_tab()
    
    def init_overview_tab(self):
        #개요 탭 초기화#
        frame = ctk.CTkFrame(self.overview_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 프로젝트 제목
        self.title_label = ctk.CTkLabel(
            frame, 
            text="", 
            font=("Helvetica", 20, "bold")
        )
        self.title_label.pack(pady=10)
        
        # 진행 상태
        status_frame = ctk.CTkFrame(frame)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_label = ctk.CTkLabel(status_frame, text="상태: ")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        self.progress_bar = ctk.CTkProgressBar(status_frame)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 설명
        ctk.CTkLabel(
            frame, 
            text="설명:", 
            font=("Helvetica", 12, "bold")
        ).pack(anchor=tk.W, padx=10)
        self.desc_text = ctk.CTkTextbox(frame, height=150)
        self.desc_text.pack(fill=tk.X, padx=10, pady=5)
        
        # 평가 계획
        ctk.CTkLabel(
            frame, 
            text="평가 계획:", 
            font=("Helvetica", 12, "bold")
        ).pack(anchor=tk.W, padx=10)
        self.eval_text = ctk.CTkTextbox(frame, height=150)
        self.eval_text.pack(fill=tk.X, padx=10, pady=5)
        
        # 버튼 프레임
        button_frame = ctk.CTkFrame(frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ctk.CTkButton(
            button_frame, 
            text="연구 시작",
            command=self.start_research
        ).pack(side=tk.LEFT, padx=5)
        
        ctk.CTkButton(
            button_frame, 
            text="결과 내보내기",
            command=self.export_results
        ).pack(side=tk.LEFT, padx=5)
    
    def init_steps_tab(self):
        #연구 단계 탭 초기화#
        frame = ctk.CTkFrame(self.steps_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 단계 목록
        columns = ("단계", "상태", "진행률", "시작 시간", "완료 시간")
        self.steps_tree = ttk.Treeview(
            frame, 
            columns=columns,
            show="headings"
        )
        
        # 열 설정
        for col in columns:
            self.steps_tree.heading(col, text=col)
            self.steps_tree.column(col, width=100)
        
        self.steps_tree.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 단계 상세 정보
        self.step_detail = ctk.CTkTextbox(frame, height=200)
        self.step_detail.pack(fill=tk.X, pady=5)
        
        # 이벤트 바인딩
        self.steps_tree.bind("<<TreeviewSelect>>", self.on_step_select)
    
    def init_results_tab(self):
        #결과 분석 탭 초기화#
        frame = ctk.CTkFrame(self.results_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 그래프
        self.fig = Figure(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 최종 결과
        self.final_result_text = ctk.CTkTextbox(frame, height=200)
        self.final_result_text.pack(fill=tk.X, pady=5)
    
    def on_step_select(self, event):
        #단계 선택 이벤트 처리#
        selection = self.steps_tree.selection()
        if not selection:
            return
        item = self.steps_tree.item(selection[0])
        print(item, flush=True)
        step_number_with_string = item['values'][0]  # "단계 N"에서 N 추출
        step_number = step_number_with_string.replace('단계 ', '')  # 'l' 문자를 제거
        int(step_number)
        print(step_number, flush=True)
        try:
            response = requests.get(
                f"{self.api_base_url}/projects/{self.project_id}/steps/{step_number}",
                headers={"Authorization": f"Bearer {self.token}"}
            )
            
            if response.status_code == 200:
                step = response.json()
                detail_text = self.format_step_detail(step)
                self.step_detail.delete("1.0", tk.END)
                self.step_detail.insert("1.0", detail_text)
            else:
                messagebox.showerror(
                    "오류", 
                    f"단계 정보 조회 실패: {response.json().get('detail', '알 수 없는 오류')}"
                )
                
        except Exception as e:
            self.logger.error(f"단계 정보 조회 중 오류 발생: {str(e)}")
            messagebox.showerror("오류", f"단계 정보 조회 실패: {str(e)}")
    
    def format_step_detail(self, step):
        #단계 상세 정보 포맷팅#
        detail = f"단계 {step['step_number']}\n"
        detail += f"상태: {step['status']}\n\n"
        detail += f"설명:\n{step['description']}\n\n"
        detail += f"키워드: {', '.join(step['keywords'])}\n\n"
        detail += f"방법론:\n{step['methodology']}\n\n"
        
        if step.get('result'):
            detail += f"결과:\n{json.dumps(step['result'], indent=2, ensure_ascii=False)}"
            
        return detail
    
    def start_research(self):
        """연구 시작"""
        if messagebox.askyesno("확인", "연구를 시작하시겠습니까?"):
            try:
                response = requests.post(
                    f"{self.api_base_url}/projects/{self.project_id}/execute",
                    headers={"Authorization": f"Bearer {self.token}"}
                )
                
                if response.status_code == 200:
                    messagebox.showinfo("성공", "연구가 시작되었습니다.")
                    self.refresh_data()  # 즉시 데이터 새로고침
                    
                    # 상태 업데이트 타이머 시작
                    self.after(5000, self.check_research_status)
                else:
                    error_detail = response.json().get('detail', '알 수 없는 오류')
                    messagebox.showerror("오류", f"연구 시작 실패: {error_detail}")
            except Exception as e:
                self.logger.error(f"연구 시작 중 오류 발생: {str(e)}")
                messagebox.showerror("오류", f"연구 시작 실패: {str(e)}")

    def check_research_status(self):
        """연구 상태 확인"""
        try:
            response = requests.get(
                f"{self.api_base_url}/projects/{self.project_id}/status",
                headers={"Authorization": f"Bearer {self.token}"}
            )
            
            if response.status_code == 200:
                status_data = response.json()
                if status_data["status"] in ["in_progress", "pending"]:
                    # 아직 진행 중이면 계속 확인
                    self.after(5000, self.check_research_status)
                
                self.update_research_progress(status_data)
        except Exception as e:
            self.logger.error(f"상태 확인 중 오류 발생: {str(e)}")
# app/ui/components/project_detail.py에 추가

    def update_steps_tree(self, steps_data: List[Dict]):
        """연구 단계 트리 업데이트"""
        try:
            # 기존 항목 삭제
            for item in self.steps_tree.get_children():
                self.steps_tree.delete(item)
                
            # 새 데이터 추가
            for step in steps_data:
                values = (
                    f"단계 {step['step_number']}",
                    step['status'],
                    f"{step.get('progress', 0)}%",
                    step.get('started_at', ''),
                    step.get('completed_at', '')
                )
                
                # 상태에 따른 태그 설정
                tags = []
                if step['status'] == 'completed':
                    tags.append('completed')
                elif step['status'] == 'in_progress':
                    tags.append('in_progress')
                elif step['status'] == 'failed':
                    tags.append('failed')
                
                self.steps_tree.insert('', 'end', values=values, tags=tags)
                
            # 진행 상황이 있는 경우 상세 정보 업데이트
            selection = self.steps_tree.selection()
            if selection:
                self.on_step_select(None)  # 현재 선택된 항목 정보 업데이트
                
        except Exception as e:
            self.logger.error(f"단계 트리 업데이트 중 오류 발생: {str(e)}")
            messagebox.showerror("오류", f"단계 정보 업데이트 실패: {str(e)}")

    # 기존의 update_research_progress 메서드도 수정
    def update_research_progress(self, status_data):
        """연구 진행 상황 UI 업데이트"""
        try:
            # 전체 진행률 업데이트
            progress = (status_data["completed_steps"] / status_data["total_steps"]) * 100 if status_data["total_steps"] > 0 else 0
            self.progress_bar.set(progress / 100)
            
            # 상태 레이블 업데이트
            status_text = f"상태: {status_data['status']} ({progress:.1f}%)"
            self.status_label.configure(text=status_text)
            
            # 단계별 상태 업데이트
            if "steps" in status_data:
                self.update_steps_tree(status_data["steps"])
            
        except Exception as e:
            self.logger.error(f"진행 상황 업데이트 중 오류 발생: {str(e)}")
                
    def export_results(self):
        """연구 결과를 DOCX 형식으로 내보내기"""
        try:
            from docx import Document
            from docx.shared import Inches, Pt
            from docx.enum.text import WD_ALIGN_PARAGRAPH
            
            # 저장할 파일 경로 선택 대화상자 표시
            file_path = filedialog.asksaveasfilename(
                defaultextension=".docx",
                filetypes=[("Word documents", "*.docx")],
                initialdir=os.path.expanduser("~"),
                initialfile=f"연구결과_{self.project_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
            )
            
            if not file_path:  # 사용자가 취소한 경우
                return

            # 프로젝트 정보 가져오기
            project_response = requests.get(
                f"{self.api_base_url}/projects/{self.project_id}",
                headers={"Authorization": f"Bearer {self.token}"}
            )
            
            # 연구 결과 가져오기    
            response = requests.get(
                f"{self.api_base_url}/projects/{self.project_id}/export",
                headers={"Authorization": f"Bearer {self.token}"}
            )
            
            if response.status_code == 200 and project_response.status_code == 200:
                # 문서 생성
                doc = Document()
                project_data = project_response.json()
                results_data = response.json()
                
                # 제목 스타일 설정
                title = doc.add_heading(project_data['title'], 0)
                title.alignment = WD_ALIGN_PARAGRAPH.CENTER
                
                # 프로젝트 개요
                doc.add_heading('프로젝트 개요', level=1)
                doc.add_paragraph(f"설명: {project_data['description']}")
                doc.add_paragraph(f"평가 계획: {project_data['evaluation_plan']}")
                doc.add_paragraph(f"제출 형식: {project_data['submission_format']}")
                doc.add_paragraph(f"상태: {project_data['evaluation_status']}")
                
                # 연구 단계별 결과
                doc.add_heading('연구 단계별 결과', level=1)
                
                for result in results_data:
                    # 단계 제목
                    doc.add_heading(f"{result['step']}: {result['title']}", level=2)
                    
                    # 상태 정보
                    p = doc.add_paragraph()
                    p.add_run('상태: ').bold = True
                    p.add_run(f"{result['status']}")
                    
                    # 시작 및 완료 시간
                    if result.get('started_at'):
                        p = doc.add_paragraph()
                        p.add_run('시작 시간: ').bold = True
                        p.add_run(f"{result['started_at']}")
                    
                    if result.get('completed_at'):
                        p = doc.add_paragraph()
                        p.add_run('완료 시간: ').bold = True
                        p.add_run(f"{result['completed_at']}")
                    
                    # 결과 내용
                    if result.get('result'):
                        doc.add_heading('결과', level=3)
                        try:
                            result_content = json.loads(result['result'])
                            # 결과가 구조화된 데이터인 경우
                            if isinstance(result_content, dict):
                                for key, value in result_content.items():
                                    p = doc.add_paragraph()
                                    p.add_run(f"{key}: ").bold = True
                                    p.add_run(str(value))
                            else:
                                doc.add_paragraph(str(result_content))
                        except json.JSONDecodeError:
                            # JSON 파싱 실패 시 원문 그대로 출력
                            doc.add_paragraph(result['result'])
                    
                    # 구분선 추가
                    doc.add_paragraph('_' * 50)
                
                # 문서 저장
                doc.save(file_path)
                messagebox.showinfo("성공", f"결과가 {file_path}에 저장되었습니다.")
            else:
                messagebox.showerror(
                    "오류", 
                    f"결과 내보내기 실패: {response.json().get('detail', '알 수 없는 오류')}"
                )
                
        except ImportError:
            messagebox.showerror("오류", "python-docx 라이브러리가 설치되어 있지 않습니다. pip install python-docx를 실행해주세요.")
        except Exception as e:
            self.logger.error(f"결과 내보내기 중 오류 발생: {str(e)}")
            messagebox.showerror("오류", f"결과 내보내기 실패: {str(e)}")
    def refresh_data(self):
        #데이터 새로고침#
        self.load_project_info()
        self.after(5000, self.refresh_data)
    
    def load_project_info(self):
        #프로젝트 정보 로드#
        try:
            # 프로젝트 기본 정보
            response = requests.get(
                f"{self.api_base_url}/projects/{self.project_id}",
                headers={"Authorization": f"Bearer {self.token}"}
            )
            
            if response.status_code == 200:
                project = response.json()
                self.update_project_info(project)
            else:
                self.logger.error(f"프로젝트 정보 로드 실패: {response.status_code}")
                return
            
            # 연구 단계 정보
            steps_response = requests.get(
                f"{self.api_base_url}/projects/{self.project_id}/steps",
                headers={"Authorization": f"Bearer {self.token}"}
            )
            
            if steps_response.status_code == 200:
                steps = steps_response.json()
                self.update_steps_info(steps)
            else:
                self.logger.error(f"연구 단계 정보 로드 실패: {steps_response.status_code}")
            
            # 진행 상황 업데이트
            progress_response = requests.get(
                f"{self.api_base_url}/projects/{self.project_id}/progress",
                headers={"Authorization": f"Bearer {self.token}"}
            )
            
            if progress_response.status_code == 200:
                progress = progress_response.json()
                self.update_progress_info(progress)
            else:
                self.logger.error(f"진행 상황 정보 로드 실패: {progress_response.status_code}")
                
        except Exception as e:
            self.logger.error(f"정보 로드 중 오류 발생: {str(e)}")
            # 오류가 발생해도 UI를 계속 업데이트하기 위해 예외를 무시
    
    def update_project_info(self, project):
        #프로젝트 정보 업데이트#
        self.title_label.configure(text=project["title"])
        self.desc_text.delete("1.0", tk.END)
        self.desc_text.insert("1.0", project["description"])
        self.eval_text.delete("1.0", tk.END)
        self.eval_text.insert("1.0", project.get("evaluation_plan", ""))
    
    def update_steps_info(self, steps):
        #연구 단계 정보 업데이트#
        for item in self.steps_tree.get_children():
            self.steps_tree.delete(item)
            
        for step in steps:
            self.steps_tree.insert("", tk.END, values=(
                f"단계 {step['step_number']}",
                step["status"],
                f"{step.get('progress_percentage', 0)}%",
                step.get("started_at", ""),
                step.get("completed_at", "")
            ))
    
    def update_progress_info(self, progress):
        #진행 상황 업데이트#
        self.progress_bar.set(progress["progress_percentage"] / 100)
        self.status_label.configure(
            text=f"상태: {progress['status']} ({progress['progress_percentage']}%)"
        )