# app/ui/components/project_list.py
from typing import List, Any, Dict
import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk
import requests
import logging
from .project_detail import ProjectDetailWindow
from .dialogs import NewProjectDialog

class ProjectList(ctk.CTkFrame):
    def __init__(self, master, token):
        super().__init__(master)
        self.token = token
        self.api_base_url = "http://localhost:8000/api/v1"
        self.projects_tree = None
        self.init_ui()

    def init_ui(self):
        # 툴바
        toolbar = ctk.CTkFrame(self)
        toolbar.pack(fill=tk.X, padx=5, pady=5)
        
        ctk.CTkButton(toolbar, text="New Project", command=self.show_new_project_dialog).pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(toolbar, text="Refresh", command=self.load_projects).pack(side=tk.LEFT, padx=5)
        
        
        # 프로젝트 목록
        self.projects_tree = ttk.Treeview(self, columns=("ID", "Title", "Status", "Created"), show="headings")
        self.projects_tree.heading("ID", text="ID")
        self.projects_tree.heading("Title", text="Title")
        self.projects_tree.heading("Status", text="Status")
        self.projects_tree.heading("Created", text="Created")
        self.projects_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 이벤트 바인딩
        self.projects_tree.bind("<Double-1>", self.on_project_double_click)

        # 초기 데이터 로드
        self.load_projects()

    def load_projects(self):
        try:
            response = requests.get(
                f"{self.api_base_url}/projects/",
                headers={"Authorization": f"Bearer {self.token}"}
            )
            
            # 트리뷰 초기화
            for item in self.projects_tree.get_children():
                self.projects_tree.delete(item)
                
            if response.status_code == 200:
                projects = response.json()
                for project in projects:
                    self.projects_tree.insert("", tk.END, values=(
                        project["id"],
                        project["title"],
                        project["status"],
                        project["created_at"]
                    ))
            else:
                logging.error(f"Failed to load projects: {response.status_code}")
                messagebox.showerror("Error", "Failed to load projects")
        
        except Exception as e:
            logging.error(f"Error loading projects: {str(e)}")
            messagebox.showerror("Error", f"Failed to load projects: {str(e)}")

    def on_project_double_click(self, event):
        selection = self.projects_tree.selection()
        if not selection:
            return
            
        item = self.projects_tree.item(selection[0])
        project_id = item['values'][0]
        try:
            detail_window = ProjectDetailWindow(self, project_id, self.token)
            detail_window.grab_set()
        except Exception as e:
            logging.error(f"Error opening project detail: {str(e)}")
            messagebox.showerror("Error", f"Failed to open project details: {str(e)}")

    def show_new_project_dialog(self):
        try:
            dialog = NewProjectDialog(self, self.token, callback=self.load_projects)
            dialog.grab_set()
        except Exception as e:
            logging.error(f"Error showing new project dialog: {str(e)}")
            messagebox.showerror("Error", f"Failed to show new project dialog: {str(e)}")