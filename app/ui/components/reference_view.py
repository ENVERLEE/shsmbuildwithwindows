# app/ui/components/reference_view.py
from typing import List, Any, Dict
import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk
import requests
import logging

class ReferenceView(ctk.CTkFrame):
    def __init__(self, master, token):
        super().__init__(master)
        self.token = token
        self.api_base_url = "http://localhost:8000/api/v1"
        self.refs_tree = None
        self.detail_text = None
        self.search_entry = None
        
        self.init_ui()

    def init_ui(self):
        # 검색 프레임
        search_frame = ctk.CTkFrame(self)
        search_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.search_entry = ctk.CTkEntry(search_frame, placeholder_text="Search references...")
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        ctk.CTkButton(search_frame, text="Search", command=self.search_references).pack(side=tk.LEFT)
        
        # 참고문헌 목록
        self.refs_tree = ttk.Treeview(self, columns=("Title", "Authors", "Date"), show="headings")
        self.refs_tree.heading("Title", text="Title")
        self.refs_tree.heading("Authors", text="Authors")
        self.refs_tree.heading("Date", text="Date")
        self.refs_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 상세 정보
        self.detail_text = ctk.CTkTextbox(self, height=200)
        self.detail_text.pack(fill=tk.X, padx=10, pady=5)

        # 이벤트 바인딩
        self.refs_tree.bind("<<TreeviewSelect>>", self.on_reference_select)

    def search_references(self):
        try:
            keywords = self.search_entry.get().strip().split()
            if not keywords:
                messagebox.showwarning("Warning", "Please enter search keywords")
                return

            response = requests.post(
                f"{self.api_base_url}/references/search",
                headers={"Authorization": f"Bearer {self.token}"},
                json={"keywords": keywords}
            )

            # 트리뷰 초기화
            for item in self.refs_tree.get_children():
                self.refs_tree.delete(item)

            if response.status_code == 200:
                references = response.json()
                for ref in references:
                    self.refs_tree.insert("", tk.END, values=(
                        ref["title"],
                        ", ".join(ref["authors"]),
                        ref["publication_date"]
                    ))
            else:
                logging.error(f"Failed to search references: {response.status_code}")
                messagebox.showerror("Error", "Failed to search references")

        except Exception as e:
            logging.error(f"Error searching references: {str(e)}")
            messagebox.showerror("Error", f"Failed to search references: {str(e)}")

    def on_reference_select(self, event):
        selection = self.refs_tree.selection()
        if not selection:
            return

        item = self.refs_tree.item(selection[0])
        title = item['values'][0]