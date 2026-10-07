import os, threading, tkinter as tk
from tkinter import ttk, messagebox
import requests
API = os.environ.get("API_URL", "http://127.0.0.1:8000")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Cyber Threat Classifier"); self.geometry("880x740")
        pad = dict(padx=10, pady=4)
        ttk.Label(self, text="Text to analyse").pack(anchor="w", **pad)
        self.text = tk.Text(self, height=8, wrap="word"); self.text.pack(fill="x", **pad)
        row = ttk.Frame(self); row.pack(fill="x", **pad)
        ttk.Label(row, text="Entity (exact words from the text):").pack(side="left")
        self.entity = ttk.Entry(row, width=36); self.entity.pack(side="left", padx=6)
        self.btn = ttk.Button(row, text="Classify", command=self.classify); self.btn.pack(side="left")
        ttk.Button(row, text="Clear", command=self.clear).pack(side="left", padx=6)
        self.status = ttk.Label(self, text="Checking API...", foreground="gray")
        self.status.pack(anchor="w", **pad)
        self.result = ttk.Label(self, text="", font=("Segoe UI", 16, "bold"))
        self.result.pack(anchor="w", **pad)
        self.tree = ttk.Treeview(self, columns=("cls", "p"), show="headings", height=7)
        self.tree.heading("cls", text="Threat class"); self.tree.heading("p", text="Probability")
        self.tree.pack(fill="x", **pad)
        ttk.Label(self, text="Why? Words that pushed the prediction (darker red = stronger)"
                ).pack(anchor="w", **pad)
        self.expl = tk.Text(self, height=7, wrap="word", state="disabled")
        self.expl.pack(fill="both", expand=True, **pad)
        self.after(200, self.check_api)
 
    def check_api(self):
        try:
            info = requests.get(f"{API}/info", timeout=3).json()
            self.status.config(text=f"API online ({len(info['classes'])} classes)", foreground="green")
        except requests.RequestException:
            self.status.config(text="API offline - run: cd api && uvicorn main:app",
            foreground="red")
    
    def clear(self):
        self.text.delete("1.0", "end"); self.entity.delete(0, "end"); self.result.config(text="")
        self.tree.delete(*self.tree.get_children())
    
    def classify(self):
        txt, ent = self.text.get("1.0", "end").strip(), self.entity.get().strip()
        if not txt:
            return messagebox.showwarning("Input needed", "Please enter some text.")
        self.btn.config(state="disabled"); self.status.config(text="Working...", foreground="gray")
        threading.Thread(target=self._call, args=(txt, ent), daemon=True).start()
    
    def _call(self, txt, ent):
        try:
            r = requests.post(f"{API}/predict", timeout=20,
            json={"text": txt, "entity": ent or None, "explain": True})
            if r.status_code == 200:
                self.after(0, self._show, r.json())
            else:
                self.after(0, self._error, r.json().get("detail", r.text))
        except requests.RequestException as ex:
            self.after(0, self._error, f"API not reachable: {ex}")
    
    def _show(self, res):
        self.btn.config(state="normal"); self.status.config(text="Done", foreground="green")
        self.result.config(text=f"{res['label']} ({res['confidence']:.1%} confidence)")
        self.tree.delete(*self.tree.get_children())
        for c, p in sorted(res["probabilities"].items(), key=lambda kv: -kv[1]):
            self.tree.insert("", "end", values=(c, f"{p:.1%}"))
        ex = res.get("explanation", [])
        mx = max((abs(x["score"]) for x in ex), default=0) or 1e-9
        self.expl.config(state="normal"); self.expl.delete("1.0", "end")
        for i, x in enumerate(ex):
            shade = int(255 - 190 * max(x["score"], 0) / mx)
            self.expl.tag_configure(f"t{i}", background=f"#ff{shade:02x}{shade:02x}")
            self.expl.insert("end", x["token"] + " ", f"t{i}")
        self.expl.config(state="disabled")
    
    def _error(self, msg):
        self.btn.config(state="normal"); self.status.config(text="Error", foreground="red")
        messagebox.showerror("Request failed", str(msg))

if __name__ == "__main__":
    App().mainloop()