#* «Copyright 2026 Roberto Reinosa Fernández»
#*
#* This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#*

import customtkinter as ctk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import dual_annealing, fsolve
from fpdf import FPDF
import os
import threading
from datetime import datetime
import warnings
from itertools import combinations

warnings.filterwarnings("ignore")

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

COLOR_BG = "#0f172a"
COLOR_SIDEBAR = "#020617"
COLOR_CARD = "#1e293b"
COLOR_ACCENT = "#0ea5e9"
COLOR_SUCCESS = "#10b981"
COLOR_DANGER = "#ef4444"
COLOR_TEXT = "#f8fafc"
COLOR_SUBTEXT = "#94a3b8"


def calculate_empirical_curves(positives, negatives):
    all_values = np.concatenate([positives, negatives])
    thresholds = np.unique(all_values)
    thresholds.sort()

    n_pos = len(positives)
    n_neg = len(negatives)

    if len(thresholds) * (n_pos + n_neg) < 10 ** 8:
        tp = np.sum(positives >= thresholds[:, None], axis=1)
        tn = np.sum(negatives < thresholds[:, None], axis=1)
    else:
        tp = np.array([np.sum(positives >= t) for t in thresholds])
        tn = np.array([np.sum(negatives < t) for t in thresholds])

    sens = tp / n_pos if n_pos > 0 else np.zeros_like(tp)
    spec = tn / n_neg if n_neg > 0 else np.zeros_like(tn)
    return thresholds, sens, spec


def exact_interpolation(thresholds, sens, spec):
    x_interp = np.linspace(thresholds.min(), thresholds.max(), 5000)
    y_sens = np.interp(x_interp, thresholds, sens)
    y_spec = np.interp(x_interp, thresholds, spec)
    idx = np.argmin(np.abs(y_sens - y_spec))
    return x_interp[idx], y_sens[idx]


def logistic_2p(x, k, x0):
    arg = np.clip(-k * (x - x0), -100, 100)
    return 1 / (1 + np.exp(arg))


def logistic_4p(x, L, A, k, x0):
    arg = np.clip(-k * (x - x0), -100, 100)
    return L + A / (1 + np.exp(arg))


def objective_function(params, x_data, y_data, model_func):
    y_pred = model_func(x_data, *params)
    return np.sum((y_data - y_pred) ** 2)


def robust_optimization(model_func, bounds, args, max_iter, init_temp, visit, n_runs=5, log_callback=None,
                        cancel_event=None):
    best_res = None
    best_error = np.inf

    def da_callback(x, f, context):
        if cancel_event and cancel_event.is_set():
            return True
        return False

    for i in range(n_runs):
        if cancel_event and cancel_event.is_set(): break
        seed = 42 + i * 123
        res = dual_annealing(objective_function, bounds=bounds, args=args, seed=seed, maxiter=max_iter,
                             initial_temp=init_temp, visit=visit, no_local_search=False, callback=da_callback)
        if res.fun < best_error:
            best_error = res.fun
            best_res = res

    return best_res.x if best_res else None


def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0


def evaluate_test_set(test_df, var, target, cutoff):
    if test_df is None or var not in test_df.columns or target not in test_df.columns:
        return None, None
    data = test_df[[var, target]].dropna()
    vals = pd.to_numeric(data[var], errors='coerce').dropna().values
    classes = data[target].loc[data[var].index].values
    pos = vals[classes == 1]
    neg = vals[classes == 0]
    n_pos, n_neg = len(pos), len(neg)
    sens = np.sum(pos >= cutoff) / n_pos if n_pos > 0 else 0
    spec = np.sum(neg < cutoff) / n_neg if n_neg > 0 else 0
    return sens, spec


def optimize_thresholdxpert(df, features, target_col, is_fast, cancel_event):
    data = df[list(features) + [target_col]].dropna()
    y_true = data[target_col].values
    X_vals = data[list(features)].values

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        return np.zeros(len(features)), 0.0, 0.0, [False] * len(features)

    bounds = [(np.min(X_vals[:, i]), np.max(X_vals[:, i])) for i in range(len(features))]

    n_points = 1_000_000 if is_fast else 10_000_000
    rng = np.random.default_rng(42)

    all_thresholds = rng.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(n_points, len(features))
    )

    all_sens = np.zeros(n_points, dtype=np.float32)
    all_spec = np.zeros(n_points, dtype=np.float32)

    chunk_size = 100_000
    pos_mask = (y_true == 1)[None, :]
    neg_mask = (y_true == 0)[None, :]

    for i in range(0, n_points, chunk_size):
        if cancel_event.is_set():
            return np.zeros(len(features)), 0.0, 0.0, [False] * len(features)

        t_chunk = all_thresholds[i:i + chunk_size]
        preds = (X_vals[None, :, :] >= t_chunk[:, None, :]).any(axis=-1)

        all_sens[i:i + chunk_size] = np.sum(preds & pos_mask, axis=1) / n_pos
        all_spec[i:i + chunk_size] = np.sum((~preds) & neg_mask, axis=1) / n_neg

    min_sens_spec = np.minimum(all_sens, all_spec)
    max_balanced = np.max(min_sens_spec)
    last_s = np.floor(max_balanced * 1000) / 1000.0
    last_sp = last_s

    valid_mask_phase1 = (all_sens >= last_s - 1e-6)
    if np.any(valid_mask_phase1):
        max_spec_for_last_s = np.max(all_spec[valid_mask_phase1])
        last_sp = np.floor(max_spec_for_last_s * 1000) / 1000.0

    final_valid = (all_sens >= last_s - 1e-6) & (all_spec >= last_sp - 1e-6)

    valid_sens = all_sens[final_valid]
    valid_spec = all_spec[final_valid]
    valid_thresholds = all_thresholds[final_valid]

    if len(valid_sens) == 0:
        return np.zeros(len(features)), 0.0, 0.0, [False] * len(features)

    sens_round = np.round(valid_sens, 4)
    spec_round = np.round(valid_spec, 4)
    unique_pairs, inverse_indices = np.unique(np.c_[sens_round, spec_round], axis=0, return_inverse=True)

    best_pair_idx = np.lexsort((unique_pairs[:, 0], unique_pairs[:, 1]))[-1]
    best_mask = (inverse_indices == best_pair_idx)
    candidate_thresholds = valid_thresholds[best_mask]

    if len(candidate_thresholds) >= 10:
        chosen_10 = candidate_thresholds[rng.choice(len(candidate_thresholds), 10, replace=False)]
    else:
        chosen_10 = candidate_thresholds

    mean_t = np.mean(chosen_10, axis=0)

    final_preds = (X_vals >= mean_t).any(axis=1)
    final_sens = np.sum(final_preds & (y_true == 1)) / n_pos if n_pos > 0 else 0.0
    final_spec = np.sum((~final_preds) & (y_true == 0)) / n_neg if n_neg > 0 else 0.0

    feature_ranges = np.max(X_vals, axis=0) - np.min(X_vals, axis=0)
    feature_ranges[feature_ranges == 0] = 1e-9
    std_t = np.std(chosen_10, axis=0)

    fluctuations = (std_t / feature_ranges) > 0.15
    txp_flags = fluctuations.tolist()

    return mean_t, final_sens, final_spec, txp_flags


def eval_txp_test(test_df, features, target_col, thresholds):
    if test_df is None: return None, None
    data = test_df[list(features) + [target_col]].dropna()
    if len(data) == 0: return None, None
    y_true = data[target_col].values
    X_vals = data[list(features)].values
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    preds = (X_vals >= thresholds).any(axis=1)
    sens = np.sum(preds & (y_true == 1)) / n_pos if n_pos > 0 else 0
    spec = np.sum((~preds) & (y_true == 0)) / n_neg if n_neg > 0 else 0

    return sens, spec


class ProfessionalPDF(FPDF):
    def header(self):
        self.set_fill_color(14, 165, 233)
        self.rect(0, 0, 210, 20, 'F')
        self.set_y(5)
        self.set_font('Arial', 'B', 12)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, 'TholdStormDX ANALYTICS | AUTOMATED CLINICAL REPORT', 0, 1, 'R')
        self.ln(15)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, f'Generated by TholdStormDX Elite v0.0.1 | Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, label):
        self.set_font('Arial', 'B', 12)
        self.set_text_color(15, 23, 42)
        self.cell(0, 8, label, 'B', 1, 'L')
        self.ln(4)

    def add_analysis_page(self, var_name, results, img_path, val_available, test_available):
        self.add_page()
        self.set_fill_color(241, 245, 249)
        self.rect(10, 30, 190, 25, 'F')
        self.set_xy(15, 32)
        self.set_font('Arial', 'B', 16)
        self.set_text_color(30)
        self.cell(100, 10, f"Biomarker: {var_name}", 0, 0)
        self.set_xy(15, 42)
        self.set_font('Arial', '', 10)
        self.set_text_color(100)
        self.cell(100, 6, f"Processed: {datetime.now().strftime('%d-%b-%Y %H:%M')}", 0, 1)
        self.ln(15)

        self.chapter_title('1. Optimization Results')
        self.set_font('Arial', 'B', 8)
        self.set_fill_color(30, 41, 59)
        self.set_text_color(255)

        cols = [45, 20, 30, 30, 30, 35]
        for i, h in enumerate(['MODEL', 'CUT-OFF', 'TRAIN (SE/SP)', 'VAL (SE/SP)', 'TEST (SE/SP)', 'R2 SCORE']):
            self.cell(cols[i], 8, h, 0, 0, 'C', 1)
        self.ln()

        self.set_text_color(0)
        data = [
            ('Empirical (Exact)', results['emp'], False),
            ('Logistic 2-Parameter', results['2p'], False),
            ('Logistic 4-Parameter (Rec.)', results['4p'], True),
            ('ThresholdXpert (Stochastic)', results['txp'], False)
        ]

        fill = False
        for name, d, bold in data:
            self.set_font('Arial', 'B' if bold else '', 8)
            self.set_fill_color(248, 250, 252) if fill else self.set_fill_color(255)

            r2_txt = f"{d['r2']:.4f}" if d['r2'] is not None else "N/A"
            train_txt = f"{d['train_sens']:.3f} / {d['train_spec']:.3f}"
            val_txt = f"{d['val_sens']:.3f} / {d['val_spec']:.3f}" if val_available and d.get(
                'val_sens') is not None else "N/A"
            test_txt = f"{d['test_sens']:.3f} / {d['test_spec']:.3f}" if test_available and d.get(
                'test_sens') is not None else "N/A"

            self.cell(cols[0], 8, name, 'LR', 0, 'L', fill)
            self.cell(cols[1], 8, f"{d['cut']:.4f}", 'LR', 0, 'C', fill)
            self.cell(cols[2], 8, train_txt, 'LR', 0, 'C', fill)
            self.cell(cols[3], 8, val_txt, 'LR', 0, 'C', fill)
            self.cell(cols[4], 8, test_txt, 'LR', 0, 'C', fill)
            self.cell(cols[5], 8, r2_txt, 'LR', 1, 'C', fill)
            fill = not fill

        self.cell(sum(cols), 0, '', 'T')
        self.ln(10)

        self.chapter_title('2. Diagnostic Performance Curves (Training)')
        if os.path.exists(img_path):
            self.image(img_path, x=10, w=190)
            self.ln(5)

    def add_top_panels_page(self, top_panels, sorting_metric):
        self.add_page()
        self.chapter_title('Top 200 Combinatorial Panels (ThresholdXpert OR-Logic)')
        self.set_font('Arial', '', 9)

        intro_text = (
            "The following multimarker panels have been optimized using high-performance vector-driven "
            "Monte Carlo simulations under a Boolean OR-logic framework. The engine employs a Max-Min "
            "Balancing logic (0.001 precision) to identify global threshold configurations that maximize "
            "the equilibrium between Sensitivity and Specificity across up to 10 million iterations. "
            f"To ensure clinical robustness, results are sorted strictly by {sorting_metric}. "
            "(* Asterisk indicates an algorithmic threshold instability > 15%, suggesting potential "
            "data sparsity or high variance in the stochastic averaging process)."
        )

        self.multi_cell(0, 5, intro_text)
        self.ln(5)

        for i, panel in enumerate(top_panels):
            self.set_font('Arial', 'B', 10)
            self.set_text_color(15, 23, 42)
            self.cell(0, 8, f"#{i + 1}: {panel['Panel']}", 0, 1, 'L')

            self.set_font('Arial', '', 9)
            self.set_text_color(50)
            self.multi_cell(0, 5, f"Optimized Thresholds: {panel['Thresholds']}")

            score_txt = f"Train Sens: {panel['Train_Sens']:.3f} | Train Spec: {panel['Train_Spec']:.3f}  [TRAIN SCORE: {panel['Score']:.3f}]"
            if panel.get('Val_Sens') is not None:
                score_txt += f"  ||  Val Sens: {panel['Val_Sens']:.3f} | Val Spec: {panel['Val_Spec']:.3f}  [VAL SCORE: {panel['Val_Score']:.3f}]"
            if panel.get('Test_Sens') is not None:
                score_txt += f"  ||  Test Sens: {panel['Test_Sens']:.3f} | Test Spec: {panel['Test_Spec']:.3f}  [TEST SCORE: {panel['Test_Score']:.3f}]"

            self.set_text_color(2, 132, 199)
            self.multi_cell(0, 6, score_txt)
            self.ln(2)


class TholdStormDXApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("TholdStormDX Elite - v0.0.1 Mega Combinatorial")
        self.geometry("1150x850")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.cancel_event = threading.Event()

        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0, fg_color=COLOR_SIDEBAR)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(4, weight=1)

        self.lbl_brand = ctk.CTkLabel(self.sidebar, text="TholdStormDX", font=("Montserrat", 26, "bold"),
                                      text_color="white")
        self.lbl_brand.grid(row=0, column=0, padx=20, pady=(40, 5), sticky="w")

        ctk.CTkLabel(self.sidebar, text="ELITE EDITION v0.0.1", font=("Roboto", 10, "bold"),
                     text_color=COLOR_ACCENT).grid(row=1, column=0, padx=20, sticky="w")

        info_text = (
            "\n\n\nCORE FEATURES:\n\n• Exact Interpolation\n• Dual Annealing Models\n• ThresholdXpert Logic\n• Vectorized Combo (OR)\n• Fast/Elite Switch\n• Anti-Overfit Sorting")
        ctk.CTkLabel(self.sidebar, text=info_text, font=("Roboto", 12), text_color=COLOR_SUBTEXT, justify="left").grid(
            row=2, column=0, padx=20, sticky="w")

        self.lbl_status = ctk.CTkLabel(self.sidebar, text="● SYSTEM READY", font=("Roboto Mono", 11),
                                       text_color=COLOR_SUCCESS)
        self.lbl_status.grid(row=4, column=0, padx=20, pady=20, sticky="sw")

        self.main = ctk.CTkFrame(self, corner_radius=0, fg_color=COLOR_BG)
        self.main.grid(row=0, column=1, sticky="nsew")
        self.main.grid_columnconfigure(0, weight=1)
        self.main.grid_rowconfigure(2, weight=1)

        self.card_cfg = ctk.CTkFrame(self.main, fg_color=COLOR_CARD, corner_radius=12)
        self.card_cfg.grid(row=0, column=0, padx=40, pady=(40, 20), sticky="ew")
        self.card_cfg.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(self.card_cfg, text="DATA CONFIGURATION", font=("Roboto", 12, "bold"),
                     text_color=COLOR_SUBTEXT).grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")

        self.ent_in = ctk.CTkEntry(self.card_cfg, placeholder_text="Path to Training CSV file...", height=40, width=500)
        self.ent_in.grid(row=1, column=0, padx=20, pady=(0, 15), sticky="ew")

        ctk.CTkButton(self.card_cfg, text="BROWSE TRAIN", width=200, height=40, fg_color="#334155",
                      hover_color="#475569", command=self.browse_in).grid(row=1, column=1, padx=(0, 20), pady=(0, 15))

        self.ent_val = ctk.CTkEntry(self.card_cfg, placeholder_text="Path to Validation CSV file (Optional)...",
                                    height=40, width=500)
        self.ent_val.grid(row=2, column=0, padx=20, pady=(0, 15), sticky="ew")

        ctk.CTkButton(self.card_cfg, text="BROWSE VALIDATION", width=200, height=40, fg_color="#334155",
                      hover_color="#475569", command=self.browse_val).grid(row=2, column=1, padx=(0, 20), pady=(0, 15))

        self.ent_test = ctk.CTkEntry(self.card_cfg, placeholder_text="Path to Test CSV file (Optional)...", height=40,
                                     width=500)
        self.ent_test.grid(row=3, column=0, padx=20, pady=(0, 15), sticky="ew")

        ctk.CTkButton(self.card_cfg, text="BROWSE TEST", width=200, height=40, fg_color="#334155",
                      hover_color="#475569", command=self.browse_test).grid(row=3, column=1, padx=(0, 20), pady=(0, 15))

        self.ent_out = ctk.CTkEntry(self.card_cfg, placeholder_text="Output Directory...", height=40, width=500)
        self.ent_out.grid(row=4, column=0, padx=20, pady=(0, 20), sticky="ew")

        ctk.CTkButton(self.card_cfg, text="BROWSE OUTPUT", width=200, height=40, fg_color="#334155", hover_color="#475569",
                      command=self.browse_out).grid(row=4, column=1, padx=(0, 20), pady=(0, 20))

        self.frm_exec = ctk.CTkFrame(self.main, fg_color="transparent")
        self.frm_exec.grid(row=1, column=0, padx=40, pady=0, sticky="ew")
        self.frm_exec.grid_columnconfigure(0, weight=1)

        self.switch_fast = ctk.CTkSwitch(self.frm_exec, text="Fast-Track Mode (Testing: 1 Million ops)",
                                         font=("Roboto", 12),
                                         progress_color=COLOR_ACCENT)
        self.switch_fast.grid(row=0, column=0, sticky="w", pady=(0, 10))

        self.frm_btns = ctk.CTkFrame(self.frm_exec, fg_color="transparent")
        self.frm_btns.grid(row=1, column=0, sticky="ew")
        self.frm_btns.grid_columnconfigure(0, weight=3)
        self.frm_btns.grid_columnconfigure(1, weight=1)

        self.btn_run = ctk.CTkButton(self.frm_btns, text="START COMBO ANALYSIS", font=("Roboto", 14, "bold"), height=55,
                                     fg_color=COLOR_ACCENT, hover_color="#0284c7", command=self.start_process)
        self.btn_run.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        self.btn_cancel = ctk.CTkButton(self.frm_btns, text="CANCEL", font=("Roboto", 14, "bold"), height=55,
                                        fg_color=COLOR_DANGER, hover_color="#b91c1c", state="disabled",
                                        command=self.cancel_process)
        self.btn_cancel.grid(row=0, column=1, sticky="ew")

        self.progress = ctk.CTkProgressBar(self.frm_exec, height=6, progress_color=COLOR_SUCCESS)
        self.progress.grid(row=2, column=0, sticky="ew", pady=(15, 0))
        self.progress.set(0)

        self.card_console = ctk.CTkFrame(self.main, fg_color="#0f172a", corner_radius=12, border_width=1,
                                         border_color="#334155")
        self.card_console.grid(row=2, column=0, padx=40, pady=30, sticky="nsew")

        ctk.CTkLabel(self.card_console, text=" >_ LIVE EXECUTION LOG", font=("Consolas", 10, "bold"),
                     text_color="#64748b").pack(anchor="w", padx=15, pady=10)

        self.console = ctk.CTkTextbox(self.card_console, font=("Consolas", 11), fg_color="transparent",
                                      text_color=COLOR_TEXT)
        self.console.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.log("TholdStormDX Elite initialized.")


    def log(self, msg):
        self.after(0, self._log_ui, msg)

    def _log_ui(self, msg):
        self.console.configure(state='normal')
        self.console.insert('end', f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
        self.console.see('end')
        self.console.configure(state='disabled')

    def set_status(self, msg, color=COLOR_TEXT):
        self.after(0, self._set_status_ui, msg, color)

    def _set_status_ui(self, msg, color):
        self.lbl_status.configure(text=f"● {msg}", text_color=color)

    def set_progress(self, val):
        self.after(0, self.progress.set, val)

    def update_ui_buttons(self, is_running):
        self.after(0, self._update_buttons_ui, is_running)

    def _update_buttons_ui(self, is_running):
        if is_running:
            self.btn_run.configure(state="disabled", text="OPTIMIZING...")
            self.btn_cancel.configure(state="normal", text="CANCEL")
            self.switch_fast.configure(state="disabled")
        else:
            self.btn_run.configure(state="normal", text="START COMBO ANALYSIS")
            self.btn_cancel.configure(state="disabled", text="CANCEL")
            self.switch_fast.configure(state="normal")

    def show_message(self, title, msg, is_error=False):
        def _show():
            if is_error:
                messagebox.showerror(title, msg)
            else:
                messagebox.showinfo(title, msg)

        self.after(0, _show)


    def browse_in(self):
        f = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if f:
            self.ent_in.delete(0, 'end')
            self.ent_in.insert(0, f)
            if not self.ent_out.get():
                self.ent_out.insert(0, os.path.dirname(f))
            self.log(f"Training dataset loaded: {os.path.basename(f)}")

    def browse_val(self):
        f = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if f:
            self.ent_val.delete(0, 'end')
            self.ent_val.insert(0, f)
            self.log(f"Validation dataset loaded: {os.path.basename(f)}")

    def browse_test(self):
        f = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if f:
            self.ent_test.delete(0, 'end')
            self.ent_test.insert(0, f)
            self.log(f"Test dataset loaded: {os.path.basename(f)}")

    def browse_out(self):
        d = filedialog.askdirectory()
        if d:
            self.ent_out.delete(0, 'end')
            self.ent_out.insert(0, d)

    def cancel_process(self):
        if not self.cancel_event.is_set():
            self.log("WARNING: Cancellation requested. Halting mathematical engine...")
            self.set_status("CANCELLING...", COLOR_DANGER)
            self.after(0, lambda: self.btn_cancel.configure(state="disabled", text="ABORTING..."))
            self.cancel_event.set()

    def start_process(self):
        self.cancel_event.clear()
        threading.Thread(target=self.run_engine,
                         args=(self.ent_in.get(), self.ent_val.get(), self.ent_test.get(), self.ent_out.get()),
                         daemon=True).start()

    def run_engine(self, in_p, val_p, test_p, out_p):
        if not in_p:
            self.log("ERROR: No training dataset provided.")
            self.show_message("Error", "Please configure the input training CSV file first.", is_error=True)
            return

        self.update_ui_buttons(is_running=True)
        self.set_status("PROCESSING STAGE 1 (INDIVIDUAL)", "#fbbf24")
        self.set_progress(0)

        is_fast = self.switch_fast.get() == 1
        runs = 1 if is_fast else 5
        iter_2p = 500 if is_fast else 3000
        iter_4p = 500 if is_fast else 2000

        self.log(
            f"MODE SELECTED: {'FAST-TRACK (1M vectors)' if is_fast else 'ELITE (10M vectors)'}")

        try:
            df = pd.read_csv(in_p, sep=';', decimal='.')
            target = df.columns[-1]
            vars_list = df.columns[:-1]
            total_vars = len(vars_list)

            df_val = None
            if val_p and os.path.exists(val_p):
                df_val = pd.read_csv(val_p, sep=';', decimal='.')

            df_test = None
            if test_p and os.path.exists(test_p):
                df_test = pd.read_csv(test_p, sep=';', decimal='.')

            pdf = ProfessionalPDF()
            pages_generated = 0

            for i, var in enumerate(vars_list):
                if self.cancel_event.is_set(): break

                self.log(f"\n--- Analyzing Phase 1 (Single Variable): {var} [{i + 1}/{total_vars}] ---")
                data = df[[var, target]].copy()
                data[var] = pd.to_numeric(data[var], errors='coerce')
                data = data.dropna()
                vals, classes = data[var].values, data[target].values
                pos, neg = vals[classes == 1], vals[classes == 0]

                if len(pos) < 5 or len(neg) < 5:
                    self.log(f"WARNING: Skipping {var}, not enough data.")
                    continue

                th_real, sens_real, spec_real = calculate_empirical_curves(pos, neg)
                cut_emp, val_emp = exact_interpolation(th_real, sens_real, spec_real)

                v_se_emp, v_sp_emp = evaluate_test_set(df_val, var, target, cut_emp)
                t_se_emp, t_sp_emp = evaluate_test_set(df_test, var, target, cut_emp)

                min_v, max_v = vals.min(), vals.max()

                b2 = [(-2000, 2000), (min_v, max_v)]
                b4 = [(-0.1, 1.1), (0.0, 1.2), (-1000, 1000), (min_v, max_v)]

                self.log(" > Optimizing 2-Parameter Dual Annealing...")
                p_s2 = robust_optimization(logistic_2p, b2, (th_real, sens_real, logistic_2p), iter_2p, 5000, 2.6, runs,
                                           cancel_event=self.cancel_event)
                p_e2 = robust_optimization(logistic_2p, b2, (th_real, spec_real, logistic_2p), iter_2p, 5000, 2.6, runs,
                                           cancel_event=self.cancel_event)
                if self.cancel_event.is_set() or p_s2 is None or p_e2 is None: break

                cut_2p, val_2p = 0.0, 0.0
                try:
                    k_s, x0_s = p_s2
                    k_e, x0_e = p_e2
                    cut_2p = float((k_s * x0_s - k_e * x0_e) / (k_s - k_e))
                    val_2p = float(logistic_2p(cut_2p, *p_s2))
                except:
                    pass
                r2_2p = (calculate_r2(sens_real, logistic_2p(th_real, *p_s2)) + calculate_r2(spec_real,
                                                                                             logistic_2p(th_real,
                                                                                                         *p_e2))) / 2
                v_se_2p, v_sp_2p = evaluate_test_set(df_val, var, target, cut_2p)
                t_se_2p, t_sp_2p = evaluate_test_set(df_test, var, target, cut_2p)

                self.log(" > Optimizing 4-Parameter Dual Annealing (Advanced Fit)...")
                p_s4 = robust_optimization(logistic_4p, b4, (th_real, sens_real, logistic_4p), iter_4p, 5000, 2.5, runs,
                                           cancel_event=self.cancel_event)
                p_e4 = robust_optimization(logistic_4p, b4, (th_real, spec_real, logistic_4p), iter_4p, 5000, 2.5, runs,
                                           cancel_event=self.cancel_event)
                if self.cancel_event.is_set() or p_s4 is None or p_e4 is None: break

                cut_4p, val_4p = 0.0, 0.0
                try:
                    def encontrar_cruce(x):
                        return logistic_4p(x, *p_s4) - logistic_4p(x, *p_e4)

                    cut_4p = float(fsolve(encontrar_cruce, (p_s4[3] + p_e4[3]) / 2)[0])
                    val_4p = float(logistic_4p(cut_4p, *p_s4))
                except:
                    pass
                r2_4p = (calculate_r2(sens_real, logistic_4p(th_real, *p_s4)) + calculate_r2(spec_real,
                                                                                             logistic_4p(th_real,
                                                                                                         *p_e4))) / 2
                v_se_4p, v_sp_4p = evaluate_test_set(df_val, var, target, cut_4p)
                t_se_4p, t_sp_4p = evaluate_test_set(df_test, var, target, cut_4p)

                self.log(" > Scanning stochastic threshold bounds ...")
                txp_cuts, txp_train_sens, txp_train_spec, txp_flags = optimize_thresholdxpert(df, [var], target,
                                                                                              is_fast,
                                                                                              self.cancel_event)
                txp_cut = txp_cuts[0]
                txp_val_sens, txp_val_spec = eval_txp_test(df_val, [var], target, txp_cuts)
                txp_test_sens, txp_test_spec = eval_txp_test(df_test, [var], target, txp_cuts)

                res = {
                    'emp': {'cut': cut_emp, 'train_sens': val_emp, 'train_spec': val_emp, 'r2': None,
                            'val_sens': v_se_emp, 'val_spec': v_sp_emp,
                            'test_sens': t_se_emp, 'test_spec': t_sp_emp},
                    '2p': {'cut': cut_2p, 'train_sens': val_2p, 'train_spec': val_2p, 'r2': r2_2p,
                           'val_sens': v_se_2p, 'val_spec': v_sp_2p,
                           'test_sens': t_se_2p, 'test_spec': t_sp_2p},
                    '4p': {'cut': cut_4p, 'train_sens': val_4p, 'train_spec': val_4p, 'r2': r2_4p,
                           'val_sens': v_se_4p, 'val_spec': v_sp_4p,
                           'test_sens': t_se_4p, 'test_spec': t_sp_4p},
                    'txp': {'cut': txp_cut, 'train_sens': txp_train_sens, 'train_spec': txp_train_spec, 'r2': None,
                            'val_sens': txp_val_sens, 'val_spec': txp_val_spec,
                            'test_sens': txp_test_sens, 'test_spec': txp_test_spec}
                }

                plt.style.use('seaborn-v0_8-whitegrid')
                fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
                ax.scatter(th_real, sens_real, c='#38bdf8', s=8, alpha=0.3, label='Empirical Sens')
                ax.scatter(th_real, spec_real, c='#fb923c', s=8, alpha=0.3, label='Empirical Spec')
                x_plt = np.linspace(min_v, max_v, 2000)
                ax.plot(x_plt, logistic_2p(x_plt, *p_s2), color='#94a3b8', linestyle=':', linewidth=1.5,
                        label='2P Model')
                ax.plot(x_plt, logistic_2p(x_plt, *p_e2), color='#94a3b8', linestyle=':', linewidth=1.5)
                ax.plot(x_plt, logistic_4p(x_plt, *p_s4), color='#0284c7', linewidth=2.5, label='4P Sens')
                ax.plot(x_plt, logistic_4p(x_plt, *p_e4), color='#ea580c', linewidth=2.5, label='4P Spec')
                ax.axvline(cut_emp, color='#64748b', linestyle='--', alpha=0.8, label=f'Empirical: {cut_emp:.2f}')
                ax.axvline(cut_4p, color='#dc2626', linestyle='-', linewidth=1.5, label=f'Optimal 4P: {cut_4p:.2f}')
                ax.set_title(f"{var}: Robust Diagnostic Optimization", fontsize=12, fontweight='bold', pad=15)
                ax.set_xlabel("Biomarker Value", fontsize=10)
                ax.set_ylabel("Probability", fontsize=10)
                ax.legend(frameon=True, fancybox=True, framealpha=0.9, fontsize=9)
                ax.grid(True, linestyle='--', alpha=0.5)

                img_path = os.path.join(out_p, f"plot_{i}.png")
                plt.savefig(img_path, bbox_inches='tight')
                plt.close()

                pdf.add_analysis_page(var, res, img_path, val_available=(df_val is not None),
                                      test_available=(df_test is not None))
                if os.path.exists(img_path): os.remove(img_path)
                pages_generated += 1
                self.set_progress((i + 1) / total_vars * 0.5)

            if not self.cancel_event.is_set():
                self.log("\n=======================================================")
                self.log("STAGE 2: VECTORIZED MULTIVARIABLE COMBINATORIAL ACTIVE")
                self.log("=======================================================")
                self.set_status("PROCESSING STAGE 2 (MULTIMARKERS)", "#c084fc")

                total_combos = sum(1 for k in range(2, total_vars + 1) for _ in combinations(vars_list, k))
                self.log(f"Initiating exhaustive matching over {total_combos} discrete configurations...")

                all_panels = []
                current_combo = 0

                for k in range(2, total_vars + 1):
                    for combo in combinations(vars_list, k):
                        if self.cancel_event.is_set(): break
                        current_combo += 1

                        if current_combo % 10 == 0 or current_combo == total_combos:
                            self.log(
                                f" > Combinatorics Progress: [{current_combo}/{total_combos}] panels evaluated...")

                        txp_cuts, txp_train_sens, txp_train_spec, txp_flags = optimize_thresholdxpert(df, combo, target,
                                                                                                      is_fast,
                                                                                                      self.cancel_event)
                        txp_val_sens, txp_val_spec = eval_txp_test(df_val, combo, target, txp_cuts)
                        txp_test_sens, txp_test_spec = eval_txp_test(df_test, combo, target, txp_cuts)

                        panel_names = [f"{c}*" if f else c for c, f in zip(combo, txp_flags)]
                        panel_str = ' + '.join(panel_names)
                        thresholds_str = ' | '.join([f"{n}: {t:.4f}" for n, t in zip(panel_names, txp_cuts)])

                        score_val = txp_train_sens + txp_train_spec

                        val_score_val = (txp_val_sens + txp_val_spec) if txp_val_sens is not None else None
                        test_score_val = (txp_test_sens + txp_test_spec) if txp_test_sens is not None else None

                        panel_data = {
                            'Panel': panel_str,
                            'Thresholds': thresholds_str,
                            'Train_Sens': txp_train_sens,
                            'Train_Spec': txp_train_spec,
                            'Score': score_val,
                            'Val_Sens': txp_val_sens,
                            'Val_Spec': txp_val_spec,
                            'Val_Score': val_score_val,
                            'Test_Sens': txp_test_sens,
                            'Test_Spec': txp_test_spec,
                            'Test_Score': test_score_val
                        }
                        all_panels.append(panel_data)

                        self.set_progress(0.5 + (current_combo / total_combos * 0.5))

                if len(all_panels) > 0 and not self.cancel_event.is_set():
                    panels_df = pd.DataFrame(all_panels)

                    sorting_metric_name = "Training Performance"
                    if 'Val_Score' in panels_df.columns and panels_df['Val_Score'].notna().any():
                        panels_df.sort_values('Val_Score', ascending=False, inplace=True)
                        sorting_metric_name = "Validation Performance"
                    else:
                        panels_df.sort_values('Score', ascending=False, inplace=True)

                    csv_path = os.path.join(out_p,
                                            f"TholdStormDX_Panels_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                    panels_df.to_csv(csv_path, index=False, sep=';', decimal='.')
                    self.log(
                        f"\nCSV DISK WRITE COMPLETE: Full Multivariable Combinations dumped to -> {os.path.basename(csv_path)}")

                    top_200 = panels_df.head(200).to_dict('records')
                    pdf.add_top_panels_page(top_200, sorting_metric_name)

            if pages_generated > 0:
                prefix = "TholdStormDX_PARTIAL_" if self.cancel_event.is_set() else "TholdStormDX_RobustReport_"
                fname = f"{prefix}{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                pdf.output(os.path.join(out_p, fname))
                self.log(f"Master Analytical Clinical PDF constructed & saved ->: {fname}")

            if self.cancel_event.is_set():
                self.log("\n--- ENGINE TERMINATED VIA MANUAL OVERRIDE. ---")
                self.set_status("CANCELLED", COLOR_DANGER)
            else:
                self.log("\n=== ALL DIRECTIVES MET. PROCESS FULLY FINISHED. ===")
                self.set_status("COMPLETED", COLOR_SUCCESS)
                self.show_message("TholdStormDX Elite", "Global Diagnostic Panel Calculations Completely Sucessful.")

        except Exception as e:
            self.log(f"RUNTIME PANIC: {e}")
            self.set_status("CRITICAL EXCEPTION", COLOR_DANGER)
            self.show_message("Runtime Math Engine Exception", str(e), is_error=True)

        finally:
            self.update_ui_buttons(is_running=False)


if __name__ == "__main__":
    app = TholdStormDXApp()
    app.mainloop()