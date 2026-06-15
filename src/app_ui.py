"""
app_ui.py — BIM AI Engine Enterprise Interface
------------------------------------------------
Clean, professional interface for architectural firms.
API key authentication. Full conversion history. 
Auto-improves as the model gets trained.
"""

import streamlit as st
import requests
import time
import os
import pathlib
import datetime
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BIM AI Engine",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Data directory (where all converted files live on the user's machine) ──────
DATA_DIR   = pathlib.Path(os.getenv("DATA_DIR", str(pathlib.Path.home() / "BIM_AI_Engine_Data")))
OUTPUT_DIR = DATA_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Backend URL ─────────────────────────────────────────────────────────────────
try:
    BACKEND = st.secrets.get("BACKEND_URL", "http://127.0.0.1:8000")
except Exception:
    BACKEND = "http://127.0.0.1:8000"

# ── Global CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Base */
    html, body, .stApp {
        background-color: #080b10 !important;
        color: #dde3ee !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0d1117 !important;
        border-right: 1px solid #1a2035 !important;
    }
    section[data-testid="stSidebar"] * { color: #c9d1e0 !important; }

    /* Hide streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 13px 0 !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        width: 100% !important;
        letter-spacing: 0.3px !important;
        transition: opacity 0.15s ease !important;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.25) !important;
    }
    div.stButton > button:hover { opacity: 0.88 !important; }

    /* Input fields */
    .stTextInput input {
        background: #111827 !important;
        border: 1px solid #1f2d3d !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
        padding: 12px 16px !important;
        font-size: 15px !important;
    }
    .stTextInput input:focus {
        border-color: #10b981 !important;
        box-shadow: 0 0 0 2px rgba(16,185,129,0.15) !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #0d1117 !important;
        border: 2px dashed #1f2d3d !important;
        border-radius: 12px !important;
        padding: 12px !important;
        transition: border-color 0.2s;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #10b981 !important;
    }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #10b981, #34d399) !important;
        border-radius: 99px !important;
    }

    /* Metrics */
    [data-testid="metric-container"] {
        background: #0d1117 !important;
        border: 1px solid #1a2035 !important;
        border-radius: 10px !important;
        padding: 16px 20px !important;
    }
    [data-testid="metric-container"] label { color: #6b7fa3 !important; font-size: 12px !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
        font-size: 26px !important;
        font-weight: 700 !important;
    }

    /* Download buttons */
    [data-testid="stDownloadButton"] button {
        background: #111827 !important;
        border: 1px solid #1f2d3d !important;
        border-radius: 8px !important;
        color: #10b981 !important;
        font-weight: 600 !important;
        padding: 12px 0 !important;
        transition: all 0.15s !important;
    }
    [data-testid="stDownloadButton"] button:hover {
        background: #10b981 !important;
        color: #ffffff !important;
        border-color: #10b981 !important;
    }

    /* Success / warning / error boxes */
    .stSuccess { background: #062917 !important; border-color: #10b981 !important; }
    .stWarning { background: #1a1400 !important; border-color: #f59e0b !important; }
    .stError   { background: #1a0a0a !important; border-color: #ef4444 !important; }
    .stInfo    { background: #071525 !important; border-color: #3b82f6 !important; }

    /* History card */
    .history-card {
        background: #0d1117;
        border: 1px solid #1a2035;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 8px;
        transition: border-color 0.15s;
    }
    .history-card:hover { border-color: #10b981; }
    .history-name {
        font-size: 13px;
        font-weight: 600;
        color: #e2e8f0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .history-meta {
        font-size: 11px;
        color: #4b5f7c;
        margin-top: 3px;
    }

    /* Status line */
    .status-line {
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 12px;
        color: #10b981;
        padding: 2px 0;
        line-height: 1.8;
    }

    /* Login card */
    .login-card {
        background: #0d1117;
        border: 1px solid #1a2035;
        border-radius: 16px;
        padding: 40px 40px 32px;
        max-width: 460px;
        margin: 0 auto;
    }

    /* Divider */
    hr { border-color: #1a2035 !important; margin: 20px 0 !important; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: #080b10; }
    ::-webkit-scrollbar-thumb { background: #1a2035; border-radius: 99px; }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ──────────────────────────────────────────────────────
for key, val in [
    ("api_key", ""),
    ("authenticated", False),
    ("job_done", False),
    ("result", None),
]:
    if key not in st.session_state:
        st.session_state[key] = val


# ── Helper: check backend + validate key ───────────────────────────────────────
def check_api_key(key: str) -> bool:
    try:
        r = requests.get(
            f"{BACKEND}/health",
            headers={"x-api-key": key},
            timeout=6
        )
        return r.status_code == 200
    except Exception:
        return False


def backend_online() -> bool:
    try:
        r = requests.get(f"{BACKEND}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def format_filesize(path: Path) -> str:
    size = path.stat().st_size
    if size < 1024:
        return f"{size} B"
    elif size < 1024 ** 2:
        return f"{size/1024:.1f} KB"
    else:
        return f"{size/1024**2:.1f} MB"


def get_output_files():
    if not OUTPUT_DIR.exists():
        return [], []
    dxf = sorted(OUTPUT_DIR.glob("*.dxf"), key=lambda f: f.stat().st_mtime, reverse=True)
    csv = sorted(OUTPUT_DIR.glob("*.csv"), key=lambda f: f.stat().st_mtime, reverse=True)
    return dxf, csv


# ══════════════════════════════════════════════════════════════════════════════
# LOGIN SCREEN
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state["authenticated"]:

    # Centered layout
    spacer1, center, spacer2 = st.columns([1, 1.8, 1])

    with center:
        st.markdown("<br><br>", unsafe_allow_html=True)

        # Logo + title
        st.markdown("""
        <div style="text-align:center; margin-bottom: 32px;">
            <div style="font-size: 52px; margin-bottom: 12px;">📐</div>
            <h1 style="font-size: 28px; font-weight: 800; color: #e2e8f0;
                       letter-spacing: -0.5px; margin: 0;">BIM AI Engine</h1>
            <p style="color: #4b5f7c; font-size: 15px; margin-top: 8px;">
                Enterprise Edition
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="login-card">', unsafe_allow_html=True)

        st.markdown("""
        <p style="color: #6b7fa3; font-size: 13px; margin-bottom: 20px; text-align:center;">
            Enter your enterprise API key to access the platform
        </p>
        """, unsafe_allow_html=True)

        key_input = st.text_input(
            "Enterprise API Key",
            type="password",
            placeholder="bim_xxxxxxxxxxxxxxxxxxxxxxxx",
            label_visibility="collapsed"
        )

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        if st.button("Activate License →", use_container_width=True):
            if not key_input:
                st.error("Enter your API key.")
            elif not key_input.startswith("bim_"):
                st.error("Invalid format. Keys start with `bim_`")
            else:
                with st.spinner("Verifying license..."):
                    if check_api_key(key_input):
                        st.session_state["api_key"]       = key_input
                        st.session_state["authenticated"] = True
                        st.rerun()
                    else:
                        st.error("Key rejected or server unreachable.")
                        st.caption("Check your key or contact vivaan@bim-ai.com")

        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown("""
        <p style="text-align:center; color: #4b5f7c; font-size: 12px;">
            Need access? <a href="mailto:vivaan@bim-ai.com"
            style="color:#10b981; text-decoration:none;">vivaan@bim-ai.com</a>
        </p>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════
API_KEY = st.session_state["api_key"]
HEADERS = {"x-api-key": API_KEY}

dxf_files, csv_files = get_output_files()


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:

    # Brand
    st.markdown("""
    <div style="padding: 4px 0 16px;">
        <span style="font-size:20px;">📐</span>
        <span style="font-size:17px; font-weight:700; color:#e2e8f0;
                     margin-left:8px; letter-spacing:-0.3px;">BIM AI Engine</span>
        <div style="font-size:11px; color:#4b5f7c; margin-top:4px;
                    margin-left:30px;">Enterprise Edition</div>
    </div>
    """, unsafe_allow_html=True)

    # Status indicators
    online = backend_online()
    model_active = (DATA_DIR / "models" / "drafting_ai.pth").exists()

    st.markdown(
        f'<div style="display:flex; gap:8px; margin-bottom:16px;">'
        f'<span style="background:{"#062917" if online else "#1a0a0a"}; '
        f'color:{"#10b981" if online else "#ef4444"}; '
        f'padding:4px 10px; border-radius:99px; font-size:11px; font-weight:600;">'
        f'{"● Online" if online else "● Offline"}</span>'
        f'<span style="background:{"#062917" if model_active else "#111827"}; '
        f'color:{"#10b981" if model_active else "#6b7fa3"}; '
        f'padding:4px 10px; border-radius:99px; font-size:11px; font-weight:600;">'
        f'{"🧠 AI Active" if model_active else "📐 Geometric"}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Conversion History ─────────────────────────────────────────────────────
    st.markdown(
        f'<div style="font-size:12px; font-weight:700; color:#6b7fa3; '
        f'text-transform:uppercase; letter-spacing:1px; margin-bottom:12px;">'
        f'Conversion History <span style="color:#10b981;">({len(dxf_files)})</span></div>',
        unsafe_allow_html=True
    )

    if dxf_files:
        for dxf_file in dxf_files[:12]:
            mod_time   = datetime.datetime.fromtimestamp(dxf_file.stat().st_mtime)
            time_str   = mod_time.strftime("%d %b %Y · %I:%M %p")
            size_str   = format_filesize(dxf_file)
            clean_name = dxf_file.stem.replace("GFC_", "").replace("_", " ")

            # Match CSV if it exists
            matching_csv = OUTPUT_DIR / dxf_file.name.replace("GFC_", "SCHEDULE_").replace(".dxf", ".csv")

            st.markdown(f"""
            <div class="history-card">
                <div class="history-name">📐 {clean_name[:30]}</div>
                <div class="history-meta">{time_str} · {size_str}</div>
            </div>
            """, unsafe_allow_html=True)

            dl_col1, dl_col2 = st.columns(2)

            with dl_col1:
                with open(dxf_file, "rb") as f:
                    st.download_button(
                        label="DXF",
                        data=f.read(),
                        file_name=dxf_file.name,
                        mime="application/octet-stream",
                        key=f"dxf_{dxf_file.name}",
                        use_container_width=True
                    )

            with dl_col2:
                if matching_csv.exists():
                    with open(matching_csv, "rb") as f:
                        st.download_button(
                            label="CSV",
                            data=f.read(),
                            file_name=matching_csv.name,
                            mime="text/csv",
                            key=f"csv_{matching_csv.name}",
                            use_container_width=True
                        )

    else:
        st.markdown("""
        <div style="text-align:center; padding: 24px 0; color: #4b5f7c; font-size:13px;">
            No conversions yet.<br>
            <span style="color:#2a3a4a;">Upload your first IFC file →</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Storage info
    total_files = len(dxf_files) + len(csv_files)
    st.markdown(f"""
    <div style="font-size:11px; color:#4b5f7c; line-height:1.8;">
        <div>📁 {total_files} files saved locally</div>
        <div style="margin-top:4px; word-break:break-all;">
            {str(OUTPUT_DIR)}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Sign out
    if st.button("Sign Out", use_container_width=True):
        st.session_state["authenticated"] = False
        st.session_state["api_key"]       = ""
        st.session_state["job_done"]      = False
        st.session_state["result"]        = None
        st.rerun()


# ── MAIN AREA ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 8px 0 24px;">
    <h1 style="font-size:26px; font-weight:800; color:#e2e8f0;
               letter-spacing:-0.5px; margin:0;">
        BIM AI Blueprint Engine
    </h1>
    <p style="color:#4b5f7c; font-size:14px; margin-top:6px;">
        Upload a 3D IFC building model. Get a production DXF drawing
        and door schedule in seconds.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr style='margin-bottom:28px;'>", unsafe_allow_html=True)

left_col, right_col = st.columns([1.1, 0.9], gap="large")


# ── LEFT — Upload ───────────────────────────────────────────────────────────────
with left_col:

    st.markdown("""
    <div style="font-size:13px; font-weight:700; color:#6b7fa3;
                text-transform:uppercase; letter-spacing:1px;
                margin-bottom:14px;">Upload Building Model</div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drag your .IFC file here or click to browse",
        type=["ifc"],
        label_visibility="collapsed"
    )

    if uploaded:
        size_mb  = len(uploaded.getvalue()) / (1024 * 1024)
        is_large = size_mb > 100

        st.markdown(f"""
        <div style="background:#0d1117; border:1px solid #1a2035;
                    border-radius:10px; padding:14px 16px; margin:12px 0;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div style="font-size:14px; font-weight:600; color:#e2e8f0;">
                        📄 {uploaded.name}
                    </div>
                    <div style="font-size:12px; color:#4b5f7c; margin-top:3px;">
                        {size_mb:.1f} MB · IFC 3D Model
                        {"· Hospital-scale file detected" if is_large else ""}
                    </div>
                </div>
                <div style="color:#10b981; font-size:20px;">✓</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if is_large:
            st.info(f"Large file ({size_mb:.0f} MB) — processing may take 5–10 minutes. Don't close this window.")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        if st.button("🚀  Process Building Model", use_container_width=True):
            st.session_state["job_done"] = False
            st.session_state["result"]   = None

            log_box      = st.empty()
            progress_bar = st.progress(0)
            log_lines    = []

            def log(msg: str, pct: int):
                log_lines.append(msg)
                log_box.markdown(
                    "".join([
                        f'<div class="status-line">▸ {l}</div>'
                        for l in log_lines
                    ]),
                    unsafe_allow_html=True
                )
                progress_bar.progress(pct)

            log("Connecting to processing engine...", 5)

            try:
                log(f"Uploading {uploaded.name} ({size_mb:.1f} MB)...", 12)

                response = requests.post(
                    f"{BACKEND}/process-bim-model",
                    files={"file": (
                        uploaded.name,
                        uploaded.getvalue(),
                        "application/octet-stream"
                    )},
                    headers=HEADERS,
                    timeout=900  # 15 min for hospital-scale
                )

                log("IFC received — parsing building structure...", 28)
                time.sleep(0.25)
                log("Extracting wall geometry vectors...", 44)
                time.sleep(0.25)
                log("Running AI annotation placement...", 60)
                time.sleep(0.25)
                log("Spatial collision avoidance pass...", 76)
                time.sleep(0.25)
                log("Compiling production DXF...", 90)
                time.sleep(0.25)

                if response.status_code == 200:
                    data = response.json()

                    if data.get("status") == "Success":
                        log("Complete. Files ready for download.", 100)
                        st.session_state["result"]   = data
                        st.session_state["job_done"] = True
                        st.toast("Conversion complete!", icon="✅")
                        st.rerun()

                    elif response.status_code == 401:
                        st.error("API key rejected. Contact vivaan@bim-ai.com")
                    else:
                        st.error(f"Pipeline failed: {data.get('error_log', 'Unknown error')}")

                elif response.status_code == 401:
                    st.error("Unauthorized. Your API key may have expired.")
                else:
                    st.error(f"Server returned {response.status_code}. Try again.")

            except requests.exceptions.Timeout:
                st.error("Request timed out. Very large files can take up to 15 minutes. Try again.")
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the processing engine.")
                st.code("cd src\npython3 app_orchestrator.py", language="bash")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

    else:
        # Empty state
        st.markdown("""
        <div style="background:#0d1117; border:1px solid #1a2035;
                    border-radius:12px; padding:32px 24px; text-align:center;
                    margin-top:8px;">
            <div style="font-size:36px; margin-bottom:12px;">🏗️</div>
            <div style="font-size:14px; color:#6b7fa3; line-height:1.7;">
                Supports IFC files from<br>
                <span style="color:#c9d1e0;">Revit · ArchiCAD · Tekla · OpenBIM</span>
            </div>
            <div style="margin-top:16px; font-size:12px; color:#2a3a4a;">
                File size up to 5 GB supported
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── RIGHT — Output ──────────────────────────────────────────────────────────────
with right_col:

    st.markdown("""
    <div style="font-size:13px; font-weight:700; color:#6b7fa3;
                text-transform:uppercase; letter-spacing:1px;
                margin-bottom:14px;">Output Files</div>
    """, unsafe_allow_html=True)

    if st.session_state.get("job_done") and st.session_state.get("result"):
        data = st.session_state["result"]

        # Success banner
        st.markdown("""
        <div style="background:#062917; border:1px solid #10b981;
                    border-radius:10px; padding:14px 18px; margin-bottom:20px;
                    display:flex; align-items:center; gap:12px;">
            <span style="font-size:22px;">✅</span>
            <div>
                <div style="font-size:14px; font-weight:700; color:#10b981;">
                    Conversion Successful
                </div>
                <div style="font-size:12px; color:#4b5f7c; margin-top:2px;">
                    Files saved to your local history
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Walls", data.get("wall_count", "—"))
        m2.metric("Drawings", "1 DXF")
        m3.metric("Schedules", "1 CSV")

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        # Download files
        dxf_name = Path(data.get("dxf_blueprint", "")).name
        csv_name = Path(data.get("csv_schedule",  "")).name

        # DXF
        st.markdown("""
        <div style="font-size:12px; color:#6b7fa3; margin-bottom:8px;
                    font-weight:600; text-transform:uppercase; letter-spacing:0.8px;">
            Blueprint Drawing
        </div>
        """, unsafe_allow_html=True)

        local_dxf = OUTPUT_DIR / dxf_name
        if local_dxf.exists():
            with open(local_dxf, "rb") as f:
                st.download_button(
                    label=f"📐  Download {dxf_name}",
                    data=f.read(),
                    file_name=dxf_name,
                    mime="application/octet-stream",
                    use_container_width=True,
                    key="main_dxf"
                )
        else:
            try:
                r = requests.get(f"{BACKEND}/output/{dxf_name}",
                                 headers=HEADERS, timeout=60)
                if r.status_code == 200:
                    st.download_button(
                        label=f"📐  Download {dxf_name}",
                        data=r.content,
                        file_name=dxf_name,
                        mime="application/octet-stream",
                        use_container_width=True,
                        key="main_dxf_remote"
                    )
            except Exception:
                st.info(f"Saved to: {OUTPUT_DIR / dxf_name}")

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # CSV
        st.markdown("""
        <div style="font-size:12px; color:#6b7fa3; margin-bottom:8px;
                    font-weight:600; text-transform:uppercase; letter-spacing:0.8px;">
            Door Schedule
        </div>
        """, unsafe_allow_html=True)

        local_csv = OUTPUT_DIR / csv_name
        if local_csv.exists():
            with open(local_csv, "rb") as f:
                st.download_button(
                    label=f"📊  Download {csv_name}",
                    data=f.read(),
                    file_name=csv_name,
                    mime="text/csv",
                    use_container_width=True,
                    key="main_csv"
                )
        else:
            try:
                r = requests.get(f"{BACKEND}/output/{csv_name}",
                                 headers=HEADERS, timeout=60)
                if r.status_code == 200:
                    st.download_button(
                        label=f"📊  Download {csv_name}",
                        data=r.content,
                        file_name=csv_name,
                        mime="text/csv",
                        use_container_width=True,
                        key="main_csv_remote"
                    )
            except Exception:
                st.info(f"Saved to: {OUTPUT_DIR / csv_name}")

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        # Open file location
        st.markdown(f"""
        <div style="background:#0d1117; border:1px solid #1a2035;
                    border-radius:8px; padding:12px 14px;">
            <div style="font-size:11px; color:#4b5f7c; margin-bottom:4px;">
                Files saved to your computer:
            </div>
            <div style="font-size:12px; color:#6b7fa3;
                        font-family: monospace; word-break:break-all;">
                {OUTPUT_DIR}
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Empty right state
        st.markdown("""
        <div style="background:#0d1117; border:1px solid #1a2035;
                    border-radius:12px; padding:40px 24px; text-align:center;">
            <div style="font-size:40px; margin-bottom:16px;">📂</div>
            <div style="font-size:14px; color:#6b7fa3; margin-bottom:20px;">
                Your converted files appear here
            </div>
            <div style="text-align:left; display:inline-block;">
                <div style="font-size:13px; color:#4b5f7c; line-height:2;">
                    ✓ &nbsp; DXF Blueprint — open in AutoCAD or DraftSight<br>
                    ✓ &nbsp; Door Schedule CSV — open in Excel<br>
                    ✓ &nbsp; All files saved to your history permanently<br>
                    ✓ &nbsp; Redownload any file anytime from the sidebar
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        # How AI improves
        st.markdown("""
        <div style="background:#0d1117; border:1px solid #1a2035;
                    border-radius:12px; padding:20px 22px;">
            <div style="font-size:12px; font-weight:700; color:#6b7fa3;
                        text-transform:uppercase; letter-spacing:1px;
                        margin-bottom:12px;">How The AI Gets Smarter</div>
            <div style="font-size:13px; color:#4b5f7c; line-height:2.2;">
                <span style="color:#10b981;">01</span> &nbsp; Add IFC files to
                <code style="background:#111827; padding:2px 6px;
                border-radius:4px; color:#c9d1e0;">dataset/raw_ifc/</code><br>
                <span style="color:#10b981;">02</span> &nbsp; Add DXF files to
                <code style="background:#111827; padding:2px 6px;
                border-radius:4px; color:#c9d1e0;">dataset/dxf_targets/</code><br>
                <span style="color:#10b981;">03</span> &nbsp; Run
                <code style="background:#111827; padding:2px 6px;
                border-radius:4px; color:#c9d1e0;">python3 src/annotation_ai.py</code><br>
                <span style="color:#10b981;">04</span> &nbsp; Every firm gets
                smarter output automatically
            </div>
        </div>
        """, unsafe_allow_html=True)