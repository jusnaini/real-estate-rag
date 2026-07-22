from pathlib import Path
import streamlit as st

BASE = Path(__file__).resolve().parent.parent

chat = st.Page(str(BASE / "app" / "app.py"), title="Chat Assistant", icon="💬")
dashboard = st.Page(str(BASE / "monitoring" / "dashboard.py"), title="Monitoring Dashboard", icon="📊")

pg = st.navigation([chat, dashboard], position="sidebar")
pg.run()
