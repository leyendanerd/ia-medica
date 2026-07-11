"""
auth.py
Autenticación básica local para DentalVision AI.

Contexto (documento del proyecto, sección Despliegue):
"Dado que funciona de manera local (On-Premise), el sistema se ampara
bajo el firewall de la propia clínica. Se implementará un sistema de
autenticación de usuarios básico en Streamlit."

Esta es una autenticación simple por credenciales locales, pensada para
un dispositivo Edge (NVIDIA Jetson) dentro de la red de la clínica, NO
para exposición a internet. En producción real se recomienda:
  - Hash de contraseñas (bcrypt/argon2) en vez de texto plano
  - Gestión de usuarios vía archivo de configuración cifrado o LDAP local
  - Bloqueo tras intentos fallidos
"""

import streamlit as st
import hashlib
import os


# ── Usuarios de demostración ──────────────────────────────────────────────
# En producción: cargar desde variable de entorno o archivo cifrado,
# nunca hardcodeados. Aquí se usa SHA-256 solo para el MVP.
DEMO_USERS = {
    "odontologo": {
        "password_hash": hashlib.sha256("clinica2026".encode()).hexdigest(),
        "role": "Odontólogo",
        "display_name": "Dr. Usuario Demo",
    },
    "admin": {
        "password_hash": hashlib.sha256("admin2026".encode()).hexdigest(),
        "role": "Administrador",
        "display_name": "Administrador de Clínica",
    },
}


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def check_login(username: str, password: str) -> dict | None:
    """Valida credenciales contra el store local. Retorna datos del usuario o None."""
    user = DEMO_USERS.get(username.strip().lower())
    if user and user["password_hash"] == _hash(password):
        return user
    return None


def render_login_screen():
    """Renderiza la pantalla de login y detiene la ejecución hasta autenticar."""

    st.markdown("""
    <style>
    .login-wrap {
        max-width: 380px;
        margin: 4rem auto 0;
        padding: 2rem 2.25rem;
        background: var(--bg-card, #111827);
        border: 1px solid var(--border, #1e2d4a);
        border-radius: 16px;
    }
    .login-icon {
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 0.25rem;
    }
    .login-title {
        text-align: center;
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 1.4rem;
        background: linear-gradient(135deg, #00d4e8, #4f8ef7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .login-sub {
        text-align: center;
        font-size: 0.8rem;
        color: #8899bb;
        margin: 0.25rem 0 1.5rem;
    }
    .login-badge {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.4rem;
        font-size: 0.72rem;
        color: #2ed573;
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        st.markdown("""
        <div class="login-wrap">
            <div class="login-icon">🦷</div>
            <p class="login-title">DentalVision AI</p>
            <p class="login-sub">Acceso restringido · Red local de la clínica</p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Usuario", placeholder="odontologo")
            password = st.text_input("Contraseña", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("Iniciar sesión", type="primary", use_container_width=True)

            if submitted:
                user = check_login(username, password)
                if user:
                    st.session_state.auth_user = user
                    st.session_state.auth_username = username.strip().lower()
                    st.rerun()
                else:
                    st.error("Usuario o contraseña incorrectos", icon="⚠️")

        st.markdown("""
        <div class="login-badge">🔒 Procesamiento 100% local · Sin envío de datos a la nube</div>
        """, unsafe_allow_html=True)

        with st.expander("ℹ️ Credenciales de demostración"):
            st.code("Odontólogo → usuario: odontologo | clave: clinica2026\nAdministrador → usuario: admin | clave: admin2026")

    st.stop()


def require_login():
    """Punto de entrada: exige login antes de continuar con la app."""
    if "auth_user" not in st.session_state:
        render_login_screen()
    return st.session_state.auth_user


def render_logout_control():
    """Muestra el usuario actual y un botón de cerrar sesión (para el sidebar)."""
    user = st.session_state.get("auth_user")
    if not user:
        return
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:0.5rem; font-size:0.78rem; color:#8899bb; margin-bottom:0.5rem;">
        <span style="width:26px; height:26px; border-radius:50%; background:#1a2235; display:inline-flex; align-items:center; justify-content:center; font-weight:600; color:#00d4e8;">
            {user['display_name'][0]}
        </span>
        <span>{user['display_name']}<br><span style="font-size:0.68rem; opacity:0.7;">{user['role']}</span></span>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Cerrar sesión", use_container_width=True):
        del st.session_state["auth_user"]
        st.rerun()
