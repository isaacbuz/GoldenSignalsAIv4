import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth
from GoldenSignalsAI.infrastructure.auth.twilio_mfa import TwilioMFA

if not firebase_admin._apps:
    cred = credentials.Certificate("path/to/firebase-credentials.json")
    firebase_admin.initialize_app(cred)

st.title("GoldenSignalsAI Login")
email = st.text_input("Email")
password = st.text_input("Password", type="password")
phone_number = st.text_input("Phone Number for MFA")

if st.button("Login"):
    try:
        user = auth.get_user_by_email(email)
        st.session_state["user"] = user.uid
        mfa = TwilioMFA()
        verification_sid = mfa.send_mfa_code(phone_number)
        st.session_state["verification_sid"] = verification_sid
        st.session_state["phone_number"] = phone_number
        st.success("MFA code sent to your phone!")
    except:
        st.error("Invalid email or password")

if "user" in st.session_state:
    mfa_code = st.text_input("Enter MFA Code")
    if st.button("Verify MFA"):
        mfa = TwilioMFA()
        if mfa.verify_mfa_code(st.session_state["phone_number"], mfa_code):
            st.success("Logged in successfully!")
            st.session_state["logged_in"] = True
        else:
            st.error("Invalid MFA code")

if "logged_in" in st.session_state and st.session_state["logged_in"]:
    st.write("Redirecting to dashboard...")
    st.experimental_rerun()
