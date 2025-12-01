import streamlit as st
from google import genai
from google.genai import types
import os
import json
import re

# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="AI Prescription Reader",
    layout="wide",
    page_icon="🩺"
)

# =========================================================
# Gemini API configuration
# =========================================================
# ⚠️ For safety, in real use put this in an env var or st.secrets
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

try:
    if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
        client = genai.Client()
    else:
        st.error("Please set your Gemini API Key (env var GEMINI_API_KEY or inside the script).")
        client = None
except Exception as e:
    st.error(f"Error initializing Gemini client: {e}")
    client = None

# =========================================================
# Helper: robust JSON parsing
# =========================================================
def parse_gemini_json(raw: str):
    """
    Try to recover valid JSON from Gemini's output.
    Handles code fences and simple trailing commas.
    Returns dict or None.
    """
    if not raw:
        return None

    raw = raw.strip()
    # Remove ```json fences if model adds them
    raw = raw.replace("```json", "").replace("```", "").strip()

    # Try to isolate the main JSON object
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        json_str = raw[start:end]
    except ValueError:
        return None

    # Remove trailing commas before } or ]
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

    try:
        return json.loads(json_str)
    except Exception:
        return None

# =========================================================
# Custom CSS
# =========================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef2ff 0%, #f9fafb 100%);
}

/* Generic card */
.app-card {
    background: #ffffff;
    padding: 20px 22px;
    border-radius: 18px;
    box-shadow: 0 12px 28px rgba(15,23,42,0.08);
    border: 1px solid #e5e7eb;
    margin-bottom: 18px;
}

.section-title {
    font-size: 22px;
    font-weight: 650;
    margin-bottom: 6px;
}
.sub-label {
    font-size: 12px;
    color: #6b7280;
}

/* Pill for medicine name */
.med-pill {
    display:inline-block;
    padding:4px 10px;
    border-radius:999px;
    background:#e0e7ff;
    color:#111827;
    font-size:12px;
    font-weight:600;
    margin:2px 6px 2px 0;
}

/* Interaction risk cards */
.risk-card {
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 12px;
    border-left: 5px solid;
    background: #ffffff;
    box-shadow: 0 4px 10px rgba(15,23,42,0.06);
}

.risk-high {
    border-color: #b91c1c;
    background: #fef2f2;
}
.risk-moderate {
    border-color: #92400e;
    background: #fffbeb;
}
.risk-low {
    border-color: #166534;
    background: #ecfdf5;
}

/* Risk label */
.risk-label {
    display:inline-block;
    padding:2px 8px;
    border-radius:999px;
    font-size:11px;
    font-weight:700;
    margin-bottom:4px;
}
.risk-label-high {
    background:#fee2e2;
    color:#b91c1c;
}
.risk-label-moderate {
    background:#fef3c7;
    color:#92400e;
}
.risk-label-low {
    background:#dcfce7;
    color:#166534;
}

.side-pill {
    display:inline-block;
    padding:3px 8px;
    margin:2px 4px 2px 0;
    border-radius:999px;
    background:#e5e7eb;
    font-size:11px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# Gemini logic – returns (structured_dict_or_None, raw_text)
# =========================================================
def process_prescription_with_gemini(data, is_file: bool):
    """
    Call Gemini and return (structured_data_dict or None, raw_text_response).
    structured_data has keys:
        medicines: [ {name, dosage, frequency, purpose, side_effects[]} ]
        interactions: [ {drug1, drug2, risk_level, effect, mechanism, recommendation} ]
        note: str
    """
    if not client:
        return None, "**Error:** Gemini client not initialized. Check your API key."

    prompt_instruction = """
You are a highly specialized medical prescription analysis system.

Your task is to read a prescription (text or scanned image/PDF) and return ONLY a valid JSON object,
with no additional commentary, Markdown, or explanation. The JSON MUST match the structure below exactly:

{
  "medicines": [
    {
      "name": "string",
      "dosage": "string",
      "frequency": "string",
      "purpose": "string",
      "side_effects": ["string", "string"]
    }
  ],
  "interactions": [
    {
      "drug1": "string",
      "drug2": "string",
      "risk_level": "High",
      "effect": "string",
      "mechanism": "string",
      "recommendation": "string"
    }
  ],
  "note": "This is not a substitute for professional medical advice."
}

----------------- EXTRACTION GUIDELINES -----------------

1. Extract every medicine distinctly, even if dosage/frequency is unclear.
2. For EACH medicine, identify:
   - Name (brand or generic)
   - Dosage & Strength (e.g., "625mg", "1 tablet")
   - Frequency/directions (e.g., "1-0-1 × 5 days", "after meals")
   - Purpose (infection, pain relief, acidity control, etc.)
   - 3–6 likely side effects based on medical knowledge.

3. Drug Interaction Analysis:
   - Compare every possible combination of medicines.
   - Only include interactions that have clinical significance.
   - The field "risk_level" MUST be exactly one of: "High", "Moderate", or "Low".
   - For each interaction include:
       - EFFECT on patient
       - MECHANISM (how the interaction occurs)
       - RECOMMENDATION (avoid / spacing / monitoring / safe alternative).

----------------- CRITICAL FORMAT RULES -----------------

✔ Output must be ONLY JSON — no Markdown, code fences, or backticks.
✔ Keys and string values MUST be double-quoted.
✔ No trailing commas.
✔ Response must be directly parseable via json.loads().

If data is missing or unreadable, use an empty string "" or [] but NEVER change keys or structure.
"""

    # Build contents (image first, then instruction – matches docs style)
    if is_file:
        file_bytes = data.getvalue()
        mime_type = data.type
        file_part = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
        contents = [file_part, prompt_instruction]
    else:
        text = data.strip()
        contents = [prompt_instruction, f"Prescription text:\n{text}"]

    try:
        # IMPORTANT: call without config to avoid version issues that cause 'NoneType' errors
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
        )

        raw = getattr(response, "text", "") or ""
        if not raw and response.candidates:
            parts = response.candidates[0].content.parts
            raw = "".join(p.text or "" for p in parts)

        data_obj = parse_gemini_json(raw)
        return data_obj, raw or "**Error:** Empty response from model."

    except Exception as e:
        # If Gemini itself errors, surface that as raw text
        return None, f"An error occurred during API call: {e}"

# =========================================================
# Session state
# =========================================================
if "analysis_data" not in st.session_state:
    st.session_state.analysis_data = None   # parsed JSON dict
if "analysis_raw" not in st.session_state:
    st.session_state.analysis_raw = None    # fallback text

# =========================================================
# Header
# =========================================================
st.markdown(
    """
    <div style="margin-bottom:18px;">
        <h1 style="margin-bottom:4px;">🩺 AI Prescription Reader</h1>
        <p style="color:#4b5563; font-size:15px;">
            Upload a prescription (image/PDF) or paste the text. The app will extract medicines and check for
            <b>potential adverse drug interactions</b> using Gemini, with highlighted risk levels.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Layout – inputs and results
# =========================================================
left_col, right_col = st.columns([1.0, 1.6])

# ---------------- LEFT: INPUT ----------------
with left_col:
    # File card
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📂 Upload Prescription (Image / PDF)</div>", unsafe_allow_html=True)
    st.markdown("<p class='sub-label'>Supported formats: JPG, JPEG, PNG, PDF</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        label=" ",
        type=["jpg", "jpeg", "png", "pdf"],
        label_visibility="collapsed",
    )
    analyze_file = st.button("✨ Analyze File", use_container_width=True, disabled=uploaded_file is None)

    if uploaded_file is not None:
        st.success(f"Selected: **{uploaded_file.name}**")

    st.markdown("</div>", unsafe_allow_html=True)

    # Text card
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📝 Paste Prescription Text</div>", unsafe_allow_html=True)

    prescription_text = st.text_area(
        label="Prescription text",
        placeholder="Example:\nWarfarin 5 mg – 1 tablet once daily\nIbuprofen 400 mg – PRN for pain\n...",
        height=220,
        label_visibility="collapsed",
        key="prescription_text_input",
    )

    c1, c2 = st.columns(2)
    with c1:
        analyze_text = st.button("🧠 Analyze Text", use_container_width=True)
    with c2:
        clear_text = st.button("🗑️ Clear", use_container_width=True)

    if clear_text:
        st.session_state.prescription_text_input = ""
        st.session_state.analysis_data = None
        st.session_state.analysis_raw = None
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- RIGHT: RESULTS ----------------
with right_col:
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>💊 Detected Medicines & Interactions</div>", unsafe_allow_html=True)

    data = st.session_state.analysis_data
    raw = st.session_state.analysis_raw

    if data:
        # ----- Medicines -----
        st.markdown("#### 1. Extracted Medicines")
        meds = data.get("medicines", [])
        if meds:
            for m in meds:
                st.markdown(
                    f"<span class='med-pill'>{m.get('name','Unknown')}</span>",
                    unsafe_allow_html=True
                )
            st.write("")  # spacing
            for m in meds:
                st.markdown(f"**• {m.get('name','Unknown')}**")
                if m.get("dosage"):
                    st.markdown(f"- **Dosage:** {m['dosage']}")
                if m.get("frequency"):
                    st.markdown(f"- **Frequency:** {m['frequency']}")
                if m.get("purpose"):
                    st.markdown(f"- **Purpose:** {m['purpose']}")
                side_effects = m.get("side_effects") or []
                if side_effects:
                    se_html = " ".join(
                        f"<span class='side-pill'>{se}</span>" for se in side_effects
                    )
                    st.markdown(f"- **Common side effects:** {se_html}", unsafe_allow_html=True)
                st.write("")
        else:
            st.write("_No medicines could be confidently extracted._")

        # ----- Interactions -----
        st.markdown("----")
        st.markdown("#### 2. Drug Interactions (Highlighted by Risk)")

        interactions = data.get("interactions", [])
        if not interactions:
            st.write("_No significant interactions detected based on standard references._")
        else:
            # group by risk
            risk_order = ["High", "Moderate", "Low"]
            for level in risk_order:
                group = [i for i in interactions if i.get("risk_level","").lower() == level.lower()]
                if not group:
                    continue

                # risk heading
                st.markdown(f"**{level} Risk ({len(group)})**")

                for inter in group:
                    d1 = inter.get("drug1", "Unknown")
                    d2 = inter.get("drug2", "Unknown")
                    effect = inter.get("effect", "")
                    mech = inter.get("mechanism", "")
                    rec = inter.get("recommendation", "")

                    # pick CSS classes
                    if level.lower() == "high":
                        card_cls = "risk-card risk-high"
                        label_cls = "risk-label risk-label-high"
                    elif level.lower() == "moderate":
                        card_cls = "risk-card risk-moderate"
                        label_cls = "risk-label risk-label-moderate"
                    else:
                        card_cls = "risk-card risk-low"
                        label_cls = "risk-label risk-label-low"

                    html = f"""
                    <div class="{card_cls}">
                        <div class="{label_cls}">{level.upper()} RISK</div>
                        <div><b>{d1}</b> + <b>{d2}</b></div>
                        <div style="font-size:13px;margin-top:6px;">
                            <b>Effect:</b> {effect or '—'}<br/>
                            <b>Mechanism:</b> {mech or '—'}<br/>
                            <b>Recommendation:</b> {rec or '—'}
                        </div>
                    </div>
                    """
                    st.markdown(html, unsafe_allow_html=True)

        # ----- note / disclaimer -----
        note = data.get("note") or (
            "This analysis is generated by an AI system and is not a substitute for professional medical advice."
        )
        st.markdown("---")
        st.markdown(f"> **Disclaimer:** {note}")

    else:
        # fallback: if we have raw text but no JSON
        if raw:
            st.markdown(
                "<p class='sub-label'>Could not parse structured data perfectly. "
                "Showing raw model output below:</p>",
                unsafe_allow_html=True,
            )
            st.markdown(raw)
        else:
            st.markdown(
                "<p class='sub-label'>Upload a file or paste the prescription on the left, then click <b>Analyze</b>.</p>",
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# Triggers
# =========================================================
if analyze_file and uploaded_file is not None:
    with st.spinner("Analyzing file and checking for interactions..."):
        data_obj, raw_text = process_prescription_with_gemini(uploaded_file, is_file=True)
        st.session_state.analysis_data = data_obj
        st.session_state.analysis_raw = raw_text
        st.rerun()

if analyze_text:
    if prescription_text.strip():
        with st.spinner("Analyzing text and checking for interactions..."):
            data_obj, raw_text = process_prescription_with_gemini(prescription_text, is_file=False)
            st.session_state.analysis_data = data_obj
            st.session_state.analysis_raw = raw_text
            st.rerun()
    else:
        st.warning("Please paste some prescription text before analyzing.")

