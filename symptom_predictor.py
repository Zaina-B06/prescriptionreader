import streamlit as st
from google import genai
from google.genai import types
import os
import json
import re
from urllib.parse import quote_plus

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Symptom → Top Conditions (Clean — No Urgency)",
    layout="wide",
    page_icon="🩺"
)

# ---------------------------
# Gemini / API key
# ---------------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

try:
    if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
        client = genai.Client()
    else:
        st.error("Please set your Gemini API Key (env var GEMINI_API_KEY or inside st.secrets).")
        client = None
except Exception as e:
    st.error(f"Error initializing Gemini client: {e}")
    client = None

# ---------------------------
# Helpers
# ---------------------------
def parse_gemini_json(raw: str):
    """Try to extract a JSON object from model output robustly."""
    if not raw:
        return None
    raw = raw.strip().replace("```json", "").replace("```", "").strip()
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        json_str = raw[start:end]
    except ValueError:
        return None
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
    try:
        return json.loads(json_str)
    except Exception:
        try:
            fixed = json_str.replace("'", '"')
            return json.loads(fixed)
        except Exception:
            return None

# ---------------------------
# Trusted links mapping (extendable)
# ---------------------------
DISEASE_LINKS = {
    "influenza": ["https://www.cdc.gov/flu/index.htm", "https://www.who.int/health-topics/influenza"],
    "covid": ["https://www.cdc.gov/coronavirus/2019-ncov/index.html", "https://www.who.int/health-topics/coronavirus"],
    "pneumonia": ["https://www.cdc.gov/pneumonia/index.html", "https://www.who.int/health-topics/pneumonia"],
    "asthma": ["https://www.cdc.gov/asthma/default.htm"],
    "migraine": ["https://www.cdc.gov/headache/migraine.htm"],
    "common cold": ["https://www.cdc.gov/antibiotic-use/community/for-patients/common-illnesses/colds.html"],
    "strep throat": ["https://www.cdc.gov/groupastrep/index.html"],
    "urinary tract infection": ["https://www.cdc.gov/antibiotic-use/community/for-patients/common-illnesses/uti.html"],
    "appendicitis": ["https://www.nhs.uk/conditions/appendicitis/"],
    "hypertension": ["https://www.cdc.gov/bloodpressure/index.htm"],
    "diabetes": ["https://www.cdc.gov/diabetes/index.html"],
}

def get_learn_more_links(disease_name: str):
    """Return trusted links or fallback search pages."""
    if not disease_name:
        return []
    dn = disease_name.lower().strip()
    for key in DISEASE_LINKS:
        if key in dn or dn in key:
            return DISEASE_LINKS[key]
    q = quote_plus(disease_name)
    return [f"https://www.cdc.gov/search?q={q}", f"https://www.who.int/search?q={q}"]

# ---------------------------
# Styling (clean)
# ---------------------------
st.markdown(
    """
<style>
.stApp { background: linear-gradient(135deg,#f6f8ff 0%, #fbfdff 100%); }
.card { background: #fff; padding:18px; border-radius:14px; box-shadow:0 8px 22px rgba(15,23,42,0.04); border:1px solid #eef2ff; margin-bottom:18px; }
.h1 { font-size:28px; font-weight:800; margin-bottom:6px; }
.h2 { font-size:18px; font-weight:700; margin-bottom:6px; color:#0f172a; }
.pred-pill { display:inline-block; padding:6px 12px; border-radius:999px; background:#e6f0ff; color:#0b2447; font-weight:700; margin-right:8px; }
.small-pill { display:inline-block; padding:4px 8px; border-radius:999px; background:#f1f5f9; color:#0f172a; font-size:12px; }
.prog-outer { width:260px; height:14px; border-radius:8px; background:#eef2ff; overflow:hidden; display:inline-block; margin-right:10px; }
.prog-inner { height:14px; border-radius:8px; }
.next-steps { background:#fbfcff; border-left:4px solid #60a5fa; padding:12px; border-radius:8px; margin-top:8px; }
.learn-more { display:inline-block; padding:8px 10px; border-radius:8px; background:#0ea5e9; color:white; text-decoration:none; margin-right:8px; font-size:13px; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# Symptoms list
# ---------------------------
COMMON_SYMPTOMS = [
    "Cough (dry)", "Cough (productive)", "Sore throat", "Hoarseness", "Nasal congestion", "Runny nose",
    "Sneezing", "Shortness of breath at rest", "Shortness of breath on exertion", "Wheezing", "Chest tightness",
    "Chest pain (sharp)", "Chest pain (pressure)", "Rapid breathing", "Coughing blood",
    "Fever (high)", "Low-grade fever", "Chills", "Night sweats", "Fatigue", "Weakness", "Dizziness",
    "Unexplained weight loss", "Unexplained weight gain", "Loss of appetite",
    "Headache (diffuse)", "Headache (localized)", "Migraine-like pain", "Confusion", "Memory loss",
    "Fainting / syncope", "Seizures", "Numbness (limb)", "Tingling (paresthesia)",
    "Nausea", "Vomiting", "Diarrhea", "Constipation", "Abdominal pain (upper)", "Abdominal pain (lower)",
    "Bloating", "Heartburn / reflux", "Blood in stool", "Black stools",
    "Palpitations", "Irregular heartbeat", "Leg swelling", "Sudden limb pain", "Cold limb",
    "High blood pressure", "Low blood pressure", "Rapid heart rate",
    "Ear pain", "Hearing loss", "Tinnitus (ringing)", "Eye pain", "Blurred vision", "Double vision",
    "Rash (red)", "Rash (blistering)", "Itching", "Skin ulcer", "Swelling (localized)", "Bruising easily",
    "Painful urination", "Frequent urination", "Blood in urine", "Vaginal bleeding", "Pelvic pain",
    "Back pain", "Neck pain", "Joint pain", "Joint swelling", "Muscle ache", "Recent injury / trauma",
    "Insomnia", "Excessive sleepiness", "Anxiety", "Depression", "Excessive thirst", "Excessive urination",
    "Swollen lymph nodes", "Bad breath", "Difficulty swallowing", "Unusual bleeding",
    "Sudden severe chest pain", "Severe shortness of breath", "Loss of consciousness", "Severe bleeding"
]

# ---------------------------
# Gemini prompt — updated to encourage common conditions
# ---------------------------
def call_gemini_for_symptoms(symptoms_text: str, top_k: int = 3):
    if not client:
        return None, "**Error:** Gemini client not initialized. Check your API key."

    # Instruct model to consider common conditions explicitly
    prompt_instruction = f"""
You are a clinical triage assistant for educational purposes only.

INPUT: symptoms text (may be a mix of checklist items and free-text).

TASK: Return ONLY a strict JSON object (no markdown or commentary) with the top {top_k} most likely conditions
ordered from most to least likely. Use the exact schema shown below.

IMPORTANT: When symptoms match common illnesses, include them in candidates (for example: "common cold", "influenza (flu)", "COVID-19", "strep throat", "bronchitis", "pneumonia", "urinary tract infection", "gastroenteritis", "migraine", "appendicitis"). Always rank by likelihood.

{{
  "predictions": [
    {{
      "disease": "string",
      "probability": 0.0,
      "description": "short (1-2 sentence) plain-language description",
      "consult": "which specialist or clinic (max 4 words)",
      "precautions": ["short bullet strings (3-6 items)"],
      "links": ["optional authoritative URLs, if known"]
    }}
  ],
  "note": "This is NOT a diagnosis. Seek medical care when appropriate."
}}

RULES:
- Output must be ONLY JSON. Keys and values double-quoted.
- "probability" between 0 and 1 (two decimal places ok).
- Keep description concise (<= 35 words).
- Provide 3-6 brief precautions per disease.
- Provide 0-3 trusted links per disease where available (CDC/WHO/NHS style). If unknown, use [].
- Do NOT suggest exact medications or dosing.

Now analyze these symptoms and return the JSON only:

Symptoms:
\"\"\"
{symptoms_text}
\"\"\"
"""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt_instruction],
        )
        raw = getattr(response, "text", "") or ""
        if not raw and response.candidates:
            parts = response.candidates[0].content.parts
            raw = "".join(p.text or "" for p in parts)
        data_obj = parse_gemini_json(raw)
        return data_obj, raw or "**Error:** Empty response from model."
    except Exception as e:
        return None, f"An error occurred during API call: {e}"

# ---------------------------
# Session init
# ---------------------------
if "selected_symptoms" not in st.session_state:
    st.session_state.selected_symptoms = []
if "custom_symptoms" not in st.session_state:
    st.session_state.custom_symptoms = []
if "predictions_data" not in st.session_state:
    st.session_state.predictions_data = None
if "predictions_raw" not in st.session_state:
    st.session_state.predictions_raw = None

# ---------------------------
# Header
# ---------------------------
st.markdown("<div class='card'><div class='h1'>🩺 Symptom → Top Conditions</div><div style='color:#475569'>Paste/select symptoms and get top possible conditions (educational only). Top results are sorted by model confidence.</div></div>", unsafe_allow_html=True)

# ---------------------------
# Layout: inputs left, results right
# ---------------------------
left_col, right_col = st.columns([1, 1.4], gap="large")

with left_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='h2'>🗂 Symptom Checklist & Inputs</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#6b7280;margin-bottom:10px;'>Select from the checklist or add a custom symptom. Optionally describe in free-text below.</div>", unsafe_allow_html=True)

    # stable multiselect (avoid focus glitch)
    widget_key = "selected_symptoms_widget"
    _valid_options = set(COMMON_SYMPTOMS)
    if widget_key not in st.session_state:
        st.session_state[widget_key] = [s for s in st.session_state.get("selected_symptoms", []) if s in _valid_options]

    selected = st.multiselect(
        "Pick symptoms (check all that apply)",
        options=COMMON_SYMPTOMS,
        default=st.session_state.get(widget_key, []),
        key=widget_key
    )
    st.session_state.selected_symptoms = list(selected)

    # add custom symptom
    new_sym = st.text_input("Add custom symptom (e.g., 'left arm pain')", key="new_symptom_input")
    add_col, dummy_col = st.columns([1, 0.15])
    with add_col:
        if st.button("➕ Add symptom", use_container_width=True):
            s = new_sym.strip()
            if s:
                st.session_state.custom_symptoms.append(s)
                st.session_state.new_symptom_input = ""
                st.rerun()

    if st.session_state.custom_symptoms:
        st.markdown("<div style='margin-top:8px;'><strong>Custom:</strong></div>", unsafe_allow_html=True)
        custom_html = " ".join(f"<span style='display:inline-block;padding:6px 10px;margin:4px;border-radius:12px;background:#eef2ff;'>{s}</span>" for s in st.session_state.custom_symptoms)
        st.markdown(custom_html, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='h2'>📝 Describe Symptoms (optional)</div>", unsafe_allow_html=True)
    free_text = st.text_area("Describe symptoms in your own words", height=180, key="symptom_free_text_input",
                             placeholder="E.g., Fever, sore throat, runny nose for 2 days...")

    col_run, col_clear = st.columns([1,1])
    with col_run:
        analyze = st.button("🔎 Predict Top 3", use_container_width=True)
    with col_clear:
        clear = st.button("🗑️ Clear All", use_container_width=True)

    if clear:
        st.session_state.selected_symptoms = []
        st.session_state.custom_symptoms = []
        st.session_state.symptom_free_text_input = ""
        st.session_state.predictions_data = None
        st.session_state.predictions_raw = None
        # clear per-disease user match keys
        for k in list(st.session_state.keys()):
            if isinstance(k, str) and k.startswith("user_match_"):
                try:
                    del st.session_state[k]
                except Exception:
                    pass
        if widget_key in st.session_state:
            del st.session_state[widget_key]
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='h2'>🔔 Results</div>", unsafe_allow_html=True)

    data = st.session_state.predictions_data
    raw = st.session_state.predictions_raw

    if data:
        preds = data.get("predictions", []) or []

        # Defensive normalization of probability and sorting by probability desc
        def prob_key(item):
            try:
                pv = float(item.get("probability", 0.0))
                if pv > 1.0:
                    pv = min(1.0, pv / 100.0)
                return pv
            except Exception:
                return 0.0

        preds_sorted = sorted(preds, key=prob_key, reverse=True)

        if preds_sorted:
            st.markdown("#### Top predictions (ranked by confidence)")
            for idx, p in enumerate(preds_sorted, start=1):
                disease = p.get("disease", "Unknown")
                prob = p.get("probability", 0.0)
                desc = p.get("description", "")
                consult = p.get("consult", "")
                precautions = p.get("precautions", []) or []

                # normalize prob to 0..1 then percent
                try:
                    pval = float(prob)
                    if pval > 1.0:
                        pval = min(1.0, pval / 100.0)
                except Exception:
                    pval = 0.0
                fill_pct = int(pval * 100)

                # progress bar color
                if fill_pct >= 70:
                    prog_color = "#16a34a"
                elif fill_pct >= 40:
                    prog_color = "#f59e0b"
                else:
                    prog_color = "#ef4444"

                # disease card
                st.markdown("<div style='padding:12px;border-radius:10px;margin-bottom:12px;border:1px solid #f1f5f9;'>", unsafe_allow_html=True)

                # header
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;'>"
                    f"<div><span class='pred-pill'>{idx}. {disease}</span></div>"
                    f"<div style='text-align:right;'><span class='small-pill'>Confidence: {fill_pct}%</span></div>"
                    f"</div>", unsafe_allow_html=True
                )

                # progress bar
                st.markdown(
                    f"<div style='display:flex;align-items:center;margin-bottom:10px;'>"
                    f"<div class='prog-outer'><div class='prog-inner' style='width:{fill_pct}%;background:{prog_color};'></div></div>"
                    f"<div style='font-size:13px;color:#374151;margin-left:6px'>{fill_pct}%</div>"
                    f"</div>", unsafe_allow_html=True
                )

                if desc:
                    st.markdown(f"**Description:** {desc}")
                if consult:
                    st.markdown(f"**Who to consult:** {consult}")

                # checkbox after consult
                user_key = f"user_match_{idx}"
                checked = st.checkbox("This matches my experience", key=user_key)

                # precautions
                if precautions:
                    st.markdown("**Precautions / Prevention:**")
                    for item in precautions:
                        st.markdown(f"- {item}")

                # learn more
                model_links = p.get("links") or []
                links = model_links or get_learn_more_links(disease)
                if links:
                    st.markdown("**Learn more:**")
                    link_html = ""
                    for l in links[:3]:
                        link_html += f"<a class='learn-more' href='{l}' target='_blank' rel='noopener noreferrer'>Open</a> "
                    st.markdown(link_html, unsafe_allow_html=True)

                # inline next steps if checked
                if checked:
                    st.markdown(
                        """
                        <div class='next-steps'>
                        <strong>What should you do next?</strong>
                        <ul style='margin-top:6px;'>
                          <li>If you have severe or worsening symptoms (chest pain, severe shortness of breath, fainting, heavy bleeding) — seek emergency care immediately.</li>
                          <li>For non-urgent issues, book with the specialist listed above and share this symptom summary.</li>
                          <li>Note onset, severity, and red flags to share with your clinician.</li>
                        </ul>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown("</div>", unsafe_allow_html=True)

            # disclaimer
            note = data.get("note") or "This is NOT a diagnosis. Seek professional care."
            st.markdown("---")
            st.markdown(f"> **Disclaimer:** {note}")

        else:
            st.write("_No structured predictions returned._")

    else:
        if raw:
            st.markdown("<div style='color:#6b7280'>Could not parse structured JSON perfectly. Showing raw model output:</div>", unsafe_allow_html=True)
            st.code(raw)
        else:
            st.markdown("<div style='color:#6b7280'>Enter symptoms and click <strong>Predict Top 3</strong>.</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Trigger analyze (call Gemini)
# ---------------------------
if 'analyze' in locals() and analyze:
    combined_list = list(st.session_state.selected_symptoms) + st.session_state.custom_symptoms
    combined_list = [s.strip() for s in combined_list if s and s.strip()]
    combined_text = ""
    if combined_list:
        combined_text += "Checklist symptoms: " + ", ".join(combined_list) + ".\n"
    free = st.session_state.symptom_free_text_input.strip()
    if free:
        combined_text += "Free-text description: " + free

    if not combined_text:
        st.warning("Please select or describe at least one symptom.")
    else:
        with st.spinner("Analyzing symptoms and generating predictions..."):
            data_obj, raw_text = call_gemini_for_symptoms(combined_text, top_k=3)
            st.session_state.predictions_data = data_obj
            st.session_state.predictions_raw = raw_text

            # clear per-disease user match_* keys so previous checkboxes don't persist
            for k in list(st.session_state.keys()):
                if isinstance(k, str) and k.startswith("user_match_"):
                    try:
                        del st.session_state[k]
                    except Exception:
                        pass

            st.rerun()
