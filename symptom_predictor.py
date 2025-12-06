# symptom_predictor_common_first.py
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
    page_title="Symptom → Top Possible Conditions",
    layout="wide",
    page_icon="🩺",
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
    """Robust JSON extractor."""
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
# Expand trusted links including chronic/common conditions
# ---------------------------
DISEASE_LINKS = {
    "influenza": ["https://www.cdc.gov/flu/index.htm"],
    "covid": ["https://www.cdc.gov/coronavirus/2019-ncov/index.html"],
    "pneumonia": ["https://www.cdc.gov/pneumonia/index.html"],
    "asthma": ["https://www.cdc.gov/asthma/default.htm"],
    "migraine": ["https://www.cdc.gov/headache/migraine.htm"],
    "common cold": ["https://www.cdc.gov/antibiotic-use/community/for-patients/common-illnesses/colds.html"],
    "strep throat": ["https://www.cdc.gov/groupastrep/index.html"],
    "urinary tract infection": ["https://www.cdc.gov/antibiotic-use/community/for-patients/common-illnesses/uti.html"],
    "gastroenteritis": ["https://www.cdc.gov/norovirus/about/index.html"],
    "appendicitis": ["https://www.nhs.uk/conditions/appendicitis/"],
    "pcos": ["https://www.ncbi.nlm.nih.gov/books/NBK279105/"],
    "hypothyroidism": ["https://www.cdc.gov/diabetes/library/features/hypothyroidism.html"],
    "anemia": ["https://www.cdc.gov/nutrition/micronutrient-malnutrition/index.html"],
    "gastritis": ["https://www.niddk.nih.gov/health-information/digestive-diseases/gastritis"],
}

def get_learn_more_links(disease_name: str):
    if not disease_name:
        return []
    dn = disease_name.lower()
    for key in DISEASE_LINKS:
        if key in dn or dn in key:
            return DISEASE_LINKS[key]
    q = quote_plus(disease_name)
    return [f"https://www.cdc.gov/search?q={q}", f"https://www.who.int/search?q={q}"]

# ---------------------------
# UI styling (kept simple)
# ---------------------------
st.markdown(
    """
<style>
.stApp { background: linear-gradient(135deg,#f6f8ff 0%, #fbfdff 100%); }
.card { background:#fff; padding:16px; border-radius:12px; box-shadow:0 8px 20px rgba(15,23,42,0.04); border:1px solid #eef2ff; margin-bottom:16px; }
.h1 { font-size:26px; font-weight:800; }
.h2 { font-size:16px; font-weight:700; }
.pred-pill { display:inline-block; padding:6px 12px; border-radius:999px; background:#e6f0ff; font-weight:700; margin-right:8px; }
.small-pill { display:inline-block; padding:4px 8px; border-radius:999px; background:#f1f5f9; font-size:12px; }
.prog-outer { width:240px; height:12px; border-radius:8px; background:#eef2ff; overflow:hidden; display:inline-block; margin-right:8px; }
.prog-inner { height:12px; border-radius:8px; }
.next-steps { background:#fbfcff; border-left:4px solid #60a5fa; padding:10px; border-radius:8px; margin-top:8px; }
.learn-more { display:inline-block; padding:6px 10px; border-radius:8px; background:#0ea5e9; color:white; text-decoration:none; margin-right:8px; font-size:13px; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# Symptom list (kept large)
# ---------------------------
COMMON_SYMPTOMS = [
    "Cough (dry)", "Cough (productive)", "Sore throat", "Nasal congestion", "Runny nose",
    "Sneezing", "Shortness of breath", "Wheezing", "Chest pain", "Chest tightness",
    "Fever", "Chills", "Fatigue", "Dizziness", "Headache", "Nausea", "Vomiting",
    "Diarrhea", "Abdominal pain", "Loss of appetite", "Palpitations", "Back pain",
    "Joint pain", "Rash", "Itching", "Painful urination", "Frequent urination",
    "Irregular periods", "Excess hair growth", "Acne (severe)", "Weight gain (unexplained)",
    "Excessive thirst", "Excessive urination", "Night sweats", "Confusion", "Seizures",
    "Loss of consciousness", "Severe bleeding", "Severe abdominal pain"
]

# ---------------------------
# Prompt (few-shot + emphasis on common conditions)
# ---------------------------
def call_gemini_for_symptoms(symptoms_text: str, top_k: int = 3):
    if not client:
        return None, "**Error:** Gemini client not initialized. Check your API key."

    # Few-shot examples + explicit bias toward common conditions when ambiguous
    prompt = f"""
You are a clinical triage assistant for educational purposes only.

TASK: Read the user's symptoms and return ONLY a strict JSON object with top {top_k} most likely conditions,
ranked by probability (most likely first). Use exactly this schema:

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

IMPORTANT GUIDANCE (READ CAREFULLY):
- When symptoms are non-specific or could be many things, **prefer common/benign causes first** (e.g., common cold, influenza, gastroenteritis, UTI, tension headache, migraine, reflux, gastritis, PCOS for relevant menstrual features, hypothyroidism for fatigue/weight gain). 
- Include chronic/non-acute conditions (PCOS, hypothyroidism, anemia) when relevant symptoms appear (e.g., irregular periods, weight gain, hair growth → PCOS).
- Keep descriptions short (<= 35 words). Do NOT recommend specific medications or doses.
- Probabilities should reflect relative likelihood (0-1). They don't need to sum to 1 exactly but should be plausible.
- If uncertain, still return common diagnoses rather than rare high-mortality diseases.

EXAMPLE 1
Symptoms:
"Runny nose, sneezing, low-grade fever for 2 days, mild sore throat"

OUTPUT:
{{
  "predictions": [
    {{
      "disease":"Common cold",
      "probability":0.55,
      "description":"Viral upper respiratory infection causing runny nose, sneezing, sore throat; usually self-limited.",
      "consult":"Primary care",
      "precautions":["Rest","Hydration","Paracetamol if fever","Avoid close contact"],
      "links":[]
    }},
    {{
      "disease":"Influenza (flu)",
      "probability":0.25,
      "description":"Viral respiratory infection with fever and body aches; can be more severe than cold.",
      "consult":"Primary care",
      "precautions":["Rest","Hydration","Seek care if severe"],
      "links":[]
    }}
  ],
  "note":"This is NOT a diagnosis. Seek care if concerned."
}}

EXAMPLE 2
Symptoms:
"Irregular periods, weight gain, increased facial hair, acne"

OUTPUT:
{{
  "predictions":[
    {{
      "disease":"Polycystic ovary syndrome (PCOS)",
      "probability":0.7,
      "description":"Hormonal disorder with irregular cycles, acne, hirsutism, and weight gain common in PCOS.",
      "consult":"Endocrinology / Gyn",
      "precautions":["Record menstrual history","Check glucose and lipids","See specialist for evaluation"],
      "links":[]
    }},
    {{
      "disease":"Hypothyroidism",
      "probability":0.15,
      "description":"Low thyroid hormone can cause weight gain and irregular menses; confirm with blood tests.",
      "consult":"Endocrinology",
      "precautions":["Check TSH/T4","Discuss symptoms with clinician"],
      "links":[]
    }}
  ],
  "note":"This is NOT a diagnosis. Seek care if concerned."
}}

Now analyze these symptoms and return JSON only.

Symptoms:
\"\"\"
{symptoms_text}
\"\"\"
"""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt],
        )
        raw = getattr(response, "text", "") or ""
        if not raw and response.candidates:
            parts = response.candidates[0].content.parts
            raw = "".join(p.text or "" for p in parts)
        parsed = parse_gemini_json(raw)
        return parsed, raw or "**Error:** Empty response from model."
    except Exception as e:
        return None, f"An error occurred during API call: {e}"

# ---------------------------
# Heuristics: inject common conditions when model is low-confidence
# ---------------------------
COMMON_HEURISTIC_MAP = {
    "common cold": ["runny nose", "sneezing", "sore throat", "nasal", "congestion"],
    "influenza (flu)": ["high fever", "body ache", "body aches", "chills", "high fever"],
    "covid-19": ["loss of smell", "loss of taste", "covid", "sore throat", "fever", "dry cough"],
    "urinary tract infection": ["painful urination", "blood in urine", "frequent urination", "burning urination"],
    "gastroenteritis": ["diarrhea", "vomiting", "stomach pain", "abdominal cramps", "nausea"],
    "migraine": ["migraine", "one-sided headache", "throbbing headache", "aura"],
    "pcos": ["irregular period", "irregular periods", "acne", "hirsutism", "excess hair", "weight gain"],
    "hypothyroidism": ["weight gain", "cold intolerance", "fatigue", "constipation", "dry skin"],
    "anemia": ["fatigue", "pale", "shortness of breath", "dizziness"],
    "gastritis": ["heartburn", "stomach pain", "upper abdominal pain", "indigestion", "bloating"],
}

def heuristic_inject(common_text: str, preds: list, min_top_thresh: float = 0.30):
    """
    If top model confidence is low (< min_top_thresh), add heuristic common candidates
    based on keyword matches. Then normalize probabilities and return new list.
    """
    # determine current top confidence
    def get_prob(item):
        try:
            pv = float(item.get("probability", 0.0))
            if pv > 1.0:
                pv = min(1.0, pv/100.0)
            return pv
        except Exception:
            return 0.0

    top_prob = 0.0
    if preds:
        top_prob = max(get_prob(p) for p in preds)

    # if model already confident enough, just return sorted preds
    if top_prob >= min_top_thresh:
        return sorted(preds, key=get_prob, reverse=True)

    # otherwise build heuristic candidates
    text = common_text.lower()
    added = {}
    for disease, keywords in COMMON_HEURISTIC_MAP.items():
        for kw in keywords:
            if kw in text:
                # small scoring heuristic by keyword match count
                count = text.count(kw)
                # assign base probability weight (favor common conditions slightly)
                base = 0.25 + min(0.35, 0.05 * count)
                # if already in model preds, we won't duplicate; we will increase its prob later
                added[disease] = max(added.get(disease, 0.0), base)
                break

    # If nothing matched heuristically, we still add "common cold" or "gastroenteritis" if symptoms mention respiratory/GI words
    if not added:
        if any(w in text for w in ["cough", "sore throat", "runny", "sneezing", "nasal"]):
            added["common cold"] = 0.30
        elif any(w in text for w in ["diarrhea", "vomit", "nausea", "stomach"]):
            added["gastroenteritis"] = 0.30
        elif any(w in text for w in ["headache", "migraine"]):
            added["migraine"] = 0.25

    # Merge model preds with heuristics: if model predicted same disease, boost it
    merged = {}
    for p in preds:
        name = (p.get("disease") or "Unknown").strip()
        merged[name.lower()] = get_prob(p)

    for name, prob in added.items():
        lname = name.lower()
        # if model already had it, increase to max(existing, heuristic)
        merged[lname] = max(merged.get(lname, 0.0), prob)

    # Build final list with normalized probabilities
    items = []
    for name, pv in merged.items():
        items.append({"disease": name.title(), "probability": float(pv), "description": "", "consult": "", "precautions": [], "links": []})

    # If still empty (rare), fallback to original preds
    if not items and preds:
        items = preds

    # Normalize probs so they sum to 1 (if total>0)
    total = sum(float(i["probability"]) for i in items)
    if total > 0:
        for i in items:
            i["probability"] = round(float(i["probability"]) / total, 3)
    # sort descending
    items_sorted = sorted(items, key=lambda x: x["probability"], reverse=True)
    # keep top-K (3) and ensure description/consult placeholders if missing
    final = []
    for it in items_sorted[:3]:
        final.append({
            "disease": it.get("disease", "Unknown"),
            "probability": it.get("probability", 0.0),
            "description": it.get("description") or "",
            "consult": it.get("consult") or "",
            "precautions": it.get("precautions") or [],
            "links": it.get("links") or []
        })
    return final

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
st.markdown("<div class='card'><div class='h1'>🩺 Symptom → Top Conditions (Common-first)</div><div style='color:#475569'>It aims at easing the possible predcitions for you.</div></div>", unsafe_allow_html=True)

# ---------------------------
# Layout: left inputs, right results
# ---------------------------
left_col, right_col = st.columns([1, 1.4], gap="large")

with left_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='h2'>🗂 Inputs</div>", unsafe_allow_html=True)
    # stable multiselect
    widget_key = "selected_symptoms_widget"
    if widget_key not in st.session_state:
        st.session_state[widget_key] = []
    selected = st.multiselect("Pick symptoms (check all that apply)", options=COMMON_SYMPTOMS, default=st.session_state.get(widget_key, []), key=widget_key)
    st.session_state.selected_symptoms = list(selected)

    new_sym = st.text_input("Add custom symptom (e.g., 'irregular periods')", key="new_symptom_input")
    if st.button("➕ Add symptom", use_container_width=True):
        s = new_sym.strip()
        if s:
            st.session_state.custom_symptoms.append(s)
            st.session_state.new_symptom_input = ""
            st.rerun()

    if st.session_state.custom_symptoms:
        st.markdown("<div style='margin-top:8px;'><strong>Custom:</strong></div>", unsafe_allow_html=True)
        custom_html = " ".join(f"<span style='display:inline-block;padding:6px 8px;margin:4px;border-radius:10px;background:#eef2ff'>{s}</span>" for s in st.session_state.custom_symptoms)
        st.markdown(custom_html, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='h2'>📝 Describe Symptoms (optional)</div>", unsafe_allow_html=True)
    free_text = st.text_area("Describe symptoms in your own words", height=160, key="symptom_free_text_input", placeholder="E.g., Fever, runny nose, sore throat for 2 days...")

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
        # clear user match keys
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

        # Normalize model probabilities defensively and sort; if low confidence, use heuristic_inject
        def norm_prob(p):
            try:
                pv = float(p.get("probability", 0.0))
                if pv > 1.0:
                    pv = min(1.0, pv / 100.0)
                return pv
            except Exception:
                return 0.0

        preds_sorted = sorted(preds, key=norm_prob, reverse=True)

        # if top confidence is low (<30%), run heuristic injection
        top_conf = norm_prob(preds_sorted[0]) if preds_sorted else 0.0
        combined_text = ""
        checklist = list(st.session_state.selected_symptoms) + st.session_state.custom_symptoms
        if checklist:
            combined_text += "Checklist: " + ", ".join(checklist) + ". "
        if free_text:
            combined_text += "Free text: " + free_text

        if top_conf < 0.30:
            # merge model preds into a normalized structure for heuristic
            # But heuristic_inject returns already-normalized top-3 list
            preds_final = heuristic_inject(combined_text.lower(), preds_sorted, min_top_thresh=0.30)
        else:
            # ensure top 3 normalized but keep descriptions from model
            preds_final = []
            total = sum(norm_prob(p) for p in preds_sorted[:3]) or 1.0
            for p in preds_sorted[:3]:
                preds_final.append({
                    "disease": p.get("disease", "Unknown"),
                    "probability": round(norm_prob(p)/total, 3),
                    "description": p.get("description",""),
                    "consult": p.get("consult",""),
                    "precautions": p.get("precautions") or [],
                    "links": p.get("links") or []
                })

        # render preds_final
        if preds_final:
            st.markdown("#### Top predictions (common-first)")
            for idx, p in enumerate(preds_final, start=1):
                disease = p.get("disease", "Unknown")
                prob = p.get("probability", 0.0)
                desc = p.get("description", "")
                consult = p.get("consult", "")
                precautions = p.get("precautions") or []

                # percent
                try:
                    pval = float(prob)
                    if pval > 1.0:
                        pval = min(1.0, pval/100.0)
                except Exception:
                    pval = 0.0
                fill_pct = int(pval * 100)

                if fill_pct >= 70:
                    prog_color = "#16a34a"
                elif fill_pct >= 40:
                    prog_color = "#f59e0b"
                else:
                    prog_color = "#ef4444"

                st.markdown("<div style='padding:12px;border-radius:10px;margin-bottom:10px;border:1px solid #f1f5f9;'>", unsafe_allow_html=True)
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;'>"
                    f"<div><span class='pred-pill'>{idx}. {disease}</span></div>"
                    f"<div style='text-align:right;'><span class='small-pill'>Confidence: {fill_pct}%</span></div>"
                    f"</div>", unsafe_allow_html=True)

                st.markdown(
                    f"<div style='display:flex;align-items:center;margin-bottom:8px;'>"
                    f"<div class='prog-outer'><div class='prog-inner' style='width:{fill_pct}%;background:{prog_color};'></div></div>"
                    f"<div style='font-size:13px;color:#374151;margin-left:6px'>{fill_pct}%</div>"
                    f"</div>", unsafe_allow_html=True)

                if desc:
                    st.markdown(f"**Description:** {desc}")
                if consult:
                    st.markdown(f"**Who to consult:** {consult}")

                # checkbox after consult, cleared on new predict
                user_key = f"user_match_{idx}"
                checked = st.checkbox("This matches my experience", key=user_key)

                if precautions:
                    st.markdown("**Precautions / Prevention:**")
                    for it in precautions:
                        st.markdown(f"- {it}")

                # learn more
                links = p.get("links") or get_learn_more_links(disease)
                if links:
                    st.markdown("**Learn more:**")
                    link_html = ""
                    for l in links[:3]:
                        link_html += f"<a class='learn-more' href='{l}' target='_blank' rel='noopener noreferrer'>Open</a> "
                    st.markdown(link_html, unsafe_allow_html=True)

                if checked:
                    st.markdown(
                        """
                        <div class='next-steps'>
                        <strong>What should you do next?</strong>
                        <ul style='margin-top:6px;'>
                          <li>If you have severe or worsening symptoms (chest pain, severe breathlessness, fainting, severe bleeding) — seek emergency care immediately.</li>
                          <li>Otherwise, schedule with the specialist listed above and share this symptom summary.</li>
                        </ul>
                        </div>
                        """,
                        unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

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
    checklist = list(st.session_state.selected_symptoms) + st.session_state.custom_symptoms
    combined_text = ""
    if checklist:
        combined_text += "Checklist: " + ", ".join(checklist) + ". "
    free = st.session_state.symptom_free_text_input.strip()
    if free:
        combined_text += "Free text: " + free

    if not combined_text.strip():
        st.warning("Please select or describe at least one symptom.")
    else:
        with st.spinner("Analyzing symptoms and generating predictions..."):
            data_obj, raw_text = call_gemini_for_symptoms(combined_text, top_k=6)
            # store raw
            st.session_state.predictions_raw = raw_text
            # Defensive: if parsed object missing predictions or probabilities, keep raw for debug
            if data_obj and isinstance(data_obj, dict) and data_obj.get("predictions"):
                st.session_state.predictions_data = data_obj
            else:
                # As fallback, set predictions_data to an object with empty predictions so UI can show raw
                st.session_state.predictions_data = {"predictions": []}
            # clear user match keys
            for k in list(st.session_state.keys()):
                if isinstance(k, str) and k.startswith("user_match_"):
                    try:
                        del st.session_state[k]
                    except Exception:
                        pass
            st.rerun()
