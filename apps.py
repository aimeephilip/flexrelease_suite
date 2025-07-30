import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import pandas as pd
from math import atan2, degrees
from datetime import datetime

# --- Shared Movement Map Instructions ---
instructions = {
    "Overhead Reach": """**Overhead Reach (L/R):**  
    1. Stand tall, feet hip-width apart.  
    2. Lift one arm (left or right) straight overhead as far as you can, keeping elbow straight.  
    3. Keep your torso upright.  
    4. Repeat for both arms for best results.""",
    "Toe Touch": """**Toe Touch (L/R):**  
    1. Stand tall.  
    2. Keeping knees straight, bend at the hips to touch your toes (reach left, right, or centre).  
    3. Note which side you feel tighter on.""",
    "Side Lunge": """**Side Lunge (L/R):**  
    1. Stand wide.  
    2. Shift your weight to the left, bending left knee (keep right leg straight).  
    3. Reach hands to left foot. Return and repeat to the right.""",
    "Lateral Reach": """**Lateral Reach (L/R):**  
    1. Stand tall.  
    2. Raise left arm overhead and reach to the right (side bend), feeling the stretch on the left.  
    3. Repeat with right arm.""",
    "Rotation Reach": """**Rotation Reach (L/R):**  
    1. Stand tall, arms out to the side.  
    2. Rotate trunk to left, reaching right hand across the body to the left.  
    3. Repeat other side."""
}

# --- Streamlit Config ---
st.set_page_config(
    page_title="FlexRelease Suite Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Authentication ---
if "client_id" not in st.session_state:
    st.session_state.client_id = ""
# Only show input if we don't yet have a client_id
if not st.session_state.client_id:
    st.session_state.client_id = st.text_input(
        "🔑 Enter Client ID", ""
    ).strip()
    if not st.session_state.client_id:
        st.stop()

# --- Initialize Navigation ---
if "page" not in st.session_state:
    st.session_state.page = None

# --- Start Handlers ---
def start_rom():
    st.session_state.rom_started = True

def start_maps():
    st.session_state.map_started = True

# --- Dashboard View ---
def show_dashboard():
    st.image("assets/logo.png", width=150)
    st.title("FlexRelease Suite")
    st.header(f"Welcome, {st.session_state.client_id}!")
    st.markdown("## Dashboard")

    # Last ROM Wizard
    if st.session_state.get("results"):
        df_rom = pd.DataFrame(st.session_state.results)
        last_rom = df_rom.iloc[-1]
        st.markdown("### Latest ROM Wizard Result")
        st.metric("Movement", last_rom["movement"])
        st.metric("Max ROM (deg)", f"{last_rom['max_rom']}°")
    else:
        st.info("No ROM Wizard data recorded yet.")

    # Movement Map summaries
    has_map = any(st.session_state.map_history[m] for m in instructions)
    if has_map:
        st.markdown("### Latest Movement Map Sessions")
        for move, hist in st.session_state.map_history.items():
            if hist:
                last = hist[-1]
                c1, c2, c3 = st.columns(3)
                c1.metric(f"{move} L-Comp", last["Left"]["Composite"])
                c2.metric(f"{move} R-Comp", last["Right"]["Composite"])
                c3.metric(f"{move} Symmetry", last.get("Symmetry", "-"))
    else:
        st.info("No Movement Map data recorded yet.")

    st.markdown("---")
    st.markdown("## Select an App to Continue")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🧠 ROM Wizard"):
            st.session_state.page = "rom"
            st.rerun()
    with c2:
        if st.button("📊 Movement Map"):
            st.session_state.page = "maps"
            st.rerun()

# Render dashboard if no page set
if st.session_state.page is None:
    # Initialize histories if first load
    if "map_history" not in st.session_state:
        st.session_state.map_history = {m: [] for m in instructions}
    if "results" not in st.session_state:
        st.session_state.results = []
    show_dashboard()
    st.stop()

# --- Sidebar Home Button ---
with st.sidebar:
    if st.button("🏠 Home"):
        st.session_state.page = None
        # reset start flags so start pages show again
        st.session_state.pop('rom_started', None)
        st.session_state.pop('map_started', None)
        st.rerun()

# --- ROM Wizard Module ---
if st.session_state.page == "rom":
    # Show start page if not yet started
    if not st.session_state.get("rom_started", False):
        st.title("🧠 Welcome to ROM Wizard")
        st.markdown("""
**FlexRelease Full Body Assessment**

This interactive wizard will guide you through a series of movements to measure and record your range of motion using your webcam. Follow the on-screen instructions and ensure your full body is visible in the camera frame.
""")
        if st.button("▶️ Start ROM Wizard"):
            start_rom()
            st.rerun()
        st.stop()

    st.sidebar.title("🧠 ROM Wizard Instructions")
    instr_map = {
        "Shoulder Flexion":    "✅ Side-on stance\n✅ Raise arm forward to ear level",
        "Shoulder Abduction":  "✅ Face camera\n✅ Raise arm directly sideways",
        "Elbow Flexion":       "✅ Side-on stance\n✅ Bend elbow bringing hand to shoulder",
        "Hip Flexion":         "✅ Side-on stance\n✅ Lift thigh forward as high as possible",
        "Hip Abduction":       "✅ Face camera\n✅ Lift thigh sideways away from midline",
        "Hip Extension":       "✅ Side-on stance\n✅ Extend thigh backward without leaning torso",
        "Knee Flexion":        "✅ Side-on stance\n✅ Bend knee by bringing heel toward buttock",
        "Trunk Flexion":       "✅ Side-on stance\n✅ Bend forward at waist keeping legs straight",
        "Trunk Extension":     "✅ Side-on stance\n✅ Arch back gently extending trunk",
        "Ankle Dorsiflexion":  "✅ Side-on stance\n✅ Pull toes up toward shin",
        "Ankle Plantarflexion":"✅ Side-on stance\n✅ Point toes down like pressing gas pedal",
    }
    # Initialize sequence
    if "full_sequence" not in st.session_state:
        st.session_state.full_sequence = [
            {"name":"Shoulder Flexion","side":"Left","joints":[11,13,23],"ideal_max":180},
            {"name":"Shoulder Flexion","side":"Right","joints":[12,14,24],"ideal_max":180},
            {"name":"Shoulder Abduction","side":"Left","joints":[11,13,23],"ideal_max":180},
            {"name":"Shoulder Abduction","side":"Right","joints":[12,14,24],"ideal_max":180},
            {"name":"Elbow Flexion","side":"Left","joints":[13,11,15],"ideal_max":150},
            {"name":"Elbow Flexion","side":"Right","joints":[14,12,16],"ideal_max":150},
            {"name":"Hip Flexion","side":"Left","joints":[23,25,11],"ideal_max":120},
            {"name":"Hip Flexion","side":"Right","joints":[24,26,12],"ideal_max":120},
            {"name":"Hip Abduction","side":"Left","joints":[23,25,11],"ideal_max":45},
            {"name":"Hip Abduction","side":"Right","joints":[24,26,12],"ideal_max":45},
            {"name":"Hip Extension","side":"Left","joints":[23,11,25],"ideal_max":30},
            {"name":"Hip Extension","side":"Right","joints":[24,12,26],"ideal_max":30},
            {"name":"Knee Flexion","side":"Left","joints":[25,23,27],"ideal_max":135},
            {"name":"Knee Flexion","side":"Right","joints":[26,24,28],"ideal_max":135},
            {"name":"Trunk Flexion","side":"Centre","joints":[23,11,25],"ideal_max":90},
            {"name":"Trunk Extension","side":"Centre","joints":[23,11,25],"ideal_max":30},
            {"name":"Ankle Dorsiflexion","side":"Left","joints":[27,25,31],"ideal_max":20},
            {"name":"Ankle Dorsiflexion","side":"Right","joints":[28,26,32],"ideal_max":20},
            {"name":"Ankle Plantarflexion","side":"Left","joints":[27,25,31],"ideal_max":50},
            {"name":"Ankle Plantarflexion","side":"Right","joints":[28,26,32],"ideal_max":50},
        ]
        st.session_state.idx = 0
        st.session_state.recording = False
    # Callbacks
    def start_recording(): st.session_state.recording = True
    def next_stretch():
        st.session_state.idx += 1
        st.session_state.recording = False
    def retry_stretch():
        if st.session_state.results: st.session_state.results.pop()
        st.session_state.recording = True
    def skip_stretch():
        st.session_state.idx += 1
        st.session_state.recording = False

    # Main ROM wizard flow
    seq = st.session_state.full_sequence
    idx = st.session_state.idx
    st.title("🧠 FlexRelease Full Body Assessment")
    if idx < len(seq):
        test = seq[idx]
        st.subheader(f"{test['name']} ({test['side']})")
        st.sidebar.markdown(instr_map[test['name']])
        if not st.session_state.recording:
            col1, col2 = st.columns(2)
            col1.button("▶️ Start Recording", on_click=start_recording)
            col2.button("⏭️ Skip", on_click=skip_stretch)
        else:
            FRAME_WINDOW = st.empty()
            cap = cv2.VideoCapture(0)
            pose = mp.solutions.pose.Pose()
            for t in range(5, 0, -1):
                ret, frame = cap.read()
                if ret:
                    cv2.putText(frame, f"Starting in {t}", (50,60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)
                    FRAME_WINDOW.image(frame, channels="BGR"); time.sleep(1)
            st.success("🎥 Recording for 10 seconds…")
            roms = []
            start_t = time.time(); alpha = 0.2
            while time.time() - start_t < 10:
                ret, frame = cap.read()
                if not ret: continue
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(img_rgb)
                if res.pose_world_landmarks:
                    lm = res.pose_world_landmarks.landmark
                    try:
                        a,m,r = test['joints']
                        A=np.array([lm[a].x,lm[a].y,lm[a].z])
                        M=np.array([lm[m].x,lm[m].y,lm[m].z])
                        R=np.array([lm[r].x,lm[r].y,lm[r].z])
                        raw = np.degrees(np.arccos(np.clip((M-A).dot(R-A)/(np.linalg.norm(M-A)*np.linalg.norm(R-A)), -1,1)))
                        rom = raw if test['name'].startswith('Shoulder') or test['name']=='Hip Flexion' else 180-raw
                        if roms: rom=alpha*rom+(1-alpha)*roms[-1]
                        roms.append(rom)
                        col=(0,255,0) if rom>=0.9*test['ideal_max'] else (0,0,255)
                        fill=int(min(rom/test['ideal_max'],1)*300)
                        cv2.rectangle(frame,(10,20),(10+fill,50),col,-1)
                        cv2.putText(frame, f"ROM: {int(rom)} deg", (320,45), cv2.FONT_HERSHEY_SIMPLEX,0.7,col,2)
                    except:
                        pass
                if res.pose_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(frame,res.pose_landmarks,mp.solutions.pose.POSE_CONNECTIONS)
                FRAME_WINDOW.image(frame, channels="BGR")
            cap.release(); pose.close(); st.session_state.recording=False
            if roms:
                mx=int(np.max(roms)); st.metric("Max ROM", f"{mx}°")
                st.session_state.results.append({"movement":f"{test['name']} {test['side']}","max_rom":mx})
                c1,c2,c3 = st.columns(3)
                c1.button("↻ Retry", on_click=retry_stretch)
                c2.button("⏭️ Skip", on_click=skip_stretch)
                c3.button("➡️ Next", on_click=next_stretch)
            else:
                st.warning("⚠️ No valid ROM data.")
                c1,c2 = st.columns(2)
                c1.button("↻ Retry", on_click=retry_stretch)
                c2.button("⏭️ Skip", on_click=skip_stretch)
    else:
        st.success("✅ Assessment Complete!")
        df = pd.DataFrame(st.session_state.results)
        st.dataframe(df)
        csv = df.to_csv(index=False).encode()
        st.download_button("📥 Download ROM Results", csv, file_name=f"{st.session_state.client_id}_rom.csv")
        if st.button("🔄 Restart ROM Wizard"):
            for k in ("full_sequence", "idx", "recording", "results", "rom_started"):
                st.session_state.pop(k, None)
            st.rerun()

# --- Movement Map Module ---
elif st.session_state.page == "maps":
    # Show start page if not yet started
    if not st.session_state.get("map_started", False):
        st.title("📊 Welcome to Movement Map Tracker")
        st.markdown("""
**FlexRelease Movement Map**

This tool analyzes key movement patterns using your webcam and provides mobility, control, and symmetry scores for each exercise. Ensure your full body is visible in the camera frame.
""")
        if st.button("▶️ Start Movement Map"):
            start_maps()
            st.rerun()
        st.stop()

    st.sidebar.title("📊 Movement Map Navigation")
    if "map_results" not in st.session_state:
        st.session_state.map_results = {m: {"Left": [], "Right": []} for m in instructions}
    if "map_dashboard" not in st.session_state:
        st.session_state.map_dashboard = False

    # Sidebar nav
    with st.sidebar:
        if not st.session_state.map_dashboard:
            if st.button("📈 View Map Dashboard"):
                st.session_state.map_dashboard = True
                st.rerun()
        else:
            if st.button("⬅️ Back to Tracker"):
                st.session_state.map_dashboard = False
                st.rerun()

    # Determine view
    if not st.session_state.map_dashboard:
        movement = st.sidebar.selectbox("Select Movement Pattern", list(instructions.keys()))
        st.sidebar.markdown(instructions[movement])
        st.sidebar.info("Tip: Stand far enough from your webcam so full body is visible.")
        run = st.button("▶️ Start Tracking")
        stop = st.button("■ Finish & Show Summary")
        img_win = st.image([])

        if run:
            st.session_state.map_results[movement] = {"Left": [], "Right": []}
            cap = cv2.VideoCapture(0)
            with mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    img = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                    res = pose.process(img)
                    ann = img.copy(); vL = vR = None
                    if res.pose_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(ann, res.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
                        lm = res.pose_landmarks.landmark
                        pts = {
                            n: [lm[getattr(mp.solutions.pose.PoseLandmark, n).value].x,
                                lm[getattr(mp.solutions.pose.PoseLandmark, n).value].y]
                            for n in ["LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_ELBOW","RIGHT_ELBOW",
                                      "LEFT_HIP","RIGHT_HIP","LEFT_KNEE","RIGHT_KNEE",
                                      "LEFT_ANKLE","RIGHT_ANKLE","LEFT_WRIST","RIGHT_WRIST",
                                      "LEFT_FOOT_INDEX","RIGHT_FOOT_INDEX"]
                        }
                        if movement == "Overhead Reach":
                            aL = calculate_angle(pts["LEFT_ELBOW"], pts["LEFT_SHOULDER"], pts["LEFT_HIP"])
                            hL = pts["LEFT_WRIST"][1] - pts["LEFT_SHOULDER"][1]
                            aR = calculate_angle(pts["RIGHT_ELBOW"], pts["RIGHT_SHOULDER"], pts["RIGHT_HIP"])
                            hR = pts["RIGHT_WRIST"][1] - pts["RIGHT_SHOULDER"][1]
                            vL, vR = [aL, hL], [aR, hR]
                        elif movement == "Toe Touch":
                            aL = calculate_angle(pts["LEFT_SHOULDER"], pts["LEFT_HIP"], pts["LEFT_KNEE"])
                            aR = calculate_angle(pts["RIGHT_SHOULDER"], pts["RIGHT_HIP"], pts["RIGHT_KNEE"])
                            dL = abs(pts["LEFT_WRIST"][1] - pts["LEFT_FOOT_INDEX"][1])
                            dR = abs(pts["RIGHT_WRIST"][1] - pts["RIGHT_FOOT_INDEX"][1])
                            vL, vR = [aL, dL], [aR, dR]
                        elif movement == "Side Lunge":
                            aL = calculate_angle(pts["LEFT_HIP"], pts["LEFT_KNEE"], pts["LEFT_ANKLE"])
                            aR = calculate_angle(pts["RIGHT_HIP"], pts["RIGHT_KNEE"], pts["RIGHT_ANKLE"])
                            fd = abs(pts["LEFT_ANKLE"][0] - pts["RIGHT_ANKLE"][0])
                            vL, vR = [aL, fd], [aR, fd]
                        elif movement == "Lateral Reach":
                            aL = calculate_angle(pts["LEFT_WRIST"], pts["LEFT_SHOULDER"], pts["LEFT_HIP"])
                            aR = calculate_angle(pts["RIGHT_WRIST"], pts["RIGHT_SHOULDER"], pts["RIGHT_HIP"])
                            latL = abs(pts["LEFT_WRIST"][0] - pts["LEFT_HIP"][0])
                            latR = abs(pts["RIGHT_WRIST"][0] - pts["RIGHT_HIP"][0])
                            vL, vR = [aL, latL], [aR, latR]
                        elif movement == "Rotation Reach":
                            sA = line_angle(pts["LEFT_SHOULDER"], pts["RIGHT_SHOULDER"])
                            hA = line_angle(pts["LEFT_HIP"], pts["RIGHT_HIP"])
                            rot = abs(sA - hA)
                            cL = abs(pts["LEFT_WRIST"][0] - pts["RIGHT_HIP"][0])
                            cR = abs(pts["RIGHT_WRIST"][0] - pts["LEFT_HIP"][0])
                            vL, vR = [rot, cL], [rot, cR]
                    if vL:
                        st.session_state.map_results[movement]["Left"].append(vL)
                    if vR:
                        st.session_state.map_results[movement]["Right"].append(vR)
                    img_win.image(ann)
                    if stop:
                        break
                cap.release()

        if stop and (st.session_state.map_results[movement]["Left"] or st.session_state.map_results[movement]["Right"]):
            st.markdown(f"## {movement} Performance Summary")
            labels_map = {
                "Overhead Reach": ["Flex (°)", "Hand Y diff"],
                "Toe Touch": ["Flex (°)", "Hand-Foot Y"],
                "Side Lunge": ["Knee (°)", "Foot X diff"],
                "Lateral Reach": ["Sidebend (°)", "Hand X diff"],
                "Rotation Reach": ["Rot (°)", "Hand X diff"]
            }
            cols = st.columns(2)
            record = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")}
            scores = {}
            for i, side in enumerate(["Left", "Right"]):
                with cols[i]:
                    st.markdown(f"### {side} Side")
                    data = np.array(st.session_state.map_results[movement][side])
                    if data.size:
                        mx = np.max(data, axis=0)
                        sd = np.std(data, axis=0)
                        mob = int(mx[0])
                        ctrl = int(np.clip(100 - sd[0], 0, 100))
                        comp = int(0.7 * mob + 0.3 * ctrl)
                        scores[side] = {"mob": mob, "ctrl": ctrl, "comp": comp}
                        st.metric("Mobility", mob)
                        st.metric("Control", ctrl)
                        st.metric("Composite", comp)
                        st.markdown(f"**Max {labels_map[movement][1]}:** {mx[1]:.1f}")
                        record[side] = {
                            "Mobility": mob,
                            "Control": ctrl,
                            "Composite": comp,
                            "Max2": float(mx[1])
                        }
                    else:
                        st.info("No data for this side.")
            if all(st.session_state.map_results[movement][s] for s in ["Left", "Right"]):
                sym = abs(scores["Left"]["mob"] - scores["Right"]["mob"])
                st.info(f"Symmetry: {sym}")
                record["Symmetry"] = sym
            st.session_state.map_history[movement].append(record)

    else:
        # Map Dashboard
        st.markdown("## 🏆 Movement Map Progress")
        for move, hist in st.session_state.map_history.items():
            st.markdown(f"### {move}")
            if hist:
                latest = hist[-1]
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("L-Composite", latest["Left"]["Composite"])
                    st.metric("L-Mobility", latest["Left"]["Mobility"])
                with c2:
                    st.metric("R-Composite", latest["Right"]["Composite"])
                    st.metric("R-Mobility", latest["Right"]["Mobility"])
                if "Symmetry" in latest:
                    st.info(f"Symmetry: {latest['Symmetry']}")
                tbl = [
                    {
                        "Timestamp": e["timestamp"],
                        "L-Comp": e["Left"]["Composite"],
                        "R-Comp": e["Right"]["Composite"],
                        "Sym": e.get("Symmetry", "-")
                    }
                    for e in hist
                ]
                st.dataframe(pd.DataFrame(tbl), use_container_width=True)
            else:
                st.info("No records yet.")
            st.divider()
