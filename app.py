import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path
from PIL import Image
import io, uuid

from src.vision import analyze_image, crop_vehicle, detect_vehicle_boxes

st.set_page_config(page_title="Video Analytics | License Plate & Color", page_icon="🚘", layout="wide")
st.markdown("""
<style>
.stApp{background:#080d20;color:#e9eefb}
section[data-testid="stSidebar"]{background:#101831}
.block-container{padding-top:1.2rem}
.hero{background:linear-gradient(100deg,#102b55,#10172d);padding:18px 24px;border:1px solid #27416b;border-radius:12px}
.hero h1{color:#f6b83f;margin:0;font-size:30px}.hero p{color:#b7c7e4;margin:4px 0 0}
.metric{background:#121d39;border:1px solid #27385e;padding:14px;border-radius:10px}
.metric small{color:#9aafd1}.metric h2{margin:4px 0;color:#fff}
div[data-testid="stImage"] img{border-radius:8px}
div.stButton>button{border-radius:7px}
</style>
""",unsafe_allow_html=True)

if "records" not in st.session_state: st.session_state.records=[]
if "selected" not in st.session_state: st.session_state.selected=None

st.markdown('<div class="hero"><h1>Video Analytics</h1><p>EP.02 · License Plate &amp; Color Recognition</p></div>',unsafe_allow_html=True)
st.write("")
records=st.session_state.records
c1,c2,c3,c4=st.columns(4)
for col,label,value in [(c1,"TOTAL ANALYZED",len(records)),(c2,"PLATES READ",sum(bool(r["plate"]) for r in records)),(c3,"VEHICLE COLORS",len(set(r["color"] for r in records))), (c4,"AI ENGINE","READY" if True else "OFF")]:
    with col: st.markdown(f'<div class="metric"><small>{label}</small><h2>{value}</h2></div>',unsafe_allow_html=True)

left,right=st.columns([3,1])
with right:
    st.subheader("Analysis")
    uploaded=st.file_uploader("Upload vehicle image",type=["jpg","jpeg","png","webp"])
    st.caption("For best results, upload a clear vehicle photo with the number plate visible.")
    use_yolo=st.checkbox("Use optional YOLO vehicle detector",value=False,help="Install requirements-optional.txt and download YOLO weights on first run.")
    if uploaded:
        raw=uploaded.getvalue()
        image=Image.open(io.BytesIO(raw)).convert("RGB")
        st.image(image,caption=uploaded.name,use_container_width=True)
        if st.button("Analyze image",type="primary",use_container_width=True):
            with st.spinner("Analyzing vehicle, color and plate..."):
                result=analyze_image(image, use_yolo=use_yolo)
                record={"id":str(uuid.uuid4())[:8],"time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "filename":uploaded.name,"plate":result["plate"],"color":result["color"],
                        "vehicle":result["vehicle"],"confidence":result["confidence"],
                        "plate_method":result["plate_method"],"image":raw,
                        "plate_crop":result["plate_crop"]}
                st.session_state.records.insert(0,record)
                st.session_state.selected=record["id"]
            st.success("Analysis complete. See Detection Details.")
    st.divider()
    st.subheader("Filters")
    color_options=["All"]+sorted(set(r["color"] for r in records))
    color_filter=st.selectbox("Vehicle color",color_options)
    plate_filter=st.selectbox("Plate status",["All","Recognized","Not recognized"])
    if st.button("Clear analysis history",use_container_width=True):
        st.session_state.records=[];st.session_state.selected=None;st.rerun()

with left:
    st.subheader("Detected Vehicles")
    filtered=[r for r in records if (color_filter=="All" or r["color"]==color_filter)
              and (plate_filter=="All" or (plate_filter=="Recognized" and r["plate"]!="Not recognized") or (plate_filter=="Not recognized" and r["plate"]=="Not recognized"))]
    if not filtered:
        st.info("No analyzed vehicles yet. Upload an image and click Analyze image.")
    else:
        grid=st.columns(4)
        for i,r in enumerate(filtered):
            with grid[i%4]:
                st.image(Image.open(io.BytesIO(r["image"])),use_container_width=True)
                st.markdown(f"**{r['plate']}**")
                st.caption(f"{r['color']} · {r['vehicle']}")
                if st.button("Detection details",key=f"detail_{r['id']}",use_container_width=True):
                    st.session_state.selected=r["id"]

selected=next((r for r in records if r["id"]==st.session_state.selected),None)
if selected:
    st.divider()
    st.subheader("Detection Details")
    a,b,c=st.columns([1,1,2])
    with a: st.image(Image.open(io.BytesIO(selected["image"])),caption="Source image",use_container_width=True)
    with b:
        if selected["plate_crop"] is not None:
            st.image(selected["plate_crop"],caption="Plate candidate",use_container_width=True)
        else: st.info("No plate candidate crop found.")
    with c:
        st.markdown(f"### {selected['plate']}")
        st.write("**Vehicle type:**",selected["vehicle"])
        st.write("**Detected color:**",selected["color"])
        st.write("**Plate confidence:**",f"{selected['confidence']:.1f}%")
        st.write("**Recognition method:**",selected["plate_method"])
        st.write("**Timestamp:**",selected["time"])
        if selected["plate"]=="Not recognized":
            manual=st.text_input("Enter plate manually",key=f"manual_{selected['id']}")
            if st.button("Save plate correction",key=f"save_{selected['id']}"):
                selected["plate"]=manual.strip().upper() or "Not recognized"
                st.success("Plate value updated.")
st.divider()
st.caption("Demo note: automated plate reading depends on image quality and OCR availability. Verify every result before using it for enforcement or access control.")
