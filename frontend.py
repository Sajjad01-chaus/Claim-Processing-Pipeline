import streamlit as st
import requests

st.set_page_config(page_title="Claim Processor AI", layout="wide")
st.title("🏥 Claim Processing Pipeline")

claim_id = st.text_input("Enter Claim ID", value="CLAIM-101")
uploaded_file = st.file_uploader("Upload Claim PDF", type="pdf")

if st.button("Process Claim") and uploaded_file:
    with st.spinner("Processing through LangGraph agents..."):
        
        response = requests.post(
            "https://claim-processing-pipeline-wqvz.onrender.com/api/process",
            data={"claim_id": claim_id},
            files={
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "application/pdf"
                )
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            
            st.success("Processing Complete!")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Patient", result["claim_summary"]["patient_name"])
            col2.metric("Total Billed", f"${result['claim_summary']['total_billed_amount']}")
            col3.metric("Status", result["status"])
            
            st.json(result["extracted_data"])
        else:
            st.error("Pipeline failed. Check logs.")