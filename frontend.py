import streamlit as st
import requests

st.set_page_config(page_title="Claim Processor AI", layout="wide")
st.title("🏥 Claim Processing Pipeline")

claim_id = st.text_input("Enter Claim ID", value="CLAIM-101")
uploaded_file = st.file_uploader("Upload Claim PDF", type="pdf")

if st.button("Process Claim") and uploaded_file:
    with st.spinner("Processing through LangGraph agents..."):
        # Call your live Render API URL
        files = {"file": uploaded_file.getvalue()}
        data = {"claim_id": claim_id}
        
        # Replace with your actual Render URL after deployment
        response = requests.post("http://localhost:8000/api/process", data=data, files={"file": uploaded_file})
        
        if response.status_code == 200:
            result = response.json()
            
            # Show summary metrics
            st.success("Processing Complete!")
            col1, col2, col3 = st.columns(3)
            col1.metric("Patient", result["claim_summary"]["patient_name"])
            col2.metric("Total Billed", f"${result['claim_summary']['total_billed_amount']}")
            col3.metric("Status", result["status"])
            
            # Show full extraction
            st.json(result["extracted_data"])
        else:
            st.error("Pipeline failed. Check logs.")