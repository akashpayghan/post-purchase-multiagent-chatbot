"""
Visual Upload Component for Image Analysis
"""

import streamlit as st
from PIL import Image
import io

def render_visual_upload():
    """Render visual upload interface"""
    
    st.title("📸 Visual Product Verification")
    st.markdown("Upload product images for AI-powered quality check and defect detection")
    
    # Upload section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Supported formats: JPG, PNG, WEBP (Max 5MB)"
        )
        
        if uploaded_file is not None:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Issue type selection
            issue_type = st.selectbox(
                "What kind of issue are you reporting?",
                ["Defect/Damage", "Wrong Item", "Color Mismatch", "Quality Issue", "Return Verification"]
            )
            
            # Analyze button
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing image with AI..."):
                    # Demo analysis
                    analysis_result = generate_demo_analysis(issue_type)
                    
                    st.success("✅ Analysis Complete!")
                    
                    # Display results
                    st.markdown("### Analysis Results")
                    
                    # Issue confirmation
                    if analysis_result['issue_confirmed']:
                        st.error(f"**Issue Detected:** {analysis_result['issue_type']}")
                    else:
                        st.success("**No Issue Detected**")
                    
                    # Details
                    with st.expander("📋 Detailed Analysis", expanded=True):
                        st.write(f"**Confidence:** {analysis_result['confidence']}%")
                        st.write(f"**Severity:** {analysis_result['severity']}")
                        st.write(f"**Description:** {analysis_result['description']}")
                    
                    # Recommended action
                    st.markdown("### Recommended Action")
                    st.info(analysis_result['recommended_action'])
                    
                    # Action buttons
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        if st.button("✅ Accept Recommendation", use_container_width=True):
                            st.success("Processing your request...")
                    
                    with col_b:
                        if st.button("💬 Speak to Agent", use_container_width=True):
                            st.info("Connecting you to a specialist...")
    
    with col2:
        st.markdown("### Tips for Best Results")
        st.markdown("""
        📌 **Photo Guidelines:**
        - Good lighting
        - Clear focus
        - Show the issue clearly
        - Include product tags (if present)
        - Multiple angles helpful
        
        ✅ **What We Check:**
        - Manufacturing defects
        - Shipping damage
        - Color accuracy
        - Item correctness
        - Overall quality
        """)
        
        # Example images
        st.markdown("### Example Submissions")
        st.caption("✅ Good: Clear, well-lit defect photo")
        st.caption("❌ Bad: Blurry, dark, unclear")
    
    st.divider()
    
    # Statistics
    st.markdown("### Visual Verification Stats")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Images Analyzed", "892", "+34 today")
    
    with col2:
        st.metric("Issues Detected", "147", "16.5%")
    
    with col3:
        st.metric("Avg Resolution Time", "3.2 min", "-45s")

def generate_demo_analysis(issue_type: str):
    """Generate demo analysis results"""
    
    analyses = {
        "Defect/Damage": {
            'issue_confirmed': True,
            'issue_type': 'Seam Defect Detected',
            'confidence': 94,
            'severity': 'Moderate',
            'description': 'AI detected loose stitching on the right seam. The product appears to have a manufacturing defect that could worsen with use.',
            'recommended_action': '🔄 **Immediate Replacement:** We\'ll send a replacement item with free express shipping. You can keep the defective item. We\'ll also add a 10% discount code for your next order.'
        },
        "Wrong Item": {
            'issue_confirmed': True,
            'issue_type': 'Incorrect Item Shipped',
            'confidence': 98,
            'severity': 'High',
            'description': 'The item in the image does not match the ordered product. This appears to be a different color/style variant.',
            'recommended_action': '📦 **Correct Item Shipping:** We\'ll ship the correct item immediately with express delivery. Keep the wrong item at no charge. Our sincere apologies!'
        },
        "Color Mismatch": {
            'issue_confirmed': True,
            'issue_type': 'Color Variance',
            'confidence': 87,
            'severity': 'Minor',
            'description': 'The actual color appears slightly darker than the website photos. This could be due to lighting differences.',
            'recommended_action': '🎨 **Exchange or Discount:** We can exchange it for a different color, or offer a 20% discount if you\'d like to keep it.'
        },
        "Quality Issue": {
            'issue_confirmed': False,
            'issue_type': 'No Quality Issues',
            'confidence': 91,
            'severity': 'None',
            'description': 'The product appears to be in good condition with no visible quality issues. Materials and construction look standard.',
            'recommended_action': '✅ **Product Acceptable:** If you still have concerns, please chat with us to discuss specific details.'
        },
        "Return Verification": {
            'issue_confirmed': True,
            'issue_type': 'Returnable Condition',
            'confidence': 96,
            'severity': 'None',
            'description': 'Item appears unworn with tags attached. Product is in acceptable condition for return.',
            'recommended_action': '✅ **Return Approved:** Your return is approved. We\'ll email you a prepaid return label immediately.'
        }
    }
    
    return analyses.get(issue_type, analyses["Defect/Damage"])
