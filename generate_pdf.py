import sys
import subprocess

try:
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
except ImportError:
    print("Installing reportlab...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet

styles = getSampleStyleSheet()
styleN = styles["Normal"]
styleH = styles["Normal"].clone("Header")
styleH.fontName = "Helvetica-Bold"

data_raw = [
    ["1", "Planning Multiple Epidemic Interventions with Reinforcement Learning", "Core MARL", "Strategizing sequential and concurrent public health interventions using RL."],
    ["2", "Applying Reinforcement Learning to Epidemic Management: Strategic Influenza Control in Multiple Scenarios", "Core MARL", "Formulating dynamic RL-based strategies specifically designed for controlling influenza spread across diverse scenarios."],
    ["3", "H2-MARL: Multi-Agent Reinforcement Learning for Pareto Optimality in Hospital Capacity Strain and Human Mobility during Epidemic", "Core MARL", "Achieving a Pareto-optimal balance between reducing hospital strain and preserving human mobility."],
    ["4", "HRL4EC: Hierarchical Reinforcement Learning for Multi-Mode Epidemic Control", "Core MARL", "Using a multi-level RL approach to manage complex, multi-tiered epidemic interventions."],
    ["5", "Deep Reinforcement Learning for Large-Scale Epidemic Control", "Core MARL", "Applying deep RL frameworks to handle continuous, large-scale agent environments for infection control."],
    ["6", "DPEpiNN: A Unified Framework for Differentially Private Epidemic Forecasting", "DP", "Combining deep neural networks with compartmental models under strong differential privacy bounds for forecasting."],
    ["7", "Epidemiological Modeling with Mobile Phone Data under Differential Privacy", "DP", "Safe utilization of real-world human mobility data using DP to preserve individual anonymity."],
    ["8", "Controlling Epidemic Spread on Networks under Edge Differential Privacy", "DP", "Protecting sensitive patient contact-network graphs while optimizing vaccination strategies."],
    ["9", "Differentially Private Local Distributed Reproduction Numbers", "DP", "A framework to securely calculate localized R0 values without leaking node-to-node interaction data."],
    ["10", "EXAM: A Federated Learning System for Predicting Future Oxygen Requirements of COVID-19 Patients", "FL", "Large-scale 20-institute collaborative FL framework validating improved predictive generalizability."],
    ["11", "Federated Deep Learning for Predicting Mortality of COVID-19 Patients", "FL", "Deploying deep learning across distributed hospital datasets to predict clinical outcomes without sharing raw PI."],
    ["12", "FedPandemic: A Cross-Device Federated Learning Framework for Epidemic Prognosis", "FL", "Integrating cross-device learning and DP-related noise algorithms to assess symptom prognosis securely."],
    ["13", "A Multi-Agent Reinforcement Learning Framework for Public Health Decision Analysis", "Core MARL", "Employing deep MARL for cross-jurisdictional decision modeling and balancing localized resource limits."],
    ["14", "Cognitively-plausible reinforcement learning in epidemiological agent-based simulations", "Core MARL", "Extending RL inside agent-based models to balance an individual's economic incentives against community health restrictions."],
    ["15", "A Reinforcement Learning Based Decision Support Tool for Epidemic Control: Validation Study for COVID-19", "Core MARL", "Validating automated RL support tools for recommending policies that minimize socioeconomic costs."]
]

# Create Paragraphs for word wrapping
data = [[Paragraph("S.No", styleH), Paragraph("Title", styleH), Paragraph("Category", styleH), Paragraph("Key Focus / Relevance", styleH)]]
for row in data_raw:
    data.append([Paragraph(x, styleN) for x in row])

out_path = r"c:\Users\mahas\OneDrive\Documents\Desktop\PFMARL2\Related_Works_Papers.pdf"
doc = SimpleDocTemplate(out_path, pagesize=landscape(letter), leftMargin=30, rightMargin=30, topMargin=30, bottomMargin=30)

# Column Widths
t = Table(data, colWidths=[30, 290, 80, 330])
t.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ('GRID', (0,0), (-1,-1), 1, colors.black),
    ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ('TOPPADDING', (0,0), (-1,-1), 8),
    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold')
]))

elements = [Paragraph("<b>Related Works: Research Papers on PF-MARL for Epidemic Control</b>", styles["Heading1"]), t]
doc.build(elements)
print("PDF Generation complete: ", out_path)
