# Credit-risk-model
## Credit Scoring Business Understanding

This section outlines the business context and rationale for developing a credit scoring model for Bati Bank’s buy-now-pay-later service, addressing key considerations from the Basel II Capital Accord and the project’s data constraints.

1. **Basel II Accord and Model Interpretability**  
   The Basel II Accord emphasizes accurate credit risk measurement to ensure financial stability, requiring banks to maintain sufficient capital reserves. Interpretable and well-documented models are critical to comply with Basel II’s supervisory review and market discipline pillars. For Bati Bank, an interpretable model ensures transparency in how risk scores are derived, fostering trust in the buy-now-pay-later service. Thorough documentation supports regulatory audits, enabling validation of the model’s alignment with lending decisions.

2. **Necessity and Risks of a Proxy Variable**  
   Lacking a direct "default" label, a proxy variable derived from behavioral data (e.g., RFM patterns) is essential to estimate credit risk and categorize customers as high or low risk. This enables Bati Bank to make informed loan approval decisions. However, reliance on a proxy introduces risks: if the proxy poorly correlates with actual default behavior, misclassifications could occur. Overestimating risk may exclude creditworthy customers, reducing revenue, while underestimating risk could increase defaults, leading to financial losses and regulatory challenges.

3. **Trade-offs Between Simple and Complex Models**  
   In a regulated financial context, simple models like Logistic Regression with Weight of Evidence offer interpretability, aligning with Basel II’s transparency requirements, but may lack accuracy for complex datasets. Complex models like Gradient Boosting provide higher predictive performance by capturing non-linear patterns but are less interpretable, complicating regulatory compliance. For Bati Bank, a simple model prioritizes regulatory alignment, while a complex model could enhance accuracy if paired with interpretability tools like SHAP values to meet compliance needs.
