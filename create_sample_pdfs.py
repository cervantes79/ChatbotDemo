import os

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

def create_company_handbook():
    filename = "data/company_handbook.pdf"
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "ACME CORPORATION EMPLOYEE HANDBOOK")
    
    # Content
    c.setFont("Helvetica", 12)
    y_position = height - 100
    
    content = [
        "Welcome to ACME Corporation!",
        "This handbook contains important information about our company policies.",
        "",
        "COMPANY POLICIES:",
        "- Work Hours: Monday to Friday, 9:00 AM to 5:00 PM",
        "- Remote Work: Available 2 days per week with manager approval",
        "- Vacation Policy: 15 days annual leave for new employees",
        "- Sick Leave: 10 days per year",
        "- Dress Code: Business casual in office, relaxed for remote work",
        "",
        "RETURN AND REFUND POLICY:",
        "For company equipment and supplies:",
        "- All items must be returned within 30 days of termination",
        "- Damaged items will be charged to final paycheck",
        "- Original packaging preferred but not required",
        "",
        "CONTACT INFORMATION:",
        "- HR Department: hr@acme.com",
        "- IT Support: it@acme.com",
        "- General Questions: info@acme.com",
        "- Phone: +1 (555) 123-4567",
        "",
        "BENEFITS:",
        "- Health Insurance: Full coverage after 90 days",
        "- Dental and Vision: Available after 90 days",
        "- 401k: Company matches up to 4%",
        "- Gym Membership: 50% reimbursement up to $100/month"
    ]
    
    for line in content:
        if line.startswith("-"):
            c.drawString(70, y_position, line)
        else:
            c.drawString(50, y_position, line)
        y_position -= 20
        
        if y_position < 50:
            c.showPage()
            y_position = height - 50
    
    c.save()
    print(f"Created {filename}")

def create_product_catalog():
    filename = "data/product_catalog.pdf"
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "ACME PRODUCTS CATALOG 2024")
    
    c.setFont("Helvetica", 12)
    y_position = height - 100
    
    content = [
        "Our Premium Product Line",
        "",
        "ACME Widget Pro - $299",
        "Features:",
        "- Advanced automation capabilities",
        "- 24/7 customer support included",
        "- 2-year warranty",
        "- Compatible with all major platforms",
        "- Cloud integration ready",
        "",
        "ACME Widget Standard - $149",
        "Features:",
        "- Basic automation features",
        "- Email support",
        "- 1-year warranty",
        "- Platform compatibility for Windows and Mac",
        "",
        "ACME Widget Lite - $79",
        "Features:",
        "- Essential features only",
        "- Community support",
        "- 6-month warranty",
        "- Windows compatible only",
        "",
        "All products include:",
        "- Free shipping within US",
        "- 30-day money back guarantee",
        "- Setup assistance via phone or email",
        "- Access to online tutorials and documentation",
        "",
        "TECHNICAL SPECIFICATIONS:",
        "- Minimum RAM: 4GB (8GB recommended)",
        "- Storage: 500MB free space required",
        "- Internet connection required for activation",
        "- Operating System: Windows 10+ or macOS 10.15+"
    ]
    
    for line in content:
        if line.startswith("-"):
            c.drawString(70, y_position, line)
        else:
            c.drawString(50, y_position, line)
        y_position -= 20
        
        if y_position < 50:
            c.showPage()
            y_position = height - 50
    
    c.save()
    print(f"Created {filename}")

def create_support_faq():
    filename = "data/support_faq.pdf"
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "FREQUENTLY ASKED QUESTIONS")
    
    c.setFont("Helvetica", 12)
    y_position = height - 100
    
    content = [
        "Q: How do I install ACME Widget?",
        "A: Download the installer from our website, run it as administrator,",
        "   and follow the on-screen instructions. Installation takes 5-10 minutes.",
        "",
        "Q: What is your return policy?",
        "A: We offer a 30-day money-back guarantee. Items must be returned",
        "   in original condition. Refunds are processed within 5-7 business days.",
        "",
        "Q: Do you offer technical support?",
        "A: Yes! Pro customers get 24/7 phone and email support.",
        "   Standard customers get email support during business hours.",
        "   Lite customers have access to community forums.",
        "",
        "Q: Can I upgrade my license later?",
        "A: Absolutely! Contact our sales team and pay only the difference",
        "   between your current license and the new one.",
        "",
        "Q: What payment methods do you accept?",
        "A: We accept all major credit cards, PayPal, and bank transfers",
        "   for enterprise customers.",
        "",
        "Q: Is there a mobile app?",
        "A: Yes, our mobile app is available for iOS and Android.",
        "   It includes basic monitoring and notification features.",
        "",
        "Q: How often do you release updates?",
        "A: We release minor updates monthly and major updates quarterly.",
        "   All updates are free for active license holders.",
        "",
        "Q: Do you offer training?",
        "A: We provide online tutorials, documentation, and webinars.",
        "   Enterprise customers can request custom training sessions."
    ]
    
    for line in content:
        c.drawString(50, y_position, line)
        y_position -= 15
        
        if y_position < 50:
            c.showPage()
            y_position = height - 50
    
    c.save()
    print(f"Created {filename}")

def create_simple_text_files():
    """Create simple text files as sample data"""
    
    # Company Handbook
    with open("data/company_handbook.txt", "w") as f:
        f.write("""ACME CORPORATION EMPLOYEE HANDBOOK

Welcome to ACME Corporation!
This handbook contains important information about our company policies.

COMPANY POLICIES:
- Work Hours: Monday to Friday, 9:00 AM to 5:00 PM
- Remote Work: Available 2 days per week with manager approval
- Vacation Policy: 15 days annual leave for new employees
- Sick Leave: 10 days per year
- Dress Code: Business casual in office, relaxed for remote work

RETURN AND REFUND POLICY:
For company equipment and supplies:
- All items must be returned within 30 days of termination
- Damaged items will be charged to final paycheck
- Original packaging preferred but not required

CONTACT INFORMATION:
- HR Department: hr@acme.com
- IT Support: it@acme.com
- General Questions: info@acme.com
- Phone: +1 (555) 123-4567

BENEFITS:
- Health Insurance: Full coverage after 90 days
- Dental and Vision: Available after 90 days
- 401k: Company matches up to 4%
- Gym Membership: 50% reimbursement up to $100/month""")
    
    # Product Catalog
    with open("data/product_catalog.txt", "w") as f:
        f.write("""ACME PRODUCTS CATALOG 2024

Our Premium Product Line

ACME Widget Pro - $299
Features:
- Advanced automation capabilities
- 24/7 customer support included
- 2-year warranty
- Compatible with all major platforms
- Cloud integration ready

ACME Widget Standard - $149
Features:
- Basic automation features
- Email support
- 1-year warranty
- Platform compatibility for Windows and Mac

ACME Widget Lite - $79
Features:
- Essential features only
- Community support
- 6-month warranty
- Windows compatible only

All products include:
- Free shipping within US
- 30-day money back guarantee
- Setup assistance via phone or email
- Access to online tutorials and documentation

TECHNICAL SPECIFICATIONS:
- Minimum RAM: 4GB (8GB recommended)
- Storage: 500MB free space required
- Internet connection required for activation
- Operating System: Windows 10+ or macOS 10.15+""")
    
    # Support FAQ
    with open("data/support_faq.txt", "w") as f:
        f.write("""FREQUENTLY ASKED QUESTIONS

Q: How do I install ACME Widget?
A: Download the installer from our website, run it as administrator, and follow the on-screen instructions. Installation takes 5-10 minutes.

Q: What is your return policy?
A: We offer a 30-day money-back guarantee. Items must be returned in original condition. Refunds are processed within 5-7 business days.

Q: Do you offer technical support?
A: Yes! Pro customers get 24/7 phone and email support. Standard customers get email support during business hours. Lite customers have access to community forums.

Q: Can I upgrade my license later?
A: Absolutely! Contact our sales team and pay only the difference between your current license and the new one.

Q: What payment methods do you accept?
A: We accept all major credit cards, PayPal, and bank transfers for enterprise customers.

Q: Is there a mobile app?
A: Yes, our mobile app is available for iOS and Android. It includes basic monitoring and notification features.

Q: How often do you release updates?
A: We release minor updates monthly and major updates quarterly. All updates are free for active license holders.

Q: Do you offer training?
A: We provide online tutorials, documentation, and webinars. Enterprise customers can request custom training sessions.""")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    
    if REPORTLAB_AVAILABLE:
        try:
            create_company_handbook()
            create_product_catalog()
            create_support_faq()
            print("All PDF files created successfully!")
        except Exception as e:
            print(f"Error creating PDFs: {e}")
            print("Creating simple text files instead...")
            create_simple_text_files()
            print("Sample text files created successfully!")
    else:
        print("reportlab not installed. Creating simple text files instead...")
        create_simple_text_files()
        print("Sample text files created successfully!")