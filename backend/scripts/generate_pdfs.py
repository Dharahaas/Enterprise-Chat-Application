#!/usr/bin/env python3
"""Generate real PDF documents for the Enterprise Knowledge Assistant sample docs.

Produces 11 PDFs in backend/data/sample_docs/{hr,it,policy}/ with proper
layout, headings, bullet lists, and page numbers — replacing the placeholder
.txt files with realistic-looking corporate policy documents.
"""

import os
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, ListFlowable, ListItem
)

BASE = Path(__file__).resolve().parent.parent / "data" / "sample_docs"

PRIMARY = HexColor("#6366F1")
ACCENT = HexColor("#06B6D4")
TEXT = HexColor("#0F172A")
MUTED = HexColor("#64748B")
LIGHT = HexColor("#F1F5F9")
HR_COLOR = HexColor("#EC4899")
IT_COLOR = HexColor("#6366F1")
POLICY_COLOR = HexColor("#14B8A6")

DOCUMENTS = [
    {
        "agent": "hr",
        "name": "HR_Policy_Manual_2024.pdf",
        "title": "HR Policy Manual 2024",
        "subtitle": "Comprehensive guide to leave, payroll, and benefits",
        "color": HR_COLOR,
        "version": "v3.2",
        "sections": [
            {"title": "Section 1: Leave Policy", "body": [
                "Employees are entitled to the following leaves per calendar year. All leave types are tracked separately and balance is visible on the HR Portal dashboard.",
                "Casual Leave (CL): 12 days per year for personal errands and short absences.",
                "Sick Leave (SL): 12 days per year for medical reasons including doctor appointments.",
                "Earned Leave (EL): 15 days per year for planned vacations and extended time off.",
            ]},
            {"title": "Leave Carry-Forward Policy", "body": [
                "Up to 45 days of earned leave can be carried forward to the next calendar year.",
                "Casual leaves cannot be carried forward and lapse at year-end.",
                "Sick leaves cannot be carried forward and lapse at year-end.",
                "Encashment: Earned leaves above 30 days can be encashed on the employee's birthday month at 1x basic daily rate.",
            ]},
            {"title": "Leave Application Process", "body": [
                "Apply through the HR Portal under Leaves -> Apply New Leave.",
                "Manager approval is required for casual and earned leaves.",
                "Sick leaves up to 2 days do not require prior approval.",
                "Medical certificate is mandatory for sick leaves of 3 or more consecutive days.",
                "Leaves cannot be applied retrospectively beyond 7 days without HR Business Partner approval.",
            ]},
            {"title": "Section 2: Payroll", "body": [
                "Payroll cycle: Monthly, with cut-off on the 20th of every month.",
                "Salary credits are processed on the last working day of the month.",
                "Attendance regularization requests must be submitted by the 18th.",
                "Payslips are available on the HR Portal by the 1st of the next month.",
            ]},
            {"title": "Reimbursement Claims", "body": [
                "Submit through HR Portal -> Reimbursements -> New Claim.",
                "Attach original bills (PDF or image, max 5MB per file).",
                "Manager approval is required for claims above Rs 5,000.",
                "Claims submitted before cut-off are processed in the same month.",
                "Claims after cut-off are processed in the next month's payroll.",
            ]},
        ],
    },
    {
        "agent": "hr",
        "name": "Leave_Policy.pdf",
        "title": "Leave Policy (Detailed)",
        "subtitle": "Detailed provisions for casual, sick, and earned leaves",
        "color": HR_COLOR,
        "version": "v2.1",
        "sections": [
            {"title": "3.1 Casual Leave (CL)", "body": [
                "12 days per calendar year.",
                "Maximum 3 consecutive days at a time.",
                "Cannot be combined with earned leave in a single application.",
                "Manager approval required (auto-approval for 1 day if balance > 5).",
                "Cannot be carried forward — lapses at year-end.",
            ]},
            {"title": "3.2 Sick Leave (SL)", "body": [
                "12 days per calendar year.",
                "Medical certificate required for 3+ consecutive days.",
                "Up to 2 days: self-certification accepted.",
                "Cannot be carried forward.",
                "Maternity/Paternity leaves are separate from sick leaves (see Section 5).",
            ]},
            {"title": "3.3 Earned Leave (EL)", "body": [
                "15 days per calendar year (accrued at 1.25 days/month).",
                "Can be carried forward (max 45 days balance).",
                "Encashment allowed on resignation at 1x basic daily rate.",
                "Requires 7 days advance notice for leaves > 3 days.",
                "Earned leave cannot be availed during notice period.",
            ]},
        ],
    },
    {
        "agent": "hr",
        "name": "Payroll_Procedure.pdf",
        "title": "Payroll Procedure",
        "subtitle": "Standard operating procedure for monthly payroll processing",
        "color": HR_COLOR,
        "version": "v1.4",
        "sections": [
            {"title": "4.1 Payroll Cycle", "body": [
                "Cut-off: 20th of every month.",
                "Credit date: Last working day of the month.",
                "Payslips: Available on the HR Portal by the 1st of next month.",
                "Form 16: Issued annually by June 15th for the previous financial year.",
            ]},
            {"title": "4.2 Reimbursement Claim Submission", "body": [
                "Submit through HR Portal -> Reimbursements -> New Claim.",
                "Attach original bills (PDF or image, max 5MB per file).",
                "Manager approval is required for claims above Rs 5,000.",
                "Claims submitted before cut-off are processed in the same month.",
                "Claims after cut-off are processed in the next month.",
            ]},
            {"title": "4.3 Statutory Deductions", "body": [
                "PF: 12% of basic (employee) + 12% (employer).",
                "Professional Tax: As per state slab (Rs 200/month typical).",
                "TDS: Computed based on declared investment proofs submitted by Jan 31st.",
                "ESI: Applicable for employees with gross salary <= Rs 21,000/month.",
            ]},
        ],
    },
    {
        "agent": "hr",
        "name": "Onboarding_Checklist.pdf",
        "title": "Onboarding Checklist for New Hires",
        "subtitle": "Step-by-step guide for the first 30 days at the company",
        "color": HR_COLOR,
        "version": "v2.0",
        "sections": [
            {"title": "Required Documents from New Hire", "body": [
                "Signed offer letter (counter-sign and return).",
                "PAN card copy (self-attested).",
                "Aadhaar card copy (self-attested).",
                "Educational certificates (10th, 12th, degree).",
                "Previous employment relieving letter.",
                "Bank account details (cancelled cheque or passbook copy).",
                "Two passport-size photographs.",
                "Form 12B (if joining with previous PF balance).",
            ]},
            {"title": "Day 1: IT and Workspace Setup", "body": [
                "Photo for ID card and HR Portal.",
                "Issuance of corporate laptop and accessories.",
                "Email account and Active Directory credentials setup.",
                "VPN client installation and 2FA enrollment.",
                "HR induction session (90 minutes).",
            ]},
            {"title": "Day 2-5: Department Onboarding", "body": [
                "Department orientation and team introductions.",
                "Mentor assignment (buddy program).",
                "Mandatory compliance trainings:",
                "  - Information Security Policy v4.0",
                "  - Code of Conduct 2024",
                "  - POSH (Prevention of Sexual Harassment)",
                "  - Data Privacy and GDPR basics.",
            ]},
            {"title": "Day 30: Confirmation Review", "body": [
                "Performance review with manager.",
                "Goals and OKRs setting for next quarter.",
                "HR feedback survey (anonymous).",
                "Confirmation of probation completion.",
            ]},
        ],
    },
    {
        "agent": "it",
        "name": "IT_SOP_PasswordReset.pdf",
        "title": "IT SOP: Password Reset",
        "subtitle": "Standard operating procedure for corporate password reset",
        "color": IT_COLOR,
        "version": "v1.3",
        "sections": [
            {"title": "Password Reset Procedure", "body": [
                "Step 1: Go to https://password.company.internal",
                "Step 2: Click 'Forgot Password' on the top right.",
                "Step 3: Enter your employee ID or registered email address.",
                "Step 4: Choose reset method: SMS OTP or Email OTP.",
                "Step 5: Enter the 6-digit OTP received (valid for 5 minutes).",
                "Step 6: Set a new password meeting the policy below.",
            ]},
            {"title": "Password Policy Requirements", "body": [
                "Minimum 12 characters in length.",
                "At least 1 uppercase letter (A-Z).",
                "At least 1 lowercase letter (a-z).",
                "At least 1 digit (0-9).",
                "At least 1 special character (!@#$%^&*).",
                "Cannot reuse last 5 passwords.",
                "Must be changed every 90 days (forced).",
            ]},
            {"title": "Post-Reset Steps", "body": [
                "The new password syncs to Active Directory within 5 minutes.",
                "Restart your machine to refresh Kerberos tickets.",
                "Update saved passwords in browsers and email clients.",
                "If using Outlook desktop, re-enter password when prompted.",
            ]},
            {"title": "Troubleshooting", "body": [
                "If OTP not received within 2 minutes: check spam folder, then contact IT desk.",
                "If account is locked after 5 failed attempts: wait 30 minutes or call IT desk.",
                "IT desk contact: ext. 4500 or it-support@company.com.",
                "Available Monday-Friday, 8 AM to 8 PM IST.",
            ]},
        ],
    },
    {
        "agent": "it",
        "name": "IT_Network_Guide_v2.pdf",
        "title": "IT Network Guide v2",
        "subtitle": "Corporate network access, VPN configuration, and Wi-Fi setup",
        "color": IT_COLOR,
        "version": "v2.0",
        "sections": [
            {"title": "Section 2: VPN Configuration", "body": [
                "macOS setup:",
                "1. Install Cisco AnyConnect Secure Mobility Client from the IT Self-Service Portal.",
                "2. Open AnyConnect from Applications.",
                "3. Enter vpn.company.com as the host address.",
                "4. Sign in with corporate credentials (employee_id@company.com).",
                "5. Enter 2FA token from Microsoft Authenticator app.",
                "6. Once connected, you will receive a 10.x.x.x internal IP address.",
            ]},
            {"title": "Windows Setup", "body": [
                "Same Cisco AnyConnect client as macOS.",
                "Auto-launches on startup if configured in Settings.",
                "Login credentials are cached for 24 hours.",
                "Re-authentication required after sleep/wake.",
            ]},
            {"title": "Linux Setup", "body": [
                "Use openconnect CLI: sudo openconnect vpn.company.com",
                "Provide corporate credentials when prompted.",
                "2FA token required after password.",
                "For Ubuntu/Debian, install via: sudo apt install openconnect.",
            ]},
            {"title": "Troubleshooting", "body": [
                "If MFA fails: re-register Microsoft Authenticator via IT Portal.",
                "If connection drops: check firewall allows UDP 443 and UDP 500.",
                "If DNS resolution fails: flush DNS (ipconfig /flushdns on Windows).",
                "Contact IT desk: ext. 4500 or it@company.com.",
            ]},
        ],
    },
    {
        "agent": "it",
        "name": "Asset_Policy.pdf",
        "title": "Asset Policy",
        "subtitle": "Hardware allocation, upgrades, damage, and return procedures",
        "color": IT_COLOR,
        "version": "v1.5",
        "sections": [
            {"title": "7.3 Hardware Upgrade Requests", "body": [
                "Raise request via IT Service Portal: Request -> Hardware -> Upgrade.",
                "Approval chain: Manager -> IT Asset Manager -> Procurement (if cost > Rs 50,000).",
                "Standard turnaround: 5 business days from approval.",
                "Eligibility: laptop age > 3 years OR documented performance SLA breach.",
                "For RAM/SSD upgrades: approval only from IT Asset Manager.",
            ]},
            {"title": "7.4 Damage and Loss", "body": [
                "Accidental damage: employee responsibility up to Rs 10,000 deductible.",
                "Theft: FIR copy required for waiver; report within 48 hours.",
                "Negligence (e.g., liquid spill): full recovery at current book value.",
                "Loss in transit (during business travel): covered by corporate insurance.",
                "Damage report must be filed via IT Portal within 24 hours of incident.",
            ]},
            {"title": "7.5 Return on Exit", "body": [
                "All company assets must be returned on or before last working day.",
                "Includes: laptop, charger, mouse, keyboard, monitor, phone (if issued).",
                "IT clearance certificate required for full and final settlement.",
                "Unreturned assets are deducted from final settlement at replacement cost.",
                "Data wiping: performed by IT after asset return; certificate issued on request.",
            ]},
        ],
    },
    {
        "agent": "policy",
        "name": "InfoSec_Policy_v4.pdf",
        "title": "Information Security Policy v4.0",
        "subtitle": "Data classification, handling, and protection standards",
        "color": POLICY_COLOR,
        "version": "v4.0",
        "sections": [
            {"title": "Section 2: Data Classification", "body": [
                "All company data is classified into four levels. Each level has specific handling, storage, and transmission requirements.",
            ]},
            {"title": "Level 1 - Public", "body": [
                "Examples: Marketing materials, website content, published whitepapers.",
                "Handling: No restrictions.",
                "Can be shared externally without approval.",
                "Storage: Any corporate system; cloud storage allowed.",
                "Encryption: Optional.",
            ]},
            {"title": "Level 2 - Internal", "body": [
                "Examples: Process documents, org charts, internal announcements.",
                "Handling: For employees and contractors only.",
                "Must not be shared externally without manager approval.",
                "Storage: Corporate systems only; no personal cloud storage.",
                "Encryption: Recommended at rest.",
            ]},
            {"title": "Level 3 - Confidential", "body": [
                "Examples: Employee data, financials, customer PII, source code.",
                "Handling: Need-to-know basis only.",
                "Encrypted at rest and in transit (TLS 1.2+).",
                "External sharing requires DPO (Data Protection Officer) approval.",
                "Storage: Encrypted corporate systems; access logged.",
            ]},
            {"title": "Level 4 - Restricted", "body": [
                "Examples: Trade secrets, credentials, private keys, board minutes.",
                "Handling: Strictly need-to-know; all access logged.",
                "Never shared externally.",
                "Storage: Hardware security modules (HSMs) or approved vaults.",
                "Violation is grounds for termination and possible legal action.",
            ]},
        ],
    },
    {
        "agent": "policy",
        "name": "Code_of_Conduct_2024.pdf",
        "title": "Code of Conduct 2024",
        "subtitle": "Ethical standards, gift acceptance, and conflict of interest",
        "color": POLICY_COLOR,
        "version": "v2024.1",
        "sections": [
            {"title": "6.4 Gift Acceptance Policy", "body": [
                "Employees may accept business gifts up to Rs 2,500 in value per occasion.",
                "Gifts above this threshold must be declined and reported to Compliance within 7 days.",
                "Cash or cash-equivalent gifts (vouchers, gift cards) are strictly prohibited regardless of value.",
                "Gifts from vendors during active procurement evaluations are prohibited.",
                "Gifts from government officials require prior Legal approval.",
                "All accepted gifts above Rs 500 must be disclosed in the annual compliance declaration.",
            ]},
            {"title": "6.5 Conflict of Interest", "body": [
                "Disclose any external employment, board positions, or material financial interests.",
                "Update disclosure annually and on material change (within 30 days).",
                "Recuse yourself from decisions involving family members or close associates.",
                "Failure to disclose is a disciplinary offense, up to termination.",
                "Disclosures reviewed by Compliance + Legal; conflicts reported to Audit Committee.",
            ]},
            {"title": "6.6 Anti-Bribery and Corruption", "body": [
                "Company prohibits all forms of bribery and corruption.",
                "No employee may offer, give, or accept anything of value to influence business decisions.",
                "Violation may result in termination and criminal prosecution under applicable laws.",
                "Concerns can be reported anonymously to ethics@company.com or via the whistleblower hotline.",
            ]},
        ],
    },
    {
        "agent": "policy",
        "name": "Hybrid_Work_Policy_2024.pdf",
        "title": "Hybrid Work Policy 2024",
        "subtitle": "Eligibility, work locations, and compliance requirements for remote work",
        "color": POLICY_COLOR,
        "version": "v2024.2",
        "sections": [
            {"title": "Eligibility", "body": [
                "Roles graded L4 and below are eligible for up to 2 days/week remote work.",
                "Subject to manager approval and team coverage requirements.",
                "Roles requiring physical access (lab, data center, facilities) are excluded.",
                "New hires become eligible after completion of 90-day probation.",
                "Performance rating must be 'Meets Expectations' or higher in last cycle.",
            ]},
            {"title": "Approved Work Locations", "body": [
                "Remote work must be from the registered work location.",
                "International remote work requires additional Legal and Tax approval (min 2 weeks notice).",
                "Working from public networks (cafes, airports) is discouraged for Confidential data.",
                "Working from co-working spaces is allowed with prior IT approval for VPN configuration.",
            ]},
            {"title": "Equipment", "body": [
                "Company laptop with VPN configured is mandatory.",
                "Personal devices may be used only with MDM enrollment.",
                "Monitor and peripherals provided on request (one-time).",
                "Internet reimbursement: up to Rs 1,500/month on submission of bills.",
                "Home office setup allowance: Rs 15,000 one-time (after 60 days of hybrid work).",
            ]},
            {"title": "Compliance", "body": [
                "Maintain core hours: 10 AM - 4 PM IST.",
                "Attend all scheduled meetings from a quiet, professional environment.",
                "Adhere to InfoSec Policy v4 at all times.",
                "Webcam must be on for client-facing meetings.",
                "Non-compliance may result in revocation of hybrid work privileges.",
            ]},
        ],
    },
    {
        "agent": "policy",
        "name": "Security_Incident_Response.pdf",
        "title": "Security Incident Response Procedure",
        "subtitle": "Reporting and response procedures for security incidents",
        "color": POLICY_COLOR,
        "version": "v1.2",
        "sections": [
            {"title": "Reporting a Phishing Email", "body": [
                "Step 1: Do NOT click any links in the email.",
                "Step 2: Use the 'Report Phishing' button in Outlook (toolbar).",
                "Alternatively: forward the email to phishing@company.internal.",
                "Step 3: Delete the email from your inbox.",
                "Step 4: If you already clicked a link or entered credentials:",
                "  a. Immediately reset your password (see IT_SOP_PasswordReset.pdf).",
                "  b. Call the IT Security desk at ext. 4700.",
                "  c. Disconnect from the corporate network until cleared.",
            ]},
            {"title": "Response SLAs", "body": [
                "SOC (Security Operations Center) acknowledgement: within 1 business hour.",
                "Initial assessment: within 4 business hours.",
                "Containment: within 24 hours for Severity 1 incidents.",
                "Eradication and recovery: 24-72 hours depending on severity.",
                "Post-incident review: within 5 business days of closure.",
            ]},
            {"title": "Other Incident Types", "body": [
                "Lost or stolen device: Report immediately to security@company.internal.",
                "Unauthorized access to systems: Report to SOC via ext. 4700.",
                "Data leak or breach: Report to DPO immediately at dpo@company.com.",
                "Do NOT attempt to investigate or remediate on your own.",
                "All incidents are logged and reviewed by the CISO monthly.",
            ]},
            {"title": "Severity Classification", "body": [
                "Severity 1: Customer data breach, ransomware, total system outage.",
                "Severity 2: Internal data leak, partial system outage, malware detection.",
                "Severity 3: Phishing email, policy violation, suspicious activity.",
                "Severity 4: General security inquiries, training requests.",
            ]},
        ],
    },
]


def build_styles(color):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="DocTitle", parent=styles["Title"], fontSize=24, leading=28, textColor=color, spaceAfter=6, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="DocSubtitle", parent=styles["Normal"], fontSize=12, leading=16, textColor=MUTED, spaceAfter=24, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="SectionHeading", parent=styles["Heading2"], fontSize=14, leading=18, textColor=color, spaceBefore=18, spaceAfter=8))
    styles.add(ParagraphStyle(name="BodyBullet", parent=styles["Normal"], fontSize=10.5, leading=15, textColor=TEXT, leftIndent=20, bulletIndent=8, spaceAfter=4))
    styles.add(ParagraphStyle(name="CoverHeading", parent=styles["Title"], fontSize=36, leading=42, textColor=color, spaceAfter=12, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="CoverSub", parent=styles["Normal"], fontSize=14, leading=20, textColor=MUTED, spaceAfter=24, alignment=TA_LEFT))
    return styles


def header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(PRIMARY)
    canvas.rect(0, height - 4, width, 4, fill=1, stroke=0)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(2*cm, 1.2*cm, "Enterprise Knowledge Assistant · Confidential")
    canvas.drawRightString(width - 2*cm, 1.2*cm, f"Page {doc.page}")
    canvas.setStrokeColor(LIGHT)
    canvas.line(2*cm, 1.5*cm, width - 2*cm, 1.5*cm)
    canvas.restoreState()


def build_doc(doc_meta):
    out_path = BASE / doc_meta["agent"] / doc_meta["name"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out_path), pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2.5*cm, bottomMargin=2*cm, title=doc_meta["title"], author="Enterprise Knowledge Assistant")
    styles = build_styles(doc_meta["color"])
    story = []
    story.append(Spacer(1, 6*cm))
    story.append(Paragraph(doc_meta["title"], styles["CoverHeading"]))
    story.append(Paragraph(doc_meta["subtitle"], styles["CoverSub"]))
    meta_data = [["Version", doc_meta["version"]], ["Document Type", "Corporate Policy"], ["Classification", "Internal"], ["Owner", "Enterprise Knowledge Assistant"], ["Effective Date", "January 2026"]]
    meta_table = Table(meta_data, colWidths=[4*cm, 8*cm])
    meta_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), LIGHT), ("TEXTCOLOR", (0, 0), (0, -1), MUTED), ("TEXTCOLOR", (1, 0), (1, -1), TEXT), ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 9), ("PADDING", (0, 0), (-1, -1), 8), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LINEBELOW", (0, 0), (-1, -1), 0.5, LIGHT)]))
    story.append(meta_table)
    story.append(PageBreak())
    for section in doc_meta["sections"]:
        story.append(Paragraph(section["title"], styles["SectionHeading"]))
        items = [Paragraph(line, styles["BodyBullet"]) for line in section["body"]]
        story.append(ListFlowable([ListItem(item, leftIndent=10) for item in items], bulletType="bullet", bulletColor=doc_meta["color"], leftIndent=12))
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    return out_path


def main():
    print(f"Generating {len(DOCUMENTS)} PDF documents in {BASE}\n")
    for doc_meta in DOCUMENTS:
        path = build_doc(doc_meta)
        size_kb = path.stat().st_size / 1024
        print(f"  ✓ {doc_meta['agent']:6} | {doc_meta['name']:42} | {size_kb:.1f} KB")
    print(f"\n✓ Generated {len(DOCUMENTS)} PDFs")


if __name__ == "__main__":
    main()
