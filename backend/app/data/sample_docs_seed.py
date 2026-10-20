"""Sample documents used to seed the vector DB on first run.

These are synthetic policies that mirror the file names referenced in the
Phase 1 frontend mock data. They give the chat agents something real to
retrieve from on a fresh install.
"""

SAMPLE_DOCS: list[dict] = [
    # -------------------- HR --------------------
    {
        "name": "HR_Policy_Manual_2024.pdf",
        "agent": "hr",
        "section": "1",
        "content": (
            "HR POLICY MANUAL 2024\n\n"
            "Section 1: Leave Policy\n"
            "Employees are entitled to the following leaves per calendar year:\n"
            "- 12 Casual Leaves (CL)\n"
            "- 12 Sick Leaves (SL)\n"
            "- 15 Earned Leaves (EL)\n\n"
            "Leave carry-forward: Up to 45 days of earned leave can be carried forward to the next year. "
            "Casual and sick leaves cannot be carried forward and lapse at year-end.\n\n"
            "Leave application process: Apply through the HR Portal under Leaves -> Apply. "
            "Manager approval is required for casual and earned leaves. "
            "Sick leaves up to 2 days do not require approval, but a medical certificate "
            "must be attached for sick leaves of 3 or more consecutive days.\n\n"
            "Section 2: Payroll\n"
            "Payroll cycle: Monthly, with cut-off on the 20th of every month. "
            "Salary credits are processed on the last working day of the month. "
            "Attendance regularization requests must be submitted by the 18th.\n"
            "Reimbursement claims with bills must be submitted before the cut-off to be "
            "included in the same month's payroll."
        ),
    },
    {
        "name": "Leave_Policy.docx",
        "agent": "hr",
        "section": "3.1",
        "content": (
            "LEAVE POLICY (detailed)\n\n"
            "3.1 Casual Leave (CL)\n"
            "- 12 days per calendar year\n"
            "- Maximum 3 consecutive days at a time\n"
            "- Cannot be combined with earned leave in a single application\n"
            "- Manager approval required\n\n"
            "3.2 Sick Leave (SL)\n"
            "- 12 days per calendar year\n"
            "- Medical certificate required for 3+ consecutive days\n"
            "- Up to 2 days: self-certification accepted\n"
            "- Cannot be carried forward\n\n"
            "3.3 Earned Leave (EL)\n"
            "- 15 days per calendar year\n"
            "- Can be carried forward (max 45 days balance)\n"
            "- Encashment allowed on resignation at 1x basic daily rate\n"
            "- Requires 7 days advance notice"
        ),
    },
    {
        "name": "Payroll_Procedure.docx",
        "agent": "hr",
        "section": "4.2",
        "content": (
            "PAYROLL PROCEDURE\n\n"
            "4.1 Payroll Cycle\n"
            "- Cut-off: 20th of every month\n"
            "- Credit date: Last working day of the month\n"
            "- Payslips: Available on the HR Portal by the 1st of next month\n\n"
            "4.2 Reimbursement Claim Submission\n"
            "- Submit through HR Portal -> Reimbursements -> New Claim\n"
            "- Attach original bills (PDF or image, max 5MB per file)\n"
            "- Manager approval required for claims above Rs 5,000\n"
            "- Claims submitted before cut-off are processed in the same month\n"
            "- Claims after cut-off are processed in the next month\n\n"
            "4.3 Statutory Deductions\n"
            "- PF: 12% of basic (employee) + 12% (employer)\n"
            "- Professional Tax: As per state slab\n"
            "- TDS: Computed based on declared investment proofs"
        ),
    },
    {
        "name": "Onboarding_Checklist.pdf",
        "agent": "hr",
        "section": "1",
        "content": (
            "ONBOARDING CHECKLIST FOR NEW HIRES\n\n"
            "Required documents from new hire:\n"
            "1. Signed offer letter (counter-sign and return)\n"
            "2. PAN card copy\n"
            "3. Aadhaar card copy\n"
            "4. Educational certificates (10th, 12th, degree)\n"
            "5. Previous employment relieving letter\n"
            "6. Bank account details (cancelled cheque or passbook copy)\n"
            "7. Two passport-size photographs\n"
            "8. Form 12B (if joining with previous PF balance)\n\n"
            "Day 1: IT and workspace setup, ID card issuance, HR induction\n"
            "Day 2: Department orientation, mentor assignment\n"
            "Day 3-5: Mandatory compliance trainings (InfoSec, Code of Conduct, POSH)\n"
            "Day 30: Confirmation review with manager"
        ),
    },

    # -------------------- IT --------------------
    {
        "name": "IT_SOP_PasswordReset.pdf",
        "agent": "it",
        "section": "1",
        "content": (
            "IT STANDARD OPERATING PROCEDURE: PASSWORD RESET\n\n"
            "Step 1: Go to https://password.company.internal\n"
            "Step 2: Click 'Forgot Password'\n"
            "Step 3: Enter your employee ID or email\n"
            "Step 4: Choose reset method: SMS OTP or Email OTP\n"
            "Step 5: Enter the 6-digit OTP received\n"
            "Step 6: Set a new password meeting the policy:\n"
            "  - Minimum 12 characters\n"
            "  - At least 1 uppercase letter\n"
            "  - At least 1 number\n"
            "  - At least 1 special character\n"
            "  - Cannot reuse last 5 passwords\n\n"
            "The new password syncs to Active Directory within 5 minutes. "
            "If you do not receive the OTP within 2 minutes, contact the IT desk at ext. 4500."
        ),
    },
    {
        "name": "IT_Network_Guide_v2.pdf",
        "agent": "it",
        "section": "2",
        "content": (
            "IT NETWORK GUIDE v2\n\n"
            "Section 2: VPN Configuration\n\n"
            "macOS:\n"
            "1. Install Cisco AnyConnect Secure Mobility Client from the IT Self-Service Portal\n"
            "2. Open AnyConnect\n"
            "3. Enter vpn.company.com as the host\n"
            "4. Sign in with corporate credentials (employee_id@company.com)\n"
            "5. Enter 2FA token from Microsoft Authenticator\n"
            "6. Once connected, you will receive a 10.x.x.x internal IP\n\n"
            "Windows:\n"
            "1. Same client, same flow\n"
            "2. Auto-launches on startup if configured\n\n"
            "Linux:\n"
            "1. Use openconnect CLI: sudo openconnect vpn.company.com\n"
            "2. Provide credentials when prompted\n\n"
            "Troubleshooting:\n"
            "- If MFA fails: re-register Microsoft Authenticator via IT Portal\n"
            "- If connection drops: check firewall allows UDP 443 and 500\n"
            "- Contact IT desk: ext. 4500 or it@company.com"
        ),
    },
    {
        "name": "Asset_Policy.docx",
        "agent": "it",
        "section": "7.3",
        "content": (
            "ASSET POLICY\n\n"
            "7.3 Hardware Upgrade Requests\n"
            "- Raise request via IT Service Portal: Request -> Hardware -> Upgrade\n"
            "- Approval chain: Manager -> IT Asset Manager -> Procurement (if cost > Rs 50,000)\n"
            "- Standard turnaround: 5 business days from approval\n"
            "- Eligibility: laptop age > 3 years OR documented performance SLA breach\n\n"
            "7.4 Damage and Loss\n"
            "- Accidental damage: employee responsibility up to Rs 10,000 deductible\n"
            "- Theft: FIR copy required for waiver\n"
            "- Negligence: full recovery at current book value\n\n"
            "7.5 Return on Exit\n"
            "- All company assets must be returned on or before last working day\n"
            "- IT clearance certificate required for full and final settlement"
        ),
    },

    # -------------------- Policy --------------------
    {
        "name": "InfoSec_Policy_v4.pdf",
        "agent": "policy",
        "section": "2",
        "content": (
            "INFORMATION SECURITY POLICY v4.0\n\n"
            "Section 2: Data Classification\n\n"
            "Level 1 - Public\n"
            "- Marketing materials, website content, published whitepapers\n"
            "- No handling restrictions\n"
            "- Can be shared externally without approval\n\n"
            "Level 2 - Internal\n"
            "- Process documents, org charts, internal announcements\n"
            "- For employees and contractors only\n"
            "- Must not be shared externally without manager approval\n\n"
            "Level 3 - Confidential\n"
            "- Employee data, financials, customer PII, source code\n"
            "- Need-to-know basis only\n"
            "- Encrypted at rest and in transit\n"
            "- External sharing requires DPO approval\n\n"
            "Level 4 - Restricted\n"
            "- Trade secrets, credentials, private keys, board minutes\n"
            "- Strictly need-to-know, logged access\n"
            "- Never shared externally\n"
            "- Violation is grounds for termination"
        ),
    },
    {
        "name": "Code_of_Conduct_2024.pdf",
        "agent": "policy",
        "section": "6.4",
        "content": (
            "CODE OF CONDUCT 2024\n\n"
            "6.4 Gift Acceptance Policy\n"
            "- Employees may accept business gifts up to Rs 2,500 in value per occasion\n"
            "- Gifts above this threshold must be declined and reported to Compliance within 7 days\n"
            "- Cash or cash-equivalent gifts (vouchers, gift cards) are strictly prohibited regardless of value\n"
            "- Gifts from vendors during active procurement evaluations are prohibited\n"
            "- Gifts from government officials require prior Legal approval\n\n"
            "6.5 Conflict of Interest\n"
            "- Disclose any external employment, board positions, or material financial interests\n"
            "- Update disclosure annually and on material change\n"
            "- Failure to disclose is a disciplinary offense"
        ),
    },
    {
        "name": "Hybrid_Work_Policy_2024.pdf",
        "agent": "policy",
        "section": "1",
        "content": (
            "HYBRID WORK POLICY 2024\n\n"
            "Eligibility:\n"
            "- Roles graded L4 and below are eligible for up to 2 days/week remote work\n"
            "- Subject to manager approval and team coverage requirements\n"
            "- Roles requiring physical access (lab, data center, facilities) are excluded\n\n"
            "Approved work locations:\n"
            "- Remote work must be from the registered work location\n"
            "- International remote work requires additional Legal and Tax approval (min 2 weeks)\n"
            "- Working from public networks (cafes, airports) is discouraged for Confidential data\n\n"
            "Equipment:\n"
            "- Company laptop with VPN configured is mandatory\n"
            "- Personal devices may be used only with MDM enrollment\n"
            "- Monitor and peripherals provided on request (one-time)\n\n"
            "Compliance:\n"
            "- Maintain core hours: 10 AM - 4 PM IST\n"
            "- Attend all scheduled meetings from a quiet, professional environment\n"
            "- Adhere to InfoSec Policy v4 at all times"
        ),
    },
    {
        "name": "Security_Incident_Response.pdf",
        "agent": "policy",
        "section": "1",
        "content": (
            "SECURITY INCIDENT RESPONSE PROCEDURE\n\n"
            "Reporting a phishing email:\n"
            "1. Do NOT click any links in the email\n"
            "2. Use the 'Report Phishing' button in Outlook (toolbar)\n"
            "   OR forward to phishing@company.internal\n"
            "3. Delete the email from your inbox\n"
            "4. If you already clicked a link or entered credentials:\n"
            "   a. Immediately reset your password (see IT_SOP_PasswordReset.pdf)\n"
            "   b. Call the IT Security desk at ext. 4700\n"
            "   c. Disconnect from the corporate network until cleared\n\n"
            "Response SLAs:\n"
            "- SOC acknowledgement: within 1 business hour\n"
            "- Initial assessment: within 4 business hours\n"
            "- Containment: within 24 hours for Severity 1 incidents\n\n"
            "Other incident types (lost device, unauthorized access, data leak):\n"
            "- Report immediately to security@company.internal\n"
            "- Do not attempt to investigate or remediate on your own"
        ),
    },
]
